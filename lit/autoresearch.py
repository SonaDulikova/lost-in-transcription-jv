"""Runner logic for the unattended experiment loop (see experiments/autoresearch/program.md)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from lit.data import DEV_DIR, REPO
from lit.postproc import apply
from lit.wer import grouped_wer

ROOT = REPO / "experiments" / "autoresearch"
KIND_PREFIX = {"postproc": "p", "decode": "d", "train": "t"}
METRIC_KEYS = ("wer_all", "wer_ind", "wer_javind", "S", "D", "I")

BUDGET_S = {"decode": 600, "proxy": 2700, "full": 18000}
DEFAULT_BASE = "openai/whisper-large-v3-turbo"
DEFAULT_MODEL = REPO / "runs" / "ct2" / "lora_v5"
DEFAULT_DECODE = {"language": "id", "beam": 5, "temperature": 0.0, "patience": None,
                  "condition_on_previous_text": False, "vad_filter": False}
UV = ["uv", "run"]

PROVENANCE_FILES = ("lit/postproc.py", "scripts/train_lora.py", "lit/autoresearch.py")


def atomic_write(path: Path, text: str) -> None:
    """Write text to a sibling temp file and os.replace() it onto the target, so an interruption
    mid-write cannot leave a truncated file behind (results.jsonl and results.md are rewritten
    wholesale on every `note`/`table` call, ~40x/night)."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def provenance(store: "Store", run_id: str) -> dict:
    """Identify the source state a row was produced with: commit plus a hash per editable file.

    The agent edits these files between runs, so two rows with the same config can still be
    different experiments; the hashes are what makes rows comparable after the fact.
    """
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO)
    src = {}
    for rel in PROVENANCE_FILES:
        p = REPO / rel
        src[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else None
    diff = subprocess.run(["git", "diff", "--", *PROVENANCE_FILES], capture_output=True, text=True, cwd=REPO)
    if diff.returncode == 0 and diff.stdout.strip():
        store.logs.mkdir(parents=True, exist_ok=True)
        (store.logs / f"{run_id}.diff").write_text(diff.stdout)
    return {"head": head.stdout.strip() if head.returncode == 0 else None, "src": src}


def make_row(run_id: str, kind: str, config: dict, hypothesis: str | None, status: str, runtime_s: float,
             metrics: dict | None = None, error: str | None = None) -> dict:
    row = {"id": run_id, "kind": kind, "ts": datetime.now().isoformat(timespec="seconds"), "status": status,
           "config": config, **{k: None for k in METRIC_KEYS},
           "runtime_s": round(runtime_s, 1), "hypothesis": hypothesis, "conclusion": None}
    if metrics:
        row.update(metrics)
    if error:
        row["error"] = error
    return row


@dataclass
class Store:
    root: Path = ROOT

    @property
    def results(self) -> Path:
        return self.root / "results.jsonl"

    @property
    def table(self) -> Path:
        return self.root / "results.md"

    @property
    def session(self) -> Path:
        return self.root / "session.json"

    @property
    def stop(self) -> Path:
        return self.root / "STOP"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def candidates(self) -> Path:
        return self.root / "candidates.md"

    def rows(self) -> list[dict]:
        if not self.results.exists():
            return []
        return [json.loads(line) for line in self.results.read_text().splitlines() if line.strip()]

    def next_id(self, kind: str) -> str:
        prefix = KIND_PREFIX[kind]
        nums = [int(r["id"][1:]) for r in self.rows() if r["id"].startswith(prefix)]
        return f"{prefix}{max(nums, default=0) + 1:03d}"

    def append(self, row: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.results, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def note(self, run_id: str, conclusion: str) -> None:
        rows = self.rows()
        hits = [r for r in rows if r["id"] == run_id]
        if not hits:
            raise KeyError(run_id)
        hits[0]["conclusion"] = conclusion
        atomic_write(self.results, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    def reset_session(self, max_hours: float = 12.0, max_runs: int = 40) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        # rows are counted from this offset, not by timestamp, so a reset in the same second as a run is unambiguous
        s = {"started": datetime.now().isoformat(timespec="seconds"), "max_hours": max_hours, "max_runs": max_runs,
             "rows_at_start": len(self.rows())}
        self.session.write_text(json.dumps(s, indent=2) + "\n")
        return s

    def refusal(self, kind: str, now: datetime | None = None, gpu_busy: bool = False,
                mode: str | None = None) -> str | None:
        if self.stop.exists():
            return "STOP file present"
        if not self.session.exists():
            return "no session: run `session --reset`"
        s = json.loads(self.session.read_text())
        now = now or datetime.now()
        started = datetime.fromisoformat(s["started"])
        if now - started > timedelta(hours=s["max_hours"]):
            return f"session exhausted: {s['max_hours']:g} h elapsed, run `session --reset`"
        done = len(self.rows()) - s["rows_at_start"]
        if done >= s["max_runs"]:
            return f"session exhausted: {s['max_runs']} runs, run `session --reset`"
        if kind == "train" and gpu_busy:
            return "a scripts/train_lora.py process is running"
        if kind == "train" and mode == "full":
            remaining = s["max_hours"] * 3600 - (now - started).total_seconds()
            if remaining < BUDGET_S["full"]:
                return f"only {remaining/3600:.1f} h left in the session; a full run needs {BUDGET_S['full']/3600:.0f} h"
        return None


def _cell(v) -> str:
    return str(v if v is not None else "").replace("|", "\\|")


def _wer(v) -> str:
    return f"{v:.4f}" if v is not None else "-"


def table_md(rows: list[dict]) -> str:
    ok = [r for r in rows if r["status"] == "ok" and r["wer_all"] is not None]
    lines = ["# Autoresearch results", "",
             "Metric: convo-2 corpus WER, lower is better. Regenerated by `uv run scripts/experiment.py table`.", "",
             "## Best per kind", "", "| kind | id | WER all | ind | javind | config |", "|---|---|---|---|---|---|"]
    for kind in KIND_PREFIX:
        best = min((r for r in ok if r["kind"] == kind), key=lambda r: r["wer_all"], default=None)
        if best:
            lines.append(f"| {kind} | {best['id']} | {_wer(best['wer_all'])} | {_wer(best['wer_ind'])} | "
                         f"{_wer(best['wer_javind'])} | `{_cell(json.dumps(best['config']))}` |")
    lines += ["", "## All runs", "",
              "| id | ts | status | WER all | ind | javind | S/D/I | runtime | config | hypothesis | conclusion |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["kind"], r["wer_all"] if r["wer_all"] is not None else 9.0, r["id"])):
        sdi = f"{r['S']}/{r['D']}/{r['I']}" if r["S"] is not None else "-"
        lines.append(f"| {r['id']} | {r['ts']} | {r['status']} | {_wer(r['wer_all'])} | {_wer(r['wer_ind'])} | "
                     f"{_wer(r['wer_javind'])} | {sdi} | {r['runtime_s']:.0f}s | `{_cell(json.dumps(r['config']))}` | "
                     f"{_cell(r['hypothesis'])} | {_cell(r['conclusion'])} |")
    return "\n".join(lines) + "\n"


def score_convo(predictions_csv: Path, rules: Sequence[str], convo: int = 2, dev_dir: Path | None = None) -> dict:
    dev_dir = dev_dir or DEV_DIR  # resolved at call time so tests can monkeypatch lit.autoresearch.DEV_DIR
    truth = pd.read_csv(dev_dir / "ground_truth.csv", keep_default_na=False)
    meta = pd.read_csv(dev_dir / "metadata.tsv", sep="\t", keep_default_na=False)
    df = truth.merge(meta[["audio_filename", "language", "convo_id"]], on="audio_filename")
    df = df[df["convo_id"].astype(int) == convo]
    pred = pd.read_csv(predictions_csv, keep_default_na=False)
    missing = set(df["audio_filename"]) - set(pred["audio_filename"])
    if missing:
        raise ValueError(f"predictions missing {len(missing)} rows, e.g. {sorted(missing)[0]}")
    hyps = pred.set_index("audio_filename").loc[df["audio_filename"], "transcript"].tolist()
    hyps = [apply(h, rules) for h in hyps]
    res = grouped_wer(df["transcript"].tolist(), hyps, df["language"].tolist())
    a = res["all"]
    return {"wer_all": round(a.wer, 4), "wer_ind": round(res["ind"].wer, 4), "wer_javind": round(res["javind"].wer, 4),
            "S": a.substitutions, "D": a.deletions, "I": a.insertions}


def run_logged(cmd: list[str], log_path: Path, budget_s: float) -> tuple[str, int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=REPO)
        try:
            rc = proc.wait(timeout=budget_s)
            status = "ok" if rc == 0 else "crash"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            rc, status = -1, "timeout"
    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
    return status, rc, tail


def gpu_busy() -> bool:
    return subprocess.run(["pgrep", "-f", "scripts/train_lora.py"], capture_output=True).returncode == 0


def _opt(args: list[str], flag: str, default: str) -> str:
    if flag in args and args.index(flag) + 1 < len(args):
        return args[args.index(flag) + 1]
    return default


def decode_cmd(model: Path, out_csv: Path, cfg: dict) -> list[str]:
    cmd = UV + ["scripts/transcribe_dev.py", "--model", str(model), "--convo", "2", "--out", str(out_csv),
                "--no-postprocess", "--language", cfg["language"], "--beam", str(cfg["beam"])]
    if cfg.get("temperature") is not None:
        cmd += ["--temperature", str(cfg["temperature"])]
    if cfg.get("patience") is not None:
        cmd += ["--patience", str(cfg["patience"])]
    if cfg.get("condition_on_previous_text"):
        cmd.append("--condition-on-previous-text")
    if cfg.get("vad_filter"):
        cmd.append("--vad-filter")
    return cmd


def train_cmd(run_id: str, train_args: list[str], out: Path, proxy: bool, dry: bool) -> list[str]:
    cmd = UV + ["scripts/train_lora.py", "--out", str(out), "--val", "data/dev_segments/convo2.csv",
                "--language", "id", *train_args]
    if proxy:
        cmd += ["--epochs", "1", "--train-limit", "500", "--val-limit", "40", "--no-wandb", "--run-name", run_id]
    else:
        cmd += ["--val-limit", "78", "--run-name", run_id]
    if dry:
        cmd += ["--epochs", "0.05", "--train-limit", "20", "--val-limit", "4", "--no-wandb"]
    return cmd


def export_cmd(train_args: list[str], adapter: Path, out: Path) -> list[str]:
    return UV + ["scripts/export_ct2.py", "--base", _opt(train_args, "--base", DEFAULT_BASE),
                 "--adapter", str(adapter), "--out", str(out)]


def _finish(store: Store, row: dict) -> dict:
    store.append(row)
    print(json.dumps(row, ensure_ascii=False))
    return row


def run_postproc(store: Store, predictions: Path, rules: Sequence[str], hypothesis: str | None) -> dict:
    run_id = store.next_id("postproc")
    config = {"predictions": str(predictions), "rules": list(rules), "provenance": provenance(store, run_id)}
    t0 = time.time()
    try:
        metrics = score_convo(predictions, rules)
        pred = pd.read_csv(predictions, keep_default_na=False)
        pred["transcript"] = pred["transcript"].map(lambda t: apply(t, rules))
        store.predictions.mkdir(parents=True, exist_ok=True)
        pred.to_csv(store.predictions / f"{run_id}.csv", index=False)
        row = make_row(run_id, "postproc", config, hypothesis, "ok", time.time() - t0, metrics)
    except Exception as e:  # noqa: BLE001 - a bad rule set is a data point, not a crash of the loop
        row = make_row(run_id, "postproc", config, hypothesis, "crash", time.time() - t0, error=f"{type(e).__name__}: {e}")
    return _finish(store, row)


def _score_or_error(csv: Path, rules: Sequence[str]) -> tuple[dict | None, str | None]:
    try:
        return score_convo(csv, rules), None
    except Exception as e:  # noqa: BLE001
        return None, f"scoring: {type(e).__name__}: {e}"


def _run_or_error(runner, cmd, log, budget) -> tuple[str, str | None]:
    try:
        status, _rc, tail = runner(cmd, log, budget)
        return status, (tail if status != "ok" else None)
    except Exception as e:  # noqa: BLE001 - a runner that cannot start is a crash row, not a lost run
        return "crash", f"{type(e).__name__}: {e}"


def run_decode(store: Store, model: Path, cfg: dict, rules: Sequence[str], hypothesis: str | None,
               runner=run_logged) -> dict:
    run_id = store.next_id("decode")
    config = {"model": str(model), **cfg, "rules": list(rules), "provenance": provenance(store, run_id)}
    out_csv = store.predictions / f"{run_id}.csv"
    t0 = time.time()
    status, tail = _run_or_error(runner, decode_cmd(model, out_csv, cfg), store.logs / f"{run_id}.log", BUDGET_S["decode"])
    metrics, error = (None, tail[-2000:]) if status != "ok" else _score_or_error(out_csv, rules)
    if error and status == "ok":
        status = "crash"
    return _finish(store, make_row(run_id, "decode", config, hypothesis, status, time.time() - t0, metrics, error))


def run_train(store: Store, train_args: list[str], mode: str, hypothesis: str | None,
              runner=run_logged, runs_dir: Path = REPO / "runs") -> dict:
    run_id = store.next_id("train")
    rules = ["diacritics"]
    config = {"mode": mode, "train_args": list(train_args), "decode": DEFAULT_DECODE, "rules": rules,
              "provenance": provenance(store, run_id)}
    out, ct2 = runs_dir / run_id, runs_dir / "ct2" / run_id
    out_csv = store.predictions / f"{run_id}.csv"
    log = store.logs / f"{run_id}.log"
    budget = BUDGET_S["full" if mode == "full" else "proxy"]
    t0 = time.time()
    steps = [
        ("train_lora", train_cmd(run_id, train_args, out, proxy=mode != "full", dry=mode == "dry")),
        ("export_ct2", export_cmd(train_args, out / "adapter", ct2)),
        ("transcribe_dev", decode_cmd(ct2, out_csv, DEFAULT_DECODE)),
    ]
    status, error, metrics = "ok", None, None
    for name, cmd in steps:
        remaining = max(budget - (time.time() - t0), 1.0)
        status, tail = _run_or_error(runner, cmd, log, remaining)
        if name == "export_ct2" and status == "ok":
            # the fp32 merged model is fully reproducible from the adapter; keeping it around
            # (~3.2 GB) is the difference between one and two full runs fitting on disk overnight
            shutil.rmtree(out / "merged", ignore_errors=True)
        if status != "ok":
            error = f"{name} {status}: {tail[-2000:]}"
            break
    if status == "ok":
        metrics, error = _score_or_error(out_csv, rules)
        if error:
            status = "crash"
    if mode != "full":
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(ct2, ignore_errors=True)
    return _finish(store, make_row(run_id, "train", config, hypothesis, status, time.time() - t0, metrics, error))
