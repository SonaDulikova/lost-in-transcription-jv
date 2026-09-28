import hashlib
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lit.autoresearch import (
    DEFAULT_DECODE, PROVENANCE_FILES, Store, decode_cmd, export_cmd, make_row, provenance, run_logged,
    score_convo, table_md, train_cmd,
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "autoresearch")


def test_ids_are_per_kind_and_zero_padded(store: Store):
    assert store.next_id("decode") == "d001"
    store.append(make_row("d001", "decode", {}, "h", "ok", 1.0))
    store.append(make_row("d007", "decode", {}, "h", "ok", 1.0))
    store.append(make_row("t002", "train", {}, "h", "crash", 1.0))
    assert store.next_id("decode") == "d008"
    assert store.next_id("train") == "t003"
    assert store.next_id("postproc") == "p001"


def test_make_row_shape():
    row = make_row("p001", "postproc", {"rules": ["case"]}, "why", "ok", 2.34,
                   metrics={"wer_all": 0.2, "wer_ind": 0.1, "wer_javind": 0.3, "S": 1, "D": 2, "I": 3})
    assert row["id"] == "p001" and row["status"] == "ok" and row["runtime_s"] == 2.3
    assert row["wer_all"] == 0.2 and row["S"] == 1 and row["conclusion"] is None
    assert "error" not in row
    crashed = make_row("t001", "train", {}, None, "crash", 5.0, error="boom")
    assert crashed["wer_all"] is None and crashed["error"] == "boom"


def test_note_only_changes_conclusion(store: Store):
    store.append(make_row("d001", "decode", {"beam": 5}, "h", "ok", 1.0))
    store.append(make_row("d002", "decode", {"beam": 8}, "h", "ok", 1.0))
    store.note("d001", "beam 5 is fine")
    rows = store.rows()
    assert rows[0]["conclusion"] == "beam 5 is fine" and rows[0]["config"] == {"beam": 5}
    assert rows[1]["conclusion"] is None
    with pytest.raises(KeyError):
        store.note("d999", "x")


def test_refusal_reasons(store: Store):
    assert store.refusal("decode") == "no session: run `session --reset`"
    s = store.reset_session(max_hours=12, max_runs=2)
    assert store.refusal("decode") is None
    assert store.refusal("train", gpu_busy=True) == "a scripts/train_lora.py process is running"
    assert store.refusal("decode", gpu_busy=True) is None
    late = datetime.fromisoformat(s["started"]) + timedelta(hours=12, minutes=1)
    assert "12 h elapsed" in store.refusal("decode", now=late)
    store.append(make_row("d001", "decode", {}, "h", "ok", 1.0))
    store.append(make_row("d002", "decode", {}, "h", "crash", 1.0))
    assert "2 runs" in store.refusal("decode")
    store.stop.touch()
    assert store.refusal("postproc") == "STOP file present"


def test_reset_session_starts_a_fresh_count(store: Store):
    store.reset_session(max_runs=1)
    store.append(make_row("d001", "decode", {}, "h", "ok", 1.0))
    assert store.refusal("decode") is not None
    store.reset_session(max_runs=1)
    assert store.refusal("decode") is None


def test_table_lists_best_per_kind_and_escapes_pipes():
    rows = [
        make_row("d001", "decode", {"beam": 5}, "a | b", "ok", 300,
                 metrics={"wer_all": 0.19, "wer_ind": 0.13, "wer_javind": 0.2, "S": 1, "D": 1, "I": 1}),
        make_row("d002", "decode", {"beam": 1}, "h", "ok", 200,
                 metrics={"wer_all": 0.18, "wer_ind": 0.13, "wer_javind": 0.2, "S": 1, "D": 1, "I": 1}),
        make_row("t001", "train", {"lr": 1e-4}, "h", "timeout", 2700),
    ]
    md = table_md(rows)
    assert "| decode | d002 | 0.1800 |" in md
    assert "a \\| b" in md
    assert "| t001 |" in md and "timeout" in md
    assert "| train |" not in md.split("## All runs")[0]


@pytest.fixture
def dev_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dev"
    d.mkdir()
    (d / "metadata.tsv").write_text(
        "audio_filename\tlanguage\tconvo_id\n"
        "a.mp3\tind\t2\nb.mp3\tjavind\t2\nc.mp3\tind\t5\n")
    (d / "ground_truth.csv").write_text(
        "audio_filename,transcript\na.mp3,aku iso ya\nb.mp3,neng kene\nc.mp3,ignored\n")
    return d


def test_score_convo_applies_rules_and_filters_convo(dev_dir: Path, tmp_path: Path):
    pred = tmp_path / "p.csv"
    pred.write_text("audio_filename,transcript\nc.mp3,wrong wrong wrong\na.mp3,aku iso ya\nb.mp3,nèng kene. Terima kasih.\n")
    raw = score_convo(pred, [], dev_dir=dev_dir)
    assert raw["wer_ind"] == 0.0 and raw["wer_javind"] > 0
    clean = score_convo(pred, ["diacritics", "tail"], dev_dir=dev_dir)
    assert clean == {"wer_all": 0.0, "wer_ind": 0.0, "wer_javind": 0.0, "S": 0, "D": 0, "I": 0}


def test_score_convo_rejects_missing_rows(dev_dir: Path, tmp_path: Path):
    pred = tmp_path / "p.csv"
    pred.write_text("audio_filename,transcript\na.mp3,aku iso ya\n")
    with pytest.raises(ValueError, match="missing 1"):
        score_convo(pred, [], dev_dir=dev_dir)


def test_run_logged_statuses(tmp_path: Path):
    log = tmp_path / "x.log"
    assert run_logged([sys.executable, "-c", "print('hi')"], log, 10)[0] == "ok"
    assert "hi" in log.read_text()
    status, rc, tail = run_logged([sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"], log, 10)
    assert (status, rc) == ("crash", 3) and "bad" in tail
    status, _, _ = run_logged([sys.executable, "-c", "import time; time.sleep(30)"], log, 0.5)
    assert status == "timeout"


def test_decode_cmd_only_passes_set_options(tmp_path: Path):
    cmd = decode_cmd(Path("runs/ct2/lora_v2"), tmp_path / "d001.csv", DEFAULT_DECODE)
    assert cmd[:3] == ["uv", "run", "scripts/transcribe_dev.py"]
    assert "--no-postprocess" in cmd and "--convo" in cmd and "--patience" not in cmd
    assert "--vad-filter" not in cmd and "--condition-on-previous-text" not in cmd
    assert "--without-timestamps" not in cmd
    cfg = {**DEFAULT_DECODE, "beam": 8, "patience": 1.5, "vad_filter": True, "without_timestamps": True}
    cmd = decode_cmd(Path("m"), tmp_path / "d.csv", cfg)
    assert cmd[cmd.index("--beam") + 1] == "8" and cmd[cmd.index("--patience") + 1] == "1.5"
    assert "--vad-filter" in cmd and "--without-timestamps" in cmd


def test_train_cmd_proxy_overrides_come_last():
    args = ["--train", "data/jember_segments/train.csv", "--epochs", "3", "--lr", "5e-5"]
    cmd = train_cmd("t001", args, Path("runs/t001"), proxy=True, dry=False)
    assert cmd[:3] == ["uv", "run", "scripts/train_lora.py"]
    assert cmd[cmd.index("--val") + 1] == "data/dev_segments/convo2.csv"
    # argparse keeps the last value, so the proxy overrides must follow the agent's flags
    assert cmd[cmd.index("--lr") + 2:] == ["--epochs", "1", "--train-limit", "500", "--val-limit", "40", "--no-wandb", "--run-name", "t001"]
    full = train_cmd("t002", args, Path("runs/t002"), proxy=False, dry=False)
    assert "--no-wandb" not in full and full[full.index("--val-limit") + 1] == "78"
    dry = train_cmd("t003", args, Path("runs/t003"), proxy=True, dry=True)
    assert dry[dry.index("--run-name") + 2:] == ["--epochs", "0.05", "--train-limit", "20", "--val-limit", "4", "--no-wandb"]


def test_export_cmd_uses_base_from_train_args():
    cmd = export_cmd(["--base", "openai/whisper-large-v3", "--train", "x.csv"], Path("runs/t1/adapter"), Path("runs/ct2/t1"))
    assert cmd[cmd.index("--base") + 1] == "openai/whisper-large-v3"
    cmd = export_cmd(["--train", "x.csv"], Path("a"), Path("b"))
    assert cmd[cmd.index("--base") + 1] == "openai/whisper-large-v3-turbo"


def test_export_cmd_falls_back_when_base_has_no_value():
    cmd = export_cmd(["--train", "x.csv", "--base"], Path("a"), Path("b"))
    assert cmd[cmd.index("--base") + 1] == "openai/whisper-large-v3-turbo"


import shutil

from lit.autoresearch import run_decode, run_postproc, run_train


def _fake_runner_factory(dev_dir: Path, fail_step: str | None = None):
    """Stands in for run_logged: writes a perfect predictions CSV when a decode command runs."""
    calls: list[list[str]] = []

    def runner(cmd, log_path, budget_s):
        calls.append(cmd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake log\nlast line\n")
        script = cmd[2]
        if fail_step and fail_step in script:
            return "crash", 1, "boom"
        if script == "scripts/transcribe_dev.py":
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("audio_filename,transcript\na.mp3,aku iso ya\nb.mp3,neng kene\n")
        if script == "scripts/train_lora.py":
            adapter = Path(cmd[cmd.index("--out") + 1]) / "adapter"
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"")
        if script == "scripts/export_ct2.py":
            Path(cmd[cmd.index("--out") + 1]).mkdir(parents=True, exist_ok=True)
        return "ok", 0, "last line"

    runner.calls = calls
    return runner


def test_run_postproc_scores_and_writes_csv(store: Store, dev_dir: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("lit.autoresearch.DEV_DIR", dev_dir)
    store.reset_session()
    src = tmp_path / "raw.csv"
    src.write_text("audio_filename,transcript\na.mp3,aku iso ya\nb.mp3,nèng kene\n")
    row = run_postproc(store, src, ["diacritics"], "strip accents")
    assert row["id"] == "p001" and row["status"] == "ok" and row["wer_all"] == 0.0
    config = dict(row["config"])
    prov = config.pop("provenance")
    assert config == {"predictions": str(src), "rules": ["diacritics"]}
    assert set(prov["src"]) == set(PROVENANCE_FILES)
    assert (store.predictions / "p001.csv").read_text().splitlines()[2] == "b.mp3,neng kene"
    assert store.rows()[0]["id"] == "p001"


def test_run_decode_uses_runner_and_scores(store: Store, dev_dir: Path, monkeypatch):
    monkeypatch.setattr("lit.autoresearch.DEV_DIR", dev_dir)
    store.reset_session()
    runner = _fake_runner_factory(dev_dir)
    cfg = {**DEFAULT_DECODE, "beam": 8}
    row = run_decode(store, Path("runs/ct2/lora_v2"), cfg, ["diacritics"], "wider beam", runner=runner)
    assert row["id"] == "d001" and row["status"] == "ok" and row["wer_all"] == 0.0
    assert row["config"]["beam"] == 8 and row["config"]["model"] == "runs/ct2/lora_v2"
    assert runner.calls[0][2] == "scripts/transcribe_dev.py"
    assert (store.logs / "d001.log").exists()


def test_run_decode_records_crash(store: Store, dev_dir: Path, monkeypatch):
    monkeypatch.setattr("lit.autoresearch.DEV_DIR", dev_dir)
    store.reset_session()
    runner = _fake_runner_factory(dev_dir, fail_step="transcribe_dev")
    row = run_decode(store, Path("m"), DEFAULT_DECODE, [], None, runner=runner)
    assert row["status"] == "crash" and row["wer_all"] is None and row["error"] == "boom"
    assert store.rows()[-1]["status"] == "crash"


def test_run_decode_records_runner_exception(store: Store, dev_dir: Path, monkeypatch):
    monkeypatch.setattr("lit.autoresearch.DEV_DIR", dev_dir)
    store.reset_session()

    def boom(cmd, log_path, budget_s):
        raise FileNotFoundError("uv: command not found")

    row = run_decode(store, Path("m"), DEFAULT_DECODE, [], None, runner=boom)
    assert row["status"] == "crash" and row["wer_all"] is None
    assert "FileNotFoundError" in row["error"]
    assert len(store.rows()) == 1


def test_run_train_records_runner_exception(store: Store, dev_dir: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("lit.autoresearch.DEV_DIR", dev_dir)
    store.reset_session()

    def boom(cmd, log_path, budget_s):
        raise OSError("no space left on device")

    row = run_train(store, ["--train", "x.csv"], "proxy", None, runner=boom, runs_dir=tmp_path / "runs")
    assert row["status"] == "crash" and row["error"].startswith("train_lora")
    assert "OSError" in row["error"]
    assert len(store.rows()) == 1


def test_run_train_proxy_chains_and_cleans_up(store: Store, dev_dir: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("lit.autoresearch.DEV_DIR", dev_dir)
    store.reset_session()
    runs = tmp_path / "runs"
    runner = _fake_runner_factory(dev_dir)
    row = run_train(store, ["--train", "x.csv", "--lr", "5e-5"], "proxy", "lower lr", runner=runner, runs_dir=runs)
    assert row["id"] == "t001" and row["status"] == "ok" and row["wer_all"] == 0.0
    assert [c[2] for c in runner.calls] == ["scripts/train_lora.py", "scripts/export_ct2.py", "scripts/transcribe_dev.py"]
    config = dict(row["config"])
    prov = config.pop("provenance")
    assert config == {"mode": "proxy", "train_args": ["--train", "x.csv", "--lr", "5e-5"], "decode": DEFAULT_DECODE, "rules": ["diacritics"]}
    assert set(prov["src"]) == set(PROVENANCE_FILES)
    assert not (runs / "t001").exists() and not (runs / "ct2" / "t001").exists()
    assert (store.predictions / "t001.csv").exists()


def test_run_train_full_keeps_artifacts_and_records_step_failure(store: Store, dev_dir: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("lit.autoresearch.DEV_DIR", dev_dir)
    store.reset_session()
    runs = tmp_path / "runs"
    ok = _fake_runner_factory(dev_dir)
    row = run_train(store, ["--train", "x.csv"], "full", None, runner=ok, runs_dir=runs)
    assert row["status"] == "ok" and (runs / "t001" / "adapter").exists() and (runs / "ct2" / "t001").exists()
    bad = _fake_runner_factory(dev_dir, fail_step="export_ct2")
    row = run_train(store, ["--train", "x.csv"], "proxy", None, runner=bad, runs_dir=runs)
    assert row["id"] == "t002" and row["status"] == "crash" and row["error"].startswith("export_ct2")
    assert not (runs / "t002").exists()


def test_run_train_full_deletes_merged_dir_after_export(store: Store, dev_dir: Path, tmp_path: Path, monkeypatch):
    # export_ct2.py writes the fp32 merged model to <out>/merged (adapter.parent / "merged") before
    # converting to CT2; a full run keeps <out> but the merged copy is reproducible from the adapter
    # and must not be left on disk (~3.2 GB/run).
    monkeypatch.setattr("lit.autoresearch.DEV_DIR", dev_dir)
    store.reset_session()
    runs = tmp_path / "runs"
    base = _fake_runner_factory(dev_dir)

    def runner(cmd, log_path, budget_s):
        if cmd[2] == "scripts/export_ct2.py":
            adapter = Path(cmd[cmd.index("--adapter") + 1])
            merged = adapter.parent / "merged"
            merged.mkdir(parents=True, exist_ok=True)
            (merged / "model.safetensors").write_bytes(b"x")
        return base(cmd, log_path, budget_s)

    row = run_train(store, ["--train", "x.csv"], "full", None, runner=runner, runs_dir=runs)
    assert row["status"] == "ok"
    assert (runs / "t001" / "adapter").exists()
    assert (runs / "ct2" / "t001").exists()
    assert not (runs / "t001" / "merged").exists()


def test_run_decode_caps_error_tail(store: Store, dev_dir: Path, monkeypatch):
    monkeypatch.setattr("lit.autoresearch.DEV_DIR", dev_dir)
    store.reset_session()

    def runner(cmd, log_path, budget_s):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("x" * 5000)
        return "crash", 1, "x" * 5000

    row = run_decode(store, Path("m"), DEFAULT_DECODE, [], None, runner=runner)
    assert row["status"] == "crash"
    assert len(row["error"]) == 2000


def test_note_write_is_atomic(store: Store, monkeypatch):
    store.append(make_row("d001", "decode", {}, "h", "ok", 1.0))
    original = store.results.read_text()

    def boom(*a, **k):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr("lit.autoresearch.os.replace", boom)
    with pytest.raises(OSError):
        store.note("d001", "boom")
    # the target file must be untouched by a write that never reached os.replace()
    assert store.results.read_text() == original
    monkeypatch.undo()
    store.note("d001", "ok now")
    assert store.rows()[0]["conclusion"] == "ok now"
    # no leftover temp file after a successful write
    assert list(store.root.glob("results.jsonl.tmp*")) == []


def test_refusal_full_train_needs_enough_time_left(store: Store):
    s = store.reset_session(max_hours=12, max_runs=40)
    started = datetime.fromisoformat(s["started"])
    # only 4 h remain in the 12 h session; a full run needs BUDGET_S["full"] == 5 h
    almost_out_of_time = started + timedelta(hours=8)
    reason = store.refusal("train", now=almost_out_of_time, mode="full")
    assert reason is not None and "h left in the session" in reason and "5 h" in reason
    # proxy/dry runs are unaffected by the full-run time check
    assert store.refusal("train", now=almost_out_of_time, mode="proxy") is None
    assert store.refusal("train", now=almost_out_of_time) is None
    # plenty of time left: no refusal
    plenty_of_time = started + timedelta(hours=1)
    assert store.refusal("train", now=plenty_of_time, mode="full") is None
    # existing precedence is preserved: STOP / no-session / exhausted / gpu-busy still win
    store.stop.touch()
    assert store.refusal("train", now=almost_out_of_time, mode="full") == "STOP file present"


def test_provenance_hash_changes_with_file_bytes(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    f = repo / "foo.py"
    f.write_text("a")
    subprocess.run(["git", "add", "foo.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    monkeypatch.setattr("lit.autoresearch.REPO", repo)
    monkeypatch.setattr("lit.autoresearch.PROVENANCE_FILES", ("foo.py", "missing.py"))
    store = Store(repo / "autoresearch")

    p1 = provenance(store, "t001")
    assert p1["head"] is not None
    assert p1["src"]["foo.py"] == hashlib.sha256(b"a").hexdigest()[:12]
    assert p1["src"]["missing.py"] is None
    assert not (store.logs / "t001.diff").exists()  # no uncommitted change yet

    f.write_text("b")  # uncommitted edit, as the agent would make between runs
    p2 = provenance(store, "t002")
    assert p2["src"]["foo.py"] == hashlib.sha256(b"b").hexdigest()[:12]
    assert p2["src"]["foo.py"] != p1["src"]["foo.py"]
    assert p2["head"] == p1["head"]  # HEAD unchanged; only the working-tree bytes moved
    assert (store.logs / "t002.diff").read_text().strip() != ""
