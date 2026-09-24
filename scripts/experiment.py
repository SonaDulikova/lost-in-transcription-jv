"""The one command the autoresearch agent runs. See experiments/autoresearch/program.md."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from lit.autoresearch import (
    DEFAULT_DECODE, DEFAULT_MODEL, REPO, Store, atomic_write, gpu_busy, run_decode, run_postproc, run_train, table_md,
)

# argparse keeps the LAST occurrence of a repeated flag, so anything the agent passes after `--`
# would silently override these runner-owned flags (see train_cmd) unless refused up front.
FORBIDDEN_TRAIN_FLAGS = ("--out", "--val", "--run-name", "--wandb-project")


def _rules(s: str) -> list[str]:
    return [r for r in s.split(",") if r]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("postproc", help="re-score an existing predictions CSV under a rule set (seconds)")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--rules", type=_rules, default=["diacritics"], help="comma-separated, applied in order")
    p.add_argument("--hypothesis", default=None)

    d = sub.add_parser("decode", help="transcribe convo 2 with a CT2 model and decode params (~5 min)")
    d.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    d.add_argument("--language", default=DEFAULT_DECODE["language"], choices=["id", "jw", "auto"])
    d.add_argument("--beam", type=int, default=DEFAULT_DECODE["beam"])
    d.add_argument("--temperature", type=float, default=DEFAULT_DECODE["temperature"])
    d.add_argument("--patience", type=float, default=None)
    d.add_argument("--condition-on-previous-text", action="store_true")
    d.add_argument("--vad-filter", action="store_true")
    d.add_argument("--rules", type=_rules, default=["diacritics"])
    d.add_argument("--hypothesis", default=None)

    t = sub.add_parser("train", help="train_lora -> export_ct2 -> decode convo 2 -> score; extra flags go to train_lora.py")
    mode = t.add_mutually_exclusive_group(required=True)
    mode.add_argument("--proxy", action="store_const", const="proxy", dest="mode")
    mode.add_argument("--full", action="store_const", const="full", dest="mode")
    mode.add_argument("--dry", action="store_const", const="dry", dest="mode", help="20 chunks, 0.05 epoch: plumbing check")
    t.add_argument("--hypothesis", default=None)

    n = sub.add_parser("note", help="fill in the conclusion of a finished run")
    n.add_argument("--id", required=True)
    n.add_argument("--conclusion", required=True)

    sub.add_parser("table", help="regenerate experiments/autoresearch/results.md")

    s = sub.add_parser("session", help="start a new night")
    s.add_argument("--reset", action="store_true", required=True)
    s.add_argument("--max-hours", type=float, default=12.0)
    s.add_argument("--max-runs", type=int, default=40)

    args, extra = ap.parse_known_args(argv)
    if extra and extra[0] == "--":  # argparse may hand the separator back; train_lora must not see it
        extra = extra[1:]
    if extra and args.cmd != "train":
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    store = Store()

    if args.cmd == "session":
        sess = store.reset_session(args.max_hours, args.max_runs)
        print(f"session started {sess['started']}: max {sess['max_hours']:g} h, {sess['max_runs']} runs")
        return 0
    if args.cmd == "note":
        store.note(args.id, args.conclusion)
        return 0
    if args.cmd == "table":
        atomic_write(store.table, table_md(store.rows()))
        print(f"wrote {store.table}")
        return 0

    reason = store.refusal(args.cmd, gpu_busy=gpu_busy() if args.cmd == "train" else False,
                            mode=getattr(args, "mode", None))
    if reason:
        print(f"refused: {reason}")
        return 2

    if args.cmd == "postproc":
        run_postproc(store, args.predictions, args.rules, args.hypothesis)
    elif args.cmd == "decode":
        cfg = {"language": args.language, "beam": args.beam, "temperature": args.temperature,
               "patience": args.patience, "condition_on_previous_text": args.condition_on_previous_text,
               "vad_filter": args.vad_filter}
        run_decode(store, args.model, cfg, args.rules, args.hypothesis)
    else:
        bad = next((f for f in FORBIDDEN_TRAIN_FLAGS
                    if any(x == f or x.startswith(f + "=") for x in extra)), None)
        if bad:
            print(f"refused: {bad} is set by the runner and cannot be passed through")
            return 2
        free_gb = shutil.disk_usage(REPO / "runs").free / 1e9
        if args.mode == "full" and free_gb < 45:
            print(f"refused: only {free_gb:.0f} GB free under runs/, a full run needs ~45 GB")
            return 2
        elif args.mode != "full" and free_gb < 20:
            print(f"warning: {free_gb:.0f} GB free under runs/")
        run_train(store, extra, args.mode, args.hypothesis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
