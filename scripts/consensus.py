"""Combine several predictions CSVs (audio_filename,transcript) by per-clip consensus into one CSV.

    uv run scripts/consensus.py experiments/autoresearch/predictions/t01{8,9}.csv \
        experiments/autoresearch/predictions/t020.csv --rules diacritics,tail --out predictions/cons_allscope.csv
    uv run scripts/score.py predictions/cons_allscope.csv --convo 2

The rules are applied to every candidate before the pick, and the output holds the post-processed winner, so
score it as-is. Rows follow the first CSV's order; ties go to the CSV listed first.
"""

import argparse
from pathlib import Path

import pandas as pd

from lit.consensus import pick_consensus
from lit.postproc import apply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path, nargs="+")
    ap.add_argument("--rules", default="diacritics", help="comma-separated postproc rules applied before the pick")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rules = [r for r in args.rules.split(",") if r]

    preds = [pd.read_csv(p, keep_default_na=False).set_index("audio_filename")["transcript"] for p in args.predictions]
    names = preds[0].index
    for p, s in zip(args.predictions, preds):
        if not s.index.is_unique or set(s.index) != set(names):
            raise SystemExit(f"{p} does not cover the same clips as {args.predictions[0]}")

    picked, wins = [], [0] * len(preds)
    for name in names:
        cands = [apply(s[name], rules) for s in preds]
        k = pick_consensus(cands)
        wins[k] += 1
        picked.append(cands[k])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"audio_filename": names, "transcript": picked}).to_csv(args.out, index=False)
    print(f"wrote {len(names)} rows to {args.out}; picks per input: {wins}")


if __name__ == "__main__":
    main()
