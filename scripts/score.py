"""Score a predictions CSV (audio_filename,transcript) against the dev ground truth."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from lit.data import DEV_DIR
from lit.wer import grouped_wer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--truth", type=Path, default=DEV_DIR / "ground_truth.csv")
    ap.add_argument("--meta", type=Path, default=DEV_DIR / "metadata.tsv")
    ap.add_argument("--convo", type=int, default=None, help="score only this convo_id")
    args = ap.parse_args()

    truth = pd.read_csv(args.truth, keep_default_na=False)
    meta = pd.read_csv(args.meta, sep="\t", keep_default_na=False)
    pred = pd.read_csv(args.predictions, keep_default_na=False)

    df = truth.merge(meta[["audio_filename", "language", "convo_id"]], on="audio_filename")
    if args.convo is not None:
        df = df[df["convo_id"] == args.convo]
    missing = set(df["audio_filename"]) - set(pred["audio_filename"])
    if missing:
        print(f"ERROR: predictions missing {len(missing)} rows, e.g. {sorted(missing)[0]}")
        sys.exit(1)
    pred = pred.set_index("audio_filename").loc[df["audio_filename"]]

    res = grouped_wer(df["transcript"].tolist(), pred["transcript"].tolist(), df["language"].tolist())
    for group, r in res.items():
        print(f"{group:>7}: {r.row()}")


if __name__ == "__main__":
    main()
