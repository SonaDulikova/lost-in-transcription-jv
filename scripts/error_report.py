"""List the worst dev clips for a predictions file, with normalized ref/hyp and S/D/I."""

import argparse
from pathlib import Path

import pandas as pd

from lit.data import DEV_DIR, load_dev
from lit.normalize import normalize_text
from lit.wer import corpus_wer, per_utterance_wer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dev = load_dev(DEV_DIR)
    pred = pd.read_csv(args.predictions, keep_default_na=False).set_index("audio_filename")
    dev["hyp"] = pred.loc[dev["audio_filename"], "transcript"].values
    dev["wer"] = per_utterance_wer(dev["transcript"].tolist(), dev["hyp"].tolist())
    dev["ref_words"] = [len(normalize_text(t).split()) for t in dev["transcript"]]
    dev["errors"] = (dev["wer"] * dev["ref_words"]).round().astype(int)
    worst = dev.sort_values("errors", ascending=False).head(args.top)

    lines = [f"# Error report for {args.predictions.name}", ""]
    for _, r in worst.iterrows():
        c = corpus_wer([r["transcript"]], [r["hyp"]])
        lines += [
            f"## {r['audio_filename']}  (lang={r['language']}, convo={r['convo_id']}, {c.row()})",
            f"REF: {normalize_text(r['transcript'])}",
            f"HYP: {normalize_text(r['hyp'])}",
            "",
        ]
    by_lang = dev.groupby("language")["wer"].mean().round(3).to_dict()
    lines += ["", f"mean per-utterance WER by language: {by_lang}"]
    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
