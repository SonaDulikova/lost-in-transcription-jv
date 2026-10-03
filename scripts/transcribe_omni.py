"""Zero-shot Omnilingual ASR (Meta omniASR) on the convo-2 dev clips, written as a predictions CSV.

Runs under the conda env that has fairseq2 (torch 2.8), not the uv env:
    ~/miniconda3/envs/slovak_llm_audio_env/bin/python scripts/transcribe_omni.py --card omniASR_CTC_1B_v2
    ... --card omniASR_LLM_1B_v2 --lang jav_Latn
Then score with: uv run scripts/experiment.py postproc --predictions predictions/omni_<card>.csv --rules diacritics,tail
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="omniASR_CTC_1B_v2")
    ap.add_argument("--lang", default=None, help="e.g. jav_Latn or ind_Latn; ignored by CTC models")
    ap.add_argument("--manifest", type=Path, default=REPO / "data/dev_segments/convo2.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    rows = list(csv.DictReader(open(args.manifest, newline="")))[: args.limit]
    paths = [r["path"] for r in rows]
    tag = args.card + (f"_{args.lang}" if args.lang else "")
    out = args.out or REPO / "predictions" / f"omni_{tag}.csv"

    t0 = time.time()
    pipe = ASRInferencePipeline(model_card=args.card, device=args.device, dtype=getattr(torch, args.dtype))
    load_s = time.time() - t0
    print(f"loaded {args.card} on {args.device} {args.dtype} in {load_s:.0f} s", file=sys.stderr)

    t0 = time.time()
    hyps = []
    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i:i + args.batch_size]
        lang = [args.lang] * len(chunk) if args.lang else None
        hyps += pipe.transcribe(chunk, lang=lang, batch_size=args.batch_size)
        print(f"{i + len(chunk)}/{len(paths)}", file=sys.stderr, end="\r")
    dec_s = time.time() - t0

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for r, h in zip(rows, hyps):
            w.writerow([Path(r["path"]).stem + ".mp3", h])
    meta = {"card": args.card, "lang": args.lang, "device": args.device, "dtype": args.dtype, "n": len(rows),
            "load_s": round(load_s, 1), "decode_s": round(dec_s, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if args.device == "cuda" else None}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {out}: {len(hyps)} rows, decode {dec_s:.0f} s, {meta['peak_vram_gb']} GB peak", file=sys.stderr)
    for h in hyps[:3]:
        print("  ", h[:160], file=sys.stderr)


if __name__ == "__main__":
    main()
