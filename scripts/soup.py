"""Average several LoRA adapters' merged deltas into their base Whisper model and convert to CTranslate2.

    uv run scripts/soup.py runs/t018/adapter runs/t019/adapter runs/t020/adapter --out runs/ct2/soup_allscope
    uv run scripts/soup.py runs/t020/checkpoint-252 runs/t020/checkpoint-378 --out runs/ct2/t020_ep23
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from lit.soup import add_soup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", type=Path, nargs="+", help="adapter/ or checkpoint-*/ dirs trained from --base")
    ap.add_argument("--base", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--weights", type=float, nargs="+", default=None, help="relative weights, normalised to sum 1")
    ap.add_argument("--scale", type=float, default=1.0, help="lambda on the averaged delta; 1.0 = plain soup")
    ap.add_argument("--out", type=Path, required=True, help="CT2 output dir")
    ap.add_argument("--quantization", default="float16")
    ap.add_argument("--keep-merged", action="store_true", help="keep the fp32 HF model next to --out")
    args = ap.parse_args()

    for a in args.adapters:
        trained_on = json.loads((a / "adapter_config.json").read_text())["base_model_name_or_path"]
        if trained_on != args.base:
            raise SystemExit(f"{a} was trained on {trained_on}, not {args.base}")

    model = WhisperForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float32)
    add_soup(model, args.adapters, weights=args.weights, scale=args.scale)
    model.generation_config.forced_decoder_ids = None
    merged_dir = args.out.with_name(args.out.name + "_merged")
    model.save_pretrained(merged_dir, safe_serialization=True)
    WhisperProcessor.from_pretrained(args.base).save_pretrained(merged_dir)
    del model
    print(f"souped {len(args.adapters)} adapters (scale {args.scale}) into {merged_dir}")

    subprocess.run([
        "uv", "run", "scripts/fetch_model.py", str(merged_dir), "--out", str(args.out),
        "--quantization", args.quantization,
    ], check=True)
    if not args.keep_merged:
        shutil.rmtree(merged_dir)
    (args.out / "soup.json").write_text(json.dumps({
        "base": args.base, "adapters": [str(a) for a in args.adapters],
        "weights": args.weights, "scale": args.scale,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
