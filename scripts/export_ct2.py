"""Merge a LoRA adapter into its base Whisper model and convert to CTranslate2."""

import argparse
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="CT2 output dir (e.g. submission_src/model)")
    ap.add_argument("--quantization", default="float16")
    args = ap.parse_args()

    merged_dir = args.adapter.parent / "merged"
    base = WhisperForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    model.generation_config.forced_decoder_ids = None
    model.save_pretrained(merged_dir, safe_serialization=True)
    WhisperProcessor.from_pretrained(args.base).save_pretrained(merged_dir)
    print(f"merged model saved to {merged_dir}")

    subprocess.run([
        "uv", "run", "scripts/fetch_model.py", str(merged_dir), "--out", str(args.out),
        "--quantization", args.quantization,
    ], check=True)


if __name__ == "__main__":
    main()
