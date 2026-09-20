"""Convert a HF Whisper checkpoint (hub id or local dir) to CTranslate2 float16."""

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF hub id or local HF model dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quantization", default="float16")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    cmd = [
        "ct2-transformers-converter",
        "--model", args.model,
        "--output_dir", str(args.out),
        "--quantization", args.quantization,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    size = sum(p.stat().st_size for p in args.out.rglob("*")) / 1e9
    print(f"converted to {args.out} ({size:.2f} GB)")


if __name__ == "__main__":
    main()
