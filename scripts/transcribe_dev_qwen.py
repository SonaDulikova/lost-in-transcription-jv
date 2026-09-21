"""Zero-shot Qwen3-ASR over the dev set (bake-off only; not used in the submission yet)."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from lit.data import DEV_DIR, load_dev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--language", default="Indonesian", help="Indonesian or auto")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from qwen_asr import Qwen3ASRModel  # qwen-asr 0.0.6: from_pretrained forwards kwargs to AutoModel

    model = Qwen3ASRModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=256, max_inference_batch_size=1
    )
    df = load_dev(DEV_DIR)
    if args.limit:
        df = df.head(args.limit)
    lang = None if args.language == "auto" else args.language

    t0 = time.time()
    hyps = []
    for p in tqdm(df["path"].tolist()):
        res = model.transcribe(audio=str(p), language=lang)  # returns List[ASRTranscription] with .text
        hyps.append(res[0].text.strip())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"audio_filename": df["audio_filename"], "transcript": hyps}).to_csv(args.out, index=False)
    print(json.dumps({"model": args.model, "language": args.language, "n": len(df), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
