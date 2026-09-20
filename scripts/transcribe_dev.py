"""Run the submission transcriber (CT2) or a HF checkpoint over the dev set."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lit.data import DEV_DIR, load_dev

REPO = Path(__file__).resolve().parents[1]


def load_main():
    spec = importlib.util.spec_from_file_location("sub_main", REPO / "submission_src" / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sub_main"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="CT2 dir, or HF dir with --backend hf")
    ap.add_argument("--backend", choices=["ct2", "hf"], default="ct2")
    ap.add_argument("--language", default="id", help="id, jw, or 'auto'")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--convo", type=int, default=None)
    ap.add_argument("--dev-dir", type=Path, default=DEV_DIR)
    args = ap.parse_args()

    df = load_dev(args.dev_dir)
    if args.convo is not None:
        df = df[df["convo_id"] == args.convo]
    if args.limit:
        df = df.head(args.limit)
    language = None if args.language == "auto" else args.language

    if args.backend == "ct2":
        m = load_main()
        cfg = m.load_config()
        cfg.update({"language": language, "beam_size": args.beam})
        transcribe = m.build_transcriber(args.model, cfg)
    else:
        from lit.hf_infer import build_hf_transcriber

        transcribe = build_hf_transcriber(args.model, language=language or "id", beam_size=args.beam)

    t0 = time.time()
    hyps = [transcribe(p) for p in tqdm(df["path"].tolist())]
    elapsed = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"audio_filename": df["audio_filename"], "transcript": hyps}).to_csv(args.out, index=False)
    meta = {"model": str(args.model), "backend": args.backend, "language": args.language,
            "beam": args.beam, "n": len(df), "seconds": round(elapsed, 1)}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
