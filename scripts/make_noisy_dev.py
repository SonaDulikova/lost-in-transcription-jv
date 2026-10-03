"""Build a degraded copy of the dev set, a stand-in for harder test audio (plan §4.4 gate).

Each clip gets babble from a random Central Javanese training clip (never from the dev set) at
5-15 dB SNR, then an MP3 re-encode at 16-32 kbps. Same file names, so transcribe_dev.py --dev-dir
and score.py work unchanged.

    uv run scripts/make_noisy_dev.py --convo 2 --out data/indonesian_dev_noisy
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from lit.augment import SR, add_babble
from lit.data import DEV_DIR, load_dev


def read16k(path: Path) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32767


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", type=Path, default=DEV_DIR)
    ap.add_argument("--convo", type=int, default=2)
    ap.add_argument("--babble", type=Path, default=Path("data/central_jv/train.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dev = load_dev(args.dev_dir)
    dev = dev[dev["convo_id"] == args.convo] if args.convo else dev
    pool = pd.read_csv(args.babble, keep_default_na=False)["path"].tolist()
    (args.out / "clips").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.dev_dir / "ground_truth.csv", args.out / "ground_truth.csv")
    meta = pd.read_csv(args.dev_dir / "metadata.tsv", sep="\t", dtype=str, keep_default_na=False)
    meta[meta["audio_filename"].isin(dev["audio_filename"])].to_csv(args.out / "metadata.tsv", sep="\t", index=False)

    log = []
    for r in dev.itertuples():
        y = read16k(r.path)
        noise = read16k(Path(pool[rng.integers(len(pool))]))
        snr, kbps = float(rng.uniform(5, 15)), int(rng.choice([16, 24, 32]))
        y = add_babble(y, noise, snr)
        pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                        "-c:a", "libmp3lame", "-b:a", f"{kbps}k", str(args.out / "clips" / r.audio_filename)],
                       input=pcm, check=True)
        log.append({"audio_filename": r.audio_filename, "snr_db": round(snr, 1), "kbps": kbps})
    pd.DataFrame(log).to_csv(args.out / "degradation.csv", index=False)
    print(f"wrote {len(log)} degraded clips to {args.out} (seed {args.seed})")


if __name__ == "__main__":
    main()
