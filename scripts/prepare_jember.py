"""Slice Jember sessions into <= max-seconds chunks at 16 kHz mono and write train/val manifests."""

import argparse
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from lit.data import JEMBER_DIR, REPO, chunk_session, load_jember_tsv, split_sessions

SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=JEMBER_DIR / "Jember Javanese Spontaneous Speech Corpus - 1-200.tsv")
    ap.add_argument("--audio-dir", type=Path, default=JEMBER_DIR / "mp3 audio")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "jember_segments")
    ap.add_argument("--max-seconds", type=float, default=25.0)
    ap.add_argument("--val-every", type=int, default=13)
    args = ap.parse_args()

    df = load_jember_tsv(args.tsv)
    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for session, group in tqdm(df.groupby("session"), desc="sessions"):
        mp3 = args.audio_dir / f"{session}.mp3"
        if not mp3.exists():
            print(f"missing audio for session {session}, skipping")
            continue
        y, _ = librosa.load(mp3, sr=SR, mono=True)
        for i, c in enumerate(chunk_session(group, args.max_seconds)):
            a, b = int(c["start"] * SR), min(int(c["end"] * SR), len(y))
            if b - a < SR // 2:
                continue
            out = wav_dir / f"{session}_{i:04d}.wav"
            sf.write(out, y[a:b], SR)
            rows.append({"path": str(out), "text": c["text"], "session": session, "duration": (b - a) / SR})

    man = pd.DataFrame(rows)
    train_s, val_s = split_sessions(sorted(man["session"].unique(), key=int), args.val_every)
    man[man["session"].isin(train_s)].to_csv(args.out / "train.csv", index=False)
    man[man["session"].isin(val_s)].to_csv(args.out / "val.csv", index=False)
    print(f"{len(man)} chunks, {man['duration'].sum() / 3600:.2f} h; "
          f"train sessions {len(train_s)}, val sessions {len(val_s)}; "
          f"mean chunk {man['duration'].mean():.1f}s, max {man['duration'].max():.1f}s")


if __name__ == "__main__":
    main()
