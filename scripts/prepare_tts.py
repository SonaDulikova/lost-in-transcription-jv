"""Convert a read-speech TTS corpus (WEBM/MP3/WAV + TSV) to 16 kHz wavs, pack consecutive sentences into
<= max-seconds chunks, and write a manifest (path,text,session,duration) for train_lora.py."""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from lit.data import REPO, group_by_duration, load_tts_tsv

SR = 16000


def to_wav(src: Path, dst: Path) -> bool:
    if dst.exists():
        return True
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)]
    return subprocess.run(cmd, check=False).returncode == 0 and dst.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--file-col", required=True, help="TSV column with the audio file name (with or without extension)")
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--ext", default=".webm", help="appended when the file column has no extension")
    ap.add_argument("--session", default="cjv", help="session label written to the manifest")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "central_jv")
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--gap", type=float, default=0.25, help="silence inserted between packed sentences")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = load_tts_tsv(args.tsv, args.file_col, args.text_col)
    raw_dir, wav_dir = args.out / "wav_raw", args.out / "wav"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    srcs = []
    for f in df["file"]:
        p = args.audio_dir / f
        srcs.append(p if p.suffix else p.with_suffix(args.ext))
    dsts = [raw_dir / (p.stem + ".wav") for p in srcs]
    with ThreadPoolExecutor(args.workers) as ex:
        ok = list(tqdm(ex.map(to_wav, srcs, dsts), total=len(srcs), desc="ffmpeg"))
    mask = pd.Series(ok, index=df.index)
    df = df[mask].reset_index(drop=True)
    dsts = [d for d, o in zip(dsts, ok) if o]
    print(f"decoded {len(dsts)} of {len(ok)} files")

    durations = [sf.info(d).duration for d in dsts]
    silence = np.zeros(int(args.gap * SR), dtype="float32")
    rows = []
    for gi, idx in enumerate(tqdm(group_by_duration(durations, args.max_seconds, args.gap), desc="pack")):
        parts = []
        for j in idx:
            y, _ = sf.read(dsts[j], dtype="float32")
            parts.append(y if y.ndim == 1 else y.mean(axis=1))
            parts.append(silence)
        y = np.concatenate(parts[:-1])
        out = wav_dir / f"{args.session}_{gi:05d}.wav"
        sf.write(out, y, SR)
        rows.append({"path": str(out), "text": " ".join(df["text"].iloc[j] for j in idx),
                     "session": args.session, "duration": len(y) / SR})

    man = pd.DataFrame(rows)
    man.to_csv(args.out / "train.csv", index=False)
    print(f"{len(man)} chunks, {man['duration'].sum() / 3600:.2f} h; "
          f"mean chunk {man['duration'].mean():.1f}s, max {man['duration'].max():.1f}s; "
          f"sentences per chunk {len(df) / len(man):.1f}")


if __name__ == "__main__":
    main()
