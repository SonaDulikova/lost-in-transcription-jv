"""Convert dev clips to 16 kHz wavs and write per-conversation manifests (path,text,session,duration)."""

import argparse
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from lit.data import DEV_DIR, REPO, load_dev

SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "data" / "dev_segments")
    args = ap.parse_args()

    dev = load_dev(DEV_DIR)
    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in tqdm(dev.itertuples(index=False), total=len(dev)):
        y, _ = librosa.load(r.path, sr=SR, mono=True)
        out = wav_dir / (Path(r.audio_filename).stem + ".wav")
        sf.write(out, y, SR)
        rows.append({"path": str(out), "text": r.transcript, "session": f"dev{r.convo_id}", "duration": len(y) / SR})
    man = pd.DataFrame(rows)
    for convo in sorted(dev["convo_id"].unique()):
        sub = man[man["session"] == f"dev{convo}"]
        sub.to_csv(args.out / f"convo{convo}.csv", index=False)
        print(f"convo{convo}: {len(sub)} clips, {sub['duration'].sum() / 60:.1f} min, max {sub['duration'].max():.1f}s")


if __name__ == "__main__":
    main()
