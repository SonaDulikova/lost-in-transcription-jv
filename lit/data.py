from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEV_DIR = REPO / "data" / "indonesian_dev"
JEMBER_DIR = REPO / "data" / "jember"


def load_dev(dev_dir: Path = DEV_DIR) -> pd.DataFrame:
    df = pd.read_csv(dev_dir / "metadata.tsv", sep="\t", dtype={"speaker": str}, keep_default_na=False)
    df["convo_id"] = df["convo_id"].astype(int)
    df["path"] = [dev_dir / "clips" / f for f in df["audio_filename"]]
    missing = [p.name for p in df["path"] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} dev clips missing, first: {missing[0]}")
    return df[["audio_filename", "speaker", "transcript", "language", "convo_id", "path"]]
