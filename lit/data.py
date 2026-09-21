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


def parse_hms(s: str) -> float:
    h, m, sec = s.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def load_jember_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = df.rename(columns={"Audio file name": "session"})
    df["session"] = df["session"].str.strip()
    df["start"] = df["start"].map(parse_hms)
    df["end"] = df["end"].map(parse_hms)
    df["text"] = df["text"].str.strip()
    df = df[(df["end"] > df["start"]) & (df["text"] != "")]
    return df[["session", "start", "end", "text"]].reset_index(drop=True)


def chunk_session(rows: pd.DataFrame, max_seconds: float) -> list[dict]:
    rows = rows.sort_values("start")
    chunks: list[dict] = []
    cur: dict | None = None
    for r in rows.itertuples(index=False):
        contiguous = cur is not None and abs(r.start - cur["end"]) < 1e-6
        fits = cur is not None and (r.end - cur["start"]) <= max_seconds
        if cur is not None and contiguous and fits:
            cur["end"] = r.end
            cur["text"] = f"{cur['text']} {r.text}"
        else:
            if cur is not None:
                chunks.append(cur)
            cur = {"session": r.session, "start": r.start, "end": r.end, "text": r.text}
    if cur is not None:
        chunks.append(cur)
    return chunks


def split_sessions(sessions: list[str], every: int = 13) -> tuple[list[str], list[str]]:
    val = [s for s in sessions if int(s) % every == 0]
    train = [s for s in sessions if s not in set(val)]
    return train, val
