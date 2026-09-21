from pathlib import Path

import pandas as pd

from lit.data import chunk_session, load_dev, load_jember_tsv, parse_hms, split_sessions


def test_load_dev(tmp_path: Path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "a.mp3").write_bytes(b"")
    (tmp_path / "metadata.tsv").write_text(
        "audio_filename\tspeaker\ttranscript\tlanguage\tconvo_id\n"
        "a.mp3\t1\tHalo, ya.\tjavind\t5\n"
    )
    df = load_dev(tmp_path)
    assert list(df.columns) == ["audio_filename", "speaker", "transcript", "language", "convo_id", "path"]
    assert df.loc[0, "path"] == tmp_path / "clips" / "a.mp3"
    assert df.loc[0, "convo_id"] == 5


def test_parse_hms():
    assert parse_hms("0:00:03") == 3.0
    assert parse_hms("1:02:05") == 3725.0


def test_load_jember_tsv(tmp_path):
    p = tmp_path / "j.tsv"
    p.write_text("Audio file name\tstart\tend\ttext\n1\t0:00:00\t0:00:03\tOke.\n1\t0:00:03\t0:00:05\tYa.\n")
    df = load_jember_tsv(p)
    assert df["session"].tolist() == ["1", "1"]
    assert df["start"].tolist() == [0.0, 3.0]
    assert df["end"].tolist() == [3.0, 5.0]


def test_chunk_session_merges_contiguous_up_to_limit():
    rows = pd.DataFrame({
        "session": ["1"] * 4,
        "start": [0.0, 3.0, 5.0, 20.0],
        "end": [3.0, 5.0, 12.0, 25.0],
        "text": ["a", "b", "c", "d"],
    })
    chunks = chunk_session(rows, max_seconds=10.0)
    assert [(c["start"], c["end"], c["text"]) for c in chunks] == [
        (0.0, 5.0, "a b"),   # a+b = 5 s; adding c would be 12 s > 10
        (5.0, 12.0, "c"),
        (20.0, 25.0, "d"),   # gap before d, never merged
    ]


def test_split_sessions():
    train, val = split_sessions([str(i) for i in range(1, 30)], every=13)
    assert val == ["13", "26"]
    assert "13" not in train and len(train) == 27
