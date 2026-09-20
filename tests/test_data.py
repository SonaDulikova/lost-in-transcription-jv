from pathlib import Path

import pandas as pd

from lit.data import load_dev


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
