from pathlib import Path

from lit.data import group_by_duration, load_dev, load_tts_tsv, strip_diacritics


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


def test_strip_diacritics():
    assert strip_diacritics("Nèngdi akèh é") == "Nengdi akeh e"
    assert strip_diacritics("plain ascii") == "plain ascii"


def test_load_tts_tsv(tmp_path: Path):
    p = tmp_path / "t.tsv"
    p.write_text("id\tsentence\tspeaker\na.webm\t Kuwi apa? \ts1\nb.webm\t\ts1\n")
    df = load_tts_tsv(p, file_col="id", text_col="sentence")
    assert df["file"].tolist() == ["a.webm"]
    assert df["text"].tolist() == ["Kuwi apa?"]


def test_group_by_duration_packs_consecutive_items():
    groups = group_by_duration([4.0, 5.0, 12.0, 3.0, 25.0, 2.0], max_seconds=20.0, gap=0.25)
    assert groups == [[0, 1], [2, 3], [4], [5]]  # 4+0.25+5 = 9.25; +12 -> 21.5 > 20


def test_group_by_duration_empty():
    assert group_by_duration([], max_seconds=20.0) == []

