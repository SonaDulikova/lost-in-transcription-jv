from pathlib import Path

import pandas as pd

from lit.data import (
    chunk_session,
    group_by_duration,
    load_dev,
    load_jember_tsv,
    load_slr35_index,
    load_tts_tsv,
    parse_hms,
    slr35_mapping,
    split_sessions,
    strip_diacritics,
)


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


def test_load_slr35_index(tmp_path: Path):
    p = tmp_path / "utt_spk_text.tsv"
    p.write_text("00004fe6aa\ta4815\tKanthong semar minangka tanduran\n"
                 "0000e5df79\tffe12\t  Banjur saluran mbelok  \n"
                 "0001bbbc2e\t0a834\t\n")
    df = load_slr35_index(p)
    assert df.columns.tolist() == ["id", "speaker", "text"]
    assert df["id"].tolist() == ["00004fe6aa", "0000e5df79"]
    assert df["text"].tolist() == ["Kanthong semar minangka tanduran", "Banjur saluran mbelok"]
    assert pd.api.types.is_string_dtype(df["speaker"])  # speaker ids like "0a834" must stay strings
    assert df["speaker"].tolist() == ["a4815", "ffe12"]


def _index(rows):
    return pd.DataFrame(rows, columns=["id", "speaker", "text"])


def test_slr35_mapping_filters_to_available():
    idx = _index([("aa11", "s1", "Siji"), ("bb22", "s1", "Loro"), ("cc33", "s2", "Telu")])
    m = slr35_mapping(idx, available_ids={"aa11", "cc33"})
    assert m["file"].tolist() == ["aa/aa11.flac", "cc/cc33.flac"]


def test_slr35_mapping_sorts_by_speaker_then_id():
    idx = _index([("zz99", "s2", "A"), ("aa11", "s2", "B"), ("mm55", "s1", "C")])
    m = slr35_mapping(idx, available_ids={"zz99", "aa11", "mm55"})
    assert m["file"].tolist() == ["mm/mm55.flac", "aa/aa11.flac", "zz/zz99.flac"]


def test_slr35_mapping_appends_period():
    idx = _index([("aa11", "s1", "Kanthong semar minangka tanduran")])
    m = slr35_mapping(idx, available_ids={"aa11"})
    assert m["text"].tolist() == ["Kanthong semar minangka tanduran."]


def test_slr35_mapping_keeps_existing_terminator():
    idx = _index([("aa11", "s1", "Kuwi apa?"), ("bb22", "s1", "Ayo!"), ("cc33", "s1", "Wis.")])
    m = slr35_mapping(idx, available_ids={"aa11", "bb22", "cc33"})
    assert m["text"].tolist() == ["Kuwi apa?", "Ayo!", "Wis."]


def test_slr35_mapping_drops_empty_text():
    idx = _index([("aa11", "s1", "   "), ("bb22", "s1", "Loro")])
    m = slr35_mapping(idx, available_ids={"aa11", "bb22"})
    assert m["file"].tolist() == ["bb/bb22.flac"]


def test_slr35_mapping_keeps_diacritics():
    idx = _index([("aa11", "s1", "Dheweké uga diundhang")])
    m = slr35_mapping(idx, available_ids={"aa11"})
    assert m["text"].tolist() == ["Dheweké uga diundhang."]
