import subprocess
import sys
from pathlib import Path

import pandas as pd

from lit.respell import CONVO5_MAP, respell

REPO = Path(__file__).resolve().parents[1]


def test_maps_whole_words_to_the_convo5_spelling():
    assert respell("aku yo ora ngerti opo sing kudu dilakoni") == "aku ya ora ngerti apa sing kudu dilakoni"
    assert respell("wes okeh sing kerjo") == "wis akeh sing kerja"


def test_keeps_sentence_initial_capitals_and_punctuation():
    assert respell("Yo, mbien aku ngono. Opo?") == "Ya, mbiyen aku ngono. Apa?"


def test_maps_each_part_of_a_reduplicated_word():
    assert respell("opo-opo lan cerito-cerito sing bedo-bedo") == "apa-apa lan cerita-cerita sing beda-beda"
    assert respell("okeh-okeh") == "akeh-akeh"


def test_leaves_words_that_only_contain_a_mapped_form():
    assert respell("Yogyakarta youtube kerjone mbiene wesi") == "Yogyakarta youtube kerjone mbiene wesi"


def test_leaves_forms_where_convo5_is_split_or_agrees():
    for w in ["mergo", "iso", "ono", "dewe", "meneh", "ning", "nggo", "kuwi", "koyo", "ora", "jowo", "gedhe"]:
        assert w not in CONVO5_MAP
        assert respell(w) == w


def test_map_targets_are_never_themselves_mapped():
    assert not set(CONVO5_MAP.values()) & set(CONVO5_MAP)


def test_respell_manifest_rewrites_text_and_keeps_every_other_column(tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"path": ["/a.wav", "/b.wav"], "text": ["Yo wes.", "ora opo-opo"],
                  "session": ["cjv", "cjv"], "duration": [1.5, 2.0]}).to_csv(src, index=False)
    out = tmp_path / "out.csv"
    subprocess.run([sys.executable, str(REPO / "scripts" / "respell_manifest.py"), str(src), "--out", str(out)],
                   check=True, cwd=REPO)
    df = pd.read_csv(out, keep_default_na=False)
    assert df["text"].tolist() == ["Ya wis.", "ora apa-apa"]
    assert df[["path", "session", "duration"]].equals(pd.read_csv(src)[["path", "session", "duration"]])
