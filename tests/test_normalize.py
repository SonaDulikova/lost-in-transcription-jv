import importlib.util
from pathlib import Path

import pytest

from lit.normalize import normalize_text

SAMPLES = [
    "Eh kalau di waktu-waktu SMP Mba I punya pengalaman apa?",
    "Aku... dan aku berencana untuk ikut half marathon pada tahun ini.",
    "Nah, e... baca [laughs] buku (?) itu — terus ya.",
    "Ya. Iso. Ora ono sopo-sopo!",
    "  double  spaces   here ",
    "¿Qué? Nèngdi tempat wisata; alam: populer.",
    "SMP itu. UGM juga.",
    "",
]


def _upstream():
    path = Path(__file__).resolve().parents[1] / "third_party" / "score.py"
    spec = importlib.util.spec_from_file_location("upstream_score", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize_text


@pytest.mark.parametrize("text", SAMPLES)
def test_matches_upstream(text):
    assert normalize_text(text) == _upstream()(text)


def test_sentence_initial_lowercase_but_acronym_kept():
    assert normalize_text("SMP itu. Bagus.") == "SMP itu bagus "


def test_ellipsis_kept_periods_dropped():
    assert normalize_text("Aku... dan aku.") == "aku... dan aku "
