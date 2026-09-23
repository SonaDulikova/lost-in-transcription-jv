import pytest

from lit.postproc import RULES, apply


def test_diacritics():
    assert apply("Nèngdi akèh, é... ya.", ["diacritics"]) == "Nengdi akeh, e... ya."


def test_case_lowercases_mid_sentence_capitals_only():
    assert apply("Kalau di Solo, Ya. Dia Bilang", ["case"]) == "Kalau di solo, ya. Dia bilang"
    assert apply("Kalau di SMP, ya.", ["case"]) == "Kalau di SMP, ya."
    assert apply("", ["case"]) == ""


def test_tail_strips_hallucinated_closers():
    assert apply("aku neng kene. Terima kasih telah menonton.", ["tail"]) == "aku neng kene."
    assert apply("iya. Terima kasih. Bye.", ["tail"]) == "iya."
    assert apply("Terima kasih", ["tail"]) == ""
    assert apply("terima kasih ya bu", ["tail"]) == "terima kasih ya bu"
    assert apply("aku pamit goodbye", ["tail"]) == "aku pamit goodbye"


def test_hai_strips_leading_hai_only():
    assert apply("Hai, apa kabar", ["hai"]) == "apa kabar"
    assert apply("Haid itu apa", ["hai"]) == "Haid itu apa"
    assert apply("apa kabar hai", ["hai"]) == "apa kabar hai"


def test_num2words_indonesian():
    assert apply("seratus 100 kg tahun 2023", ["num2words"]) == "seratus seratus kg tahun dua ribu dua puluh tiga"
    assert apply("harga 1.000 rupiah", ["num2words"]) == "harga seribu rupiah"
    assert apply("nomor 102-103", ["num2words"]) == "nomor seratus dua seratus tiga"


def test_apply_order_and_unknown_rule():
    assert apply("Hai, Nèng Solo. Terima kasih.", ["diacritics", "hai", "tail", "case"]) == "Neng solo."
    assert apply("x", []) == "x"
    with pytest.raises(KeyError):
        apply("x", ["nope"])


def test_rule_names_are_stable():
    assert list(RULES) == ["diacritics", "case", "tail", "hai", "num2words"]
