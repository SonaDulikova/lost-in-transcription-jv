from lit.longsplit import choose_cut, split_text


def timed(words: str, start: float = 0.0, step: float = 1.0, pauses: dict[int, float] | None = None):
    """Hypothesis words one per `step` seconds; pauses[i] adds silence before word i."""
    out, t = [], start
    for i, w in enumerate(words.split()):
        t += (pauses or {}).get(i, 0.0)
        out.append((" " + w, t, t + step * 0.8))
        t += step
    return out


def test_cuts_at_the_sentence_end_that_keeps_both_halves_under_30_s():
    ref = " ".join(f"w{i}" for i in range(17)) + ". " + " ".join(f"v{i}" for i in range(17)) + "."
    hyp = timed(ref, pauses={17: 0.6})
    cut = choose_cut(ref, hyp, duration=35.0)
    assert cut is not None
    n, t = cut
    assert n == 17
    assert 16.8 < t < 17.6  # inside the pause between w16 and v0
    assert t <= 29.5 and 35.0 - t <= 29.5


def test_prefers_a_sentence_end_over_a_comma_nearer_the_middle():
    ref = "a b c d e f g h i j k. l m n o p, q r s t u v w x y z aa bb cc dd ee ff gg"
    cut = choose_cut(ref, timed(ref), duration=34.0)
    assert cut is not None and split_text(ref, cut[0])[0].endswith("k.")


def test_falls_back_to_a_comma_then_a_plain_word_boundary():
    ref = "a b c d e f g h i j k l m n o p, q r s t u v w x y z aa bb cc dd ee ff gg"
    cut = choose_cut(ref, timed(ref), duration=34.0)
    assert cut is not None and split_text(ref, cut[0])[0].endswith("p,")
    plain = ref.replace(",", "")
    cut = choose_cut(plain, timed(plain), duration=34.0)
    assert cut is not None and 3.5 < cut[1] < 30.5


def test_matches_words_ignoring_case_punctuation_and_diacritics():
    ref = "Aku ngerti. " + " ".join(f"x{i}" for i in range(15)) + ". Nèng kono " + " ".join(f"y{i}" for i in range(15))
    hyp = timed(ref.lower().replace(".", "").replace("è", "e"))
    cut = choose_cut(ref, hyp, duration=33.0)
    assert cut is not None and split_text(ref, cut[0])[1].startswith("Nèng kono")


def test_skips_a_boundary_the_hypothesis_did_not_anchor():
    # the recogniser got the word after the only sentence end wrong, so that boundary has no timing
    ref = " ".join(f"w{i}" for i in range(17)) + ". " + " ".join(f"v{i}" for i in range(17))
    hyp_text = ref.replace("v0", "zzz")
    cut = choose_cut(ref, timed(hyp_text), duration=35.0)
    assert cut is None or not split_text(ref, cut[0])[0].endswith("w16.")


def test_no_cut_when_no_boundary_keeps_both_halves_under_the_limit():
    ref = "a b c d e f g h i j"
    hyp = [(" " + w, i * 0.1, i * 0.1 + 0.05) for i, w in enumerate(ref.split())]  # all speech in the first second
    assert choose_cut(ref, hyp, duration=36.0) is None


def test_split_text_keeps_every_original_token():
    ref = "Nah, selang dua tahun kuwi. Kucing sing ibu i... mati"
    a, b = split_text(ref, 5)
    assert a == "Nah, selang dua tahun kuwi." and b == "Kucing sing ibu i... mati"
    assert f"{a} {b}".split() == ref.split()
