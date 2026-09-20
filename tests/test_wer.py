from lit.wer import corpus_wer, grouped_wer, per_utterance_wer


def test_perfect_match_is_zero():
    r = corpus_wer(["aku iso", "ya"], ["aku iso", "ya"])
    assert r.wer == 0.0 and r.ref_words == 3


def test_normalization_applied_before_scoring():
    r = corpus_wer(["Aku iso, ya."], ["aku iso ya"])
    assert r.wer == 0.0


def test_counts():
    r = corpus_wer(["a b c"], ["a x c d"])
    assert (r.substitutions, r.deletions, r.insertions) == (1, 0, 1)
    assert abs(r.wer - 2 / 3) < 1e-9


def test_grouped():
    out = grouped_wer(["a b", "c d"], ["a b", "c x"], ["ind", "javind"])
    assert out["ind"].wer == 0.0
    assert out["javind"].wer == 0.5
    assert out["all"].wer == 0.25


def test_per_utterance():
    assert per_utterance_wer(["a b", "c"], ["a b", "x"]) == [0.0, 1.0]
