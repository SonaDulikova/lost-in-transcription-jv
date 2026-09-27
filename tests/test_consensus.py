import pytest

from lit.consensus import pick_consensus, word_edit_distance


def test_word_edit_distance():
    assert word_edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert word_edit_distance(["a", "b", "c"], ["a", "c"]) == 1
    assert word_edit_distance(["a", "b"], ["x", "a", "b", "y"]) == 2
    assert word_edit_distance([], ["a", "b"]) == 2
    assert word_edit_distance(["kowe", "wis"], ["kowe", "wes"]) == 1


def test_picks_the_candidate_the_others_agree_with():
    assert pick_consensus(["aku ning omah", "aku neng omah", "aku neng omah"]) == 1
    assert pick_consensus(["x y z", "aku neng omah", "aku neng omah"]) == 1


def test_ties_go_to_the_first_listed_candidate():
    assert pick_consensus(["aku neng omah", "kowe ning pasar"]) == 0
    assert pick_consensus(["kowe ning pasar", "aku neng omah"]) == 0


def test_compares_normalized_words_not_raw_text():
    # the first two differ only in casing of the first letter and punctuation, which the scorer removes
    assert pick_consensus(["Ya, gak papa.", "ya gak papa", "yo ora popo"]) == 0


def test_distance_is_divided_by_the_other_candidates_length():
    # raw distance sums are 4 / 5 / 5 and would pick candidate 0; divided by the other candidate's
    # length they are 2/4+2/1 = 2.5, 2/2+3/1 = 4, 2/2+3/4 = 1.75, which picks candidate 2
    assert pick_consensus(["kowe wis", "kowe wis aku kowe", "aku"]) == 2


def test_single_candidate():
    assert pick_consensus(["apa wae"]) == 0


def test_rejects_no_candidates():
    with pytest.raises(ValueError):
        pick_consensus([])
