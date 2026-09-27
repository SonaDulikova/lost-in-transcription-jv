"""Pick, per clip, the transcript that agrees best with the other systems' transcripts (a medoid).

Candidates are compared as scorer-normalized words: candidate a's risk is the sum over all candidates b of
word_edit_distance(a, b) / len(b), and the lowest risk wins, ties going to the first listed candidate.
Pure Python on purpose: the runtime image has no rapidfuzz, and clips are a few dozen words.
"""

from __future__ import annotations

from collections.abc import Sequence

from lit.normalize import normalize_text


def word_edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


def pick_consensus(candidates: Sequence[str]) -> int:
    """Index of the candidate transcript with the lowest mean normalized distance to the others."""
    if not candidates:
        raise ValueError("no candidates")
    words = [normalize_text(c).split() for c in candidates]
    risk = [sum(word_edit_distance(a, b) / max(len(b), 1) for b in words) for a in words]
    return min(range(len(candidates)), key=risk.__getitem__)
