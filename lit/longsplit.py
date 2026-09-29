"""Split a > 30 s dev clip into two <= 30 s training pieces without splitting the text by duration.

The cut goes where the reference text has a word boundary that the recogniser's word timestamps
anchor on both sides (the two reference words are recognised exactly, back to back), preferring a
sentence end, then a comma, then the largest pause, then the middle of the clip.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

Word = tuple[str, float, float]  # (text, start s, end s), as faster-whisper's word timestamps


def norm(token: str) -> str:
    t = "".join(c for c in unicodedata.normalize("NFD", token) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w-]", "", t.lower()).strip("-_")


def _strength(token: str) -> int:
    if token.endswith("..."):
        return 1  # trailing off, not a finished sentence
    if token[-1:] in ".?!":
        return 2
    return 1 if token.endswith(",") else 0


def split_text(ref: str, n: int) -> tuple[str, str]:
    tokens = ref.split()
    return " ".join(tokens[:n]), " ".join(tokens[n:])


def choose_cut(ref: str, hyp: list[Word], duration: float, max_s: float = 30.0,
               margin: float = 0.5) -> tuple[int, float] | None:
    """(number of reference tokens in the first piece, cut time in s), or None if no safe cut."""
    tokens = ref.split()
    ref_n, hyp_n = [norm(t) for t in tokens], [norm(w) for w, _, _ in hyp]
    to_hyp = {}
    for a, b, size in SequenceMatcher(None, ref_n, hyp_n, autojunk=False).get_matching_blocks():
        for k in range(size):
            if ref_n[a + k]:
                to_hyp[a + k] = b + k
    limit = max_s - margin
    best, best_key = None, None
    for n in range(1, len(tokens)):
        j = to_hyp.get(n - 1)
        if j is None or to_hyp.get(n) != j + 1:
            continue  # this boundary has no timing we can trust
        end, start = hyp[j][2], hyp[j + 1][1]
        t = (end + start) / 2
        if t > limit or duration - t > limit:
            continue
        key = (_strength(tokens[n - 1]), min(max(start - end, 0.0), 0.5), -abs(t - duration / 2))
        if best_key is None or key > best_key:
            best, best_key = (n, t), key
    return best
