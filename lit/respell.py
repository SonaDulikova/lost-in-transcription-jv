"""Rewrite read-speech training transcripts toward convo 5's spelling conventions.

The map covers only variant classes where the Central Javanese corpus is near 100% one form and
convo 5 clearly writes the other (counts, CJV / convo 5, measured 2026-09-28). Where convo 5 is
split, the standard form wins (wes -> wis) or the word is left alone. Derived from convo 5 only,
never from convo 2, which is the score set.
"""

from __future__ import annotations

import re

CONVO5_MAP = {
    "yo": "ya",            # 528/0  vs 21/424
    "opo": "apa",          # 222/1  vs 22/92
    "ojo": "aja",          # 13/0   vs 1/25
    "cerito": "cerita",    # 74/10  vs 0/18
    "kerjo": "kerja",      # 120/28 vs 0/9
    "bedo": "beda",        # 48/1   vs 0/10
    "limo": "lima",        # 7/0    vs 0/30
    "mbien": "mbiyen",     # 130/0  vs 0/6
    "ngenggo": "nganggo",  # 49/5   vs 0/9
    "wes": "wis",          # 524/3  vs 12/11, split: the standard form
    "okeh": "akeh",        # 610/5  vs 0/3, and convo 5 never writes okeh
}

# a maximal letter run, so hyphen halves (opo-opo) map and longer words (Yogyakarta, kerjone) do not
_WORD = re.compile(r"[A-Za-z]+")


def _swap(m: re.Match) -> str:
    w = m.group(0)
    new = CONVO5_MAP.get(w.lower())
    if new is None:
        return w
    return new.capitalize() if w[0].isupper() else new


def respell(text: str) -> str:
    return _WORD.sub(_swap, text)
