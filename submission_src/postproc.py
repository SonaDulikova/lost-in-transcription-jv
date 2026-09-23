"""Toggleable post-processing rules applied to a transcript before scoring or submission."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence

# leading separator is only space/comma so the period that ends the real last sentence survives
_TRAILING = re.compile(r"(?:[\s,]*(?:terima kasih(?: telah menonton)?|sampai jumpa|\bbye)[\s.,!?]*)+$", re.IGNORECASE)
_LEADING_HAI = re.compile(r"^\W*hai\b\W*", re.IGNORECASE)
_RANGE_DASH = re.compile(r"(?<=\d)-(?=\d)")
_NUMBER = re.compile(r"\d+(?:[.,]\d{3})*")


def diacritics(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def case(text: str) -> str:
    # Whisper capitalises every segment start; the scorer only lowercases after . ! ?
    words = text.split(" ")
    out = []
    for i, w in enumerate(words):
        prev = words[i - 1] if i else ""
        sentence_start = i == 0 or (prev != "" and prev[-1] in ".!?")
        if sentence_start or w.isupper() or not w[:1].isupper():
            out.append(w)
        else:
            out.append(w[0].lower() + w[1:])
    return " ".join(out)


def tail(text: str) -> str:
    return _TRAILING.sub("", text).rstrip()


def hai(text: str) -> str:
    return _LEADING_HAI.sub("", text)


def num2words_id(text: str) -> str:
    from num2words import num2words  # dev-only dependency; vendored into the zip only if this rule is promoted

    def repl(m: re.Match) -> str:
        return num2words(int(m.group(0).replace(".", "").replace(",", "")), lang="id")

    return _NUMBER.sub(repl, _RANGE_DASH.sub(" ", text))


RULES: dict[str, Callable[[str], str]] = {
    "diacritics": diacritics,
    "case": case,
    "tail": tail,
    "hai": hai,
    "num2words": num2words_id,
}


def apply(text: str, rules: Sequence[str]) -> str:
    for name in rules:
        if name not in RULES:
            raise KeyError(f"unknown postprocess rule {name!r}; known: {list(RULES)}")
        text = RULES[name](text)
    return text
