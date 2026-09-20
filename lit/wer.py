from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import jiwer

from lit.normalize import normalize_text


@dataclass
class WerResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_words: int

    def row(self) -> str:
        return (
            f"WER {self.wer:.4f}  S={self.substitutions} D={self.deletions} "
            f"I={self.insertions} H={self.hits} N={self.ref_words}"
        )


def _norm(texts: list[str]) -> list[str]:
    return [normalize_text(t if isinstance(t, str) else "") for t in texts]


def corpus_wer(refs: list[str], hyps: list[str]) -> WerResult:
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} refs vs {len(hyps)} hyps")
    nr, nh = _norm(refs), _norm(hyps)
    # jiwer rejects empty references; keep them as a single placeholder token
    nr = [r if r.strip() else "<empty>" for r in nr]
    out = jiwer.process_words(nr, nh)
    n = out.hits + out.substitutions + out.deletions
    return WerResult(
        wer=out.wer,
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        hits=out.hits,
        ref_words=n,
    )


def grouped_wer(refs: list[str], hyps: list[str], groups: list[str]) -> dict[str, WerResult]:
    buckets: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for r, h, g in zip(refs, hyps, groups, strict=True):
        buckets[g][0].append(r)
        buckets[g][1].append(h)
    result = {g: corpus_wer(r, h) for g, (r, h) in sorted(buckets.items())}
    result["all"] = corpus_wer(refs, hyps)
    return result


def per_utterance_wer(refs: list[str], hyps: list[str]) -> list[float]:
    return [corpus_wer([r], [h]).wer for r, h in zip(refs, hyps, strict=True)]
