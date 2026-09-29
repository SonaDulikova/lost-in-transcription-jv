"""Recover the > 30 s dev clips as two <= 30 s training pieces each (lit/longsplit.py), verified.

A cut is kept only if, re-transcribing each piece with the same model, (1) each piece's hypothesis
starts/ends with the reference words the cut assigned to it, and (2) neither piece's WER is more
than --tol above the whole clip's WER. A wrong cut spills words across the boundary and fails both.

    uv run scripts/split_long_clips.py data/dev_segments/convo2.csv data/dev_segments/convo5.csv \
        --model runs/ct2/soup_alldev5 --out data/dev_segments/long_split.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch  # noqa: F401  (preloads the nvidia-* wheels so ctranslate2 finds libcublas, see main.py)
import faster_whisper.transcribe as ft
from faster_whisper import WhisperModel

from lit.longsplit import choose_cut, norm, split_text
from lit.wer import corpus_wer

MAX_S = 30.0

_find_alignment = ft.WhisperModel.find_alignment


def _guarded_find_alignment(self, tokenizer, text_tokens, *args, **kwargs):
    # faster-whisper 1.2.1 crashes (IndexError) when ctranslate2 returns an empty alignment for a
    # segment; losing that segment's word timings only removes candidate cuts, so skip it instead
    try:
        return _find_alignment(self, tokenizer, text_tokens, *args, **kwargs)
    except IndexError:
        return [[] for _ in text_tokens]


ft.WhisperModel.find_alignment = _guarded_find_alignment


def edge_ok(ref_piece: str, hyp_piece: str, first: bool, window: int = 4) -> bool:
    ref_w = [w for w in map(norm, ref_piece.split()) if w]
    hyp_w = [w for w in map(norm, hyp_piece.split()) if w]
    if not ref_w or not hyp_w:
        return False
    return ref_w[0] in hyp_w[:window] if first else ref_w[-1] in hyp_w[-window:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", type=Path, nargs="+")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="manifest of the accepted pieces")
    ap.add_argument("--wav-dir", type=Path, default=Path("data/dev_segments/wav_long"))
    ap.add_argument("--tol", type=float, default=0.10, help="max piece WER above the whole clip's WER")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p, keep_default_na=False) for p in args.manifests], ignore_index=True)
    long = df[df["duration"] > MAX_S].reset_index(drop=True)
    model = WhisperModel(str(args.model), device="cuda", compute_type="float16")
    opts = dict(language="id", beam_size=5, condition_on_previous_text=False, vad_filter=False)
    args.wav_dir.mkdir(parents=True, exist_ok=True)

    def hyp_of(audio) -> str:
        return " ".join(s.text.strip() for s in model.transcribe(audio, **opts)[0])

    pieces, report = [], []
    for r in long.itertuples():
        y, sr = sf.read(r.path, dtype="float32")
        segs = list(model.transcribe(r.path, word_timestamps=True, **opts)[0])
        words = [(w.word, w.start, w.end) for s in segs for w in (s.words or [])]
        whole = corpus_wer([r.text], [" ".join(s.text.strip() for s in segs)]).wer
        row = {"clip": Path(r.path).stem, "duration": round(r.duration, 1), "whole_wer": round(whole, 3)}
        cut = choose_cut(r.text, words, r.duration, max_s=MAX_S)
        if cut is None:
            report.append({**row, "status": "no safe cut"})
            continue
        n, t = cut
        texts = split_text(r.text, n)
        k = int(round(t * sr))
        audio = [y[:k], y[k:]]
        hyps = [hyp_of(a) for a in audio]
        wers = [corpus_wer([tx], [h]).wer for tx, h in zip(texts, hyps)]
        edges = edge_ok(texts[0], hyps[0], first=False) and edge_ok(texts[1], hyps[1], first=True)
        ok = edges and max(wers) <= whole + args.tol
        report.append({**row, "cut_s": round(t, 2), "piece_wer_a": round(wers[0], 3),
                       "piece_wer_b": round(wers[1], 3), "edges": edges, "status": "kept" if ok else "rejected"})
        if not ok:
            continue
        for tag, a, tx in zip("ab", audio, texts):
            path = (args.wav_dir / f"{Path(r.path).stem}_{tag}.wav").resolve()
            sf.write(path, a, sr, subtype="PCM_16")
            pieces.append({"path": str(path), "text": tx, "session": f"{r.session}_long", "duration": len(a) / sr})

    rep = pd.DataFrame(report)
    rep.to_csv(args.out.with_suffix(".report.csv"), index=False)
    pd.DataFrame(pieces, columns=["path", "text", "session", "duration"]).to_csv(args.out, index=False)
    print(rep.to_string(index=False))
    kept = rep[rep["status"] == "kept"]
    print(f"\n{len(kept)} of {len(rep)} long clips kept -> {len(pieces)} pieces, "
          f"{sum(p['duration'] for p in pieces) / 60:.1f} min, written to {args.out}")


if __name__ == "__main__":
    main()
