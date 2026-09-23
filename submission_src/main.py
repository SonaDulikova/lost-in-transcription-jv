"""Lost in Transcription submission: offline faster-whisper inference.

Reads submission_format.csv and clips/ from the data dir, writes submission.csv.
Runs on GPU when CTranslate2 sees one, otherwise on CPU. Weights live in ./model.
Logs only counts and timings, never filenames or transcripts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LIT_DATA_DIR", "/code_execution/data"))
SUBMISSION_PATH = Path(os.environ.get("LIT_SUBMISSION_PATH", "/code_execution/submission/submission.csv"))
MODEL_DIR = Path(os.environ.get("LIT_MODEL_DIR", HERE / "model"))
CONFIG_PATH = Path(os.environ.get("LIT_CONFIG", HERE / "config.json"))

sys.path.insert(0, str(HERE))
from postproc import apply  # noqa: E402  (sibling module; the zip has no package)


def log(msg: str) -> None:
    print(f"[main] {msg}", flush=True)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def build_transcriber(model_dir: Path, cfg: dict) -> Callable[[Path], str]:
    # The pip CTranslate2 wheel does not bundle cuBLAS/cuDNN. Importing torch first
    # preloads its nvidia-* wheels so ctranslate2 can dlopen libcublas.so.12 and
    # libcudnn.so.9. Harmless on CPU and in the runtime image, which also has torch.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    import ctranslate2
    from faster_whisper import WhisperModel

    use_gpu = ctranslate2.get_cuda_device_count() > 0
    device = "cuda" if use_gpu else "cpu"
    compute_type = cfg["compute_type_gpu"] if use_gpu else cfg["compute_type_cpu"]
    log(f"loading model from {model_dir.name} on {device} ({compute_type})")
    model = WhisperModel(str(model_dir), device=device, compute_type=compute_type, cpu_threads=os.cpu_count() or 4)

    def transcribe(path: Path) -> str:
        segments, _info = model.transcribe(
            str(path),
            language=cfg.get("language"),
            beam_size=cfg.get("beam_size", 5),
            temperature=cfg.get("temperature", 0.0),
            patience=cfg.get("patience", 1.0),
            condition_on_previous_text=cfg.get("condition_on_previous_text", False),
            vad_filter=cfg.get("vad_filter", False),
        )
        return apply(" ".join(s.text.strip() for s in segments).strip(), cfg.get("postprocess", ["diacritics"]))

    return transcribe


def run(transcribe: Callable[[Path], str], data_dir: Path, out_path: Path) -> int:
    fmt = pl.read_csv(data_dir / "submission_format.csv")
    clips_dir = data_dir / "clips"
    texts: list[str] = []
    failures = 0
    t0 = time.time()
    for i, name in enumerate(fmt["audio_filename"], start=1):
        try:
            text = transcribe(clips_dir / name)
            if not isinstance(text, str):
                text = ""
        except Exception as e:  # noqa: BLE001 - one bad clip must not sink the run
            failures += 1
            log(f"clip {i}: transcription failed ({type(e).__name__})")
            text = ""
        texts.append(text)
        if i % 50 == 0:
            log(f"{i}/{fmt.height} clips, {time.time() - t0:.0f}s elapsed")
    out = fmt.with_columns(pl.Series("transcript", texts))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    log(f"wrote {out.height} rows to {out_path} ({failures} failures, {time.time() - t0:.0f}s)")
    return out.height


def main() -> None:
    cfg = load_config()
    transcribe = build_transcriber(MODEL_DIR, cfg)
    run(transcribe, DATA_DIR, SUBMISSION_PATH)


if __name__ == "__main__":
    sys.exit(main())
