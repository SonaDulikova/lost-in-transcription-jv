from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR = 16000


def build_hf_transcriber(model_dir: Path | str, language: str = "id", beam_size: int = 5,
                         adapter: Path | None = None) -> Callable[[Path], str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir), torch_dtype=dtype).to(device)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.eval()
    model.generation_config.forced_decoder_ids = None

    @torch.inference_mode()
    def transcribe(path: Path) -> str:
        y, _ = librosa.load(path, sr=SR, mono=True)
        feats = processor(y, sampling_rate=SR, return_tensors="pt").input_features.to(device, dtype)
        ids = model.generate(feats, language=language, task="transcribe", num_beams=beam_size, max_new_tokens=440)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    return transcribe
