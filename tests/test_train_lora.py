import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py"


def _load():
    spec = importlib.util.spec_from_file_location("train_lora", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_lora"] = mod
    spec.loader.exec_module(mod)
    return mod


tl = _load()


class FixedChoice(random.Random):
    """An rng whose choice() always returns one factor, but still counts the draws."""

    def __init__(self, factor: float):
        super().__init__(0)
        self.factor, self.draws = factor, 0

    def choice(self, seq):
        self.draws += 1
        assert self.factor in seq
        return self.factor


def clip(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * tl.SR), dtype=np.float32)


def test_slow_down_stretches_a_short_clip():
    y = clip(10.0)
    out = tl.speed_perturb(y, FixedChoice(0.9))
    assert len(out) / tl.SR > 11.0


def test_slow_down_skipped_when_the_stretched_clip_would_pass_30_s():
    # 28 s / 0.9 = 31.1 s: the feature extractor would cut the audio but the label keeps every word
    y = clip(28.0)
    assert tl.speed_perturb(y, FixedChoice(0.9)) is y


def test_speed_up_still_applies_to_long_clips():
    y = clip(29.5)
    out = tl.speed_perturb(y, FixedChoice(1.1))
    assert len(out) / tl.SR < 27.0


def tiny_whisper():
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    torch.manual_seed(0)
    cfg = WhisperConfig(vocab_size=64, d_model=16, encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=2, decoder_attention_heads=2,
                        encoder_ffn_dim=32, decoder_ffn_dim=32, num_mel_bins=8,
                        max_source_positions=16, max_target_positions=16,
                        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)
    return WhisperForConditionalGeneration(cfg)


def lora_a(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_A" in k}


def lora_init(seed: int, rng_state: int, scope: str = "all") -> dict:
    import torch

    model = tiny_whisper()
    # the real script reaches get_peft_model with a per-process random torch RNG state
    torch.manual_seed(rng_state)
    return lora_a(tl.add_lora(model, rank=4, scope=scope, seed=seed))


def test_lora_init_is_reproducible_for_a_seed():
    import torch

    a, b = lora_init(seed=3, rng_state=111), lora_init(seed=3, rng_state=222)
    assert a and a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0, msg=k)


def test_lora_init_differs_between_seeds():
    import torch

    a, b = lora_init(seed=0, rng_state=111), lora_init(seed=1, rng_state=111)
    assert any(not torch.equal(a[k], b[k]) for k in a)


def test_decoder_scope_adapts_only_the_decoder():
    keys = lora_init(seed=0, rng_state=0, scope="decoder").keys()
    assert keys and all(".decoder." in k for k in keys)


def test_default_keeps_the_best_epoch_with_early_stopping():
    from transformers import EarlyStoppingCallback

    kwargs, callbacks = tl.selection(no_select=False)
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "wer" and kwargs["greater_is_better"] is False
    assert any(isinstance(c, EarlyStoppingCallback) for c in callbacks)


def test_no_select_trains_every_epoch_and_keeps_the_last():
    from transformers import EarlyStoppingCallback

    kwargs, callbacks = tl.selection(no_select=True)
    assert kwargs["load_best_model_at_end"] is False
    assert not any(isinstance(c, EarlyStoppingCallback) for c in callbacks)


def test_no_select_flag_parses():
    base = ["--train", "a.csv", "--val", "b.csv", "--out", "o"]
    assert tl.parse_args(base).no_select is False
    assert tl.parse_args(base + ["--no-select"]).no_select is True


class RecordingModel:
    def __init__(self):
        self.saved = []

    def save_pretrained(self, path):
        self.saved.append(Path(path))


def run_snapshots(tmp_path, max_steps: int, epochs: float, from_epoch: float, per_epoch: int = 4) -> list[str]:
    from types import SimpleNamespace

    cb = tl.SnapshotCallback(tmp_path, from_epoch=from_epoch, per_epoch=per_epoch)
    args = SimpleNamespace(num_train_epochs=epochs)
    state = SimpleNamespace(max_steps=max_steps, global_step=0, is_world_process_zero=True)
    model = RecordingModel()
    cb.on_train_begin(args, state, None)
    for step in range(1, max_steps + 1):
        state.global_step = step
        cb.on_step_end(args, state, None, model=model)
    assert all(p.parent == tmp_path / "snapshots" for p in model.saved)
    return [p.name for p in model.saved]


def test_snapshots_every_quarter_epoch_from_epoch_two(tmp_path):
    # 3 epochs of 40 optimizer steps: quarter-epoch = 10 steps, epoch 2 ends at step 80
    assert run_snapshots(tmp_path, max_steps=120, epochs=3, from_epoch=2) == [
        "step-80", "step-90", "step-100", "step-110", "step-120"]


def test_snapshots_sit_on_the_nearest_step_to_each_quarter_epoch(tmp_path):
    # 3 epochs of 21 steps: quarter epochs 2.0 .. 3.0 fall at steps 42, 47.25, 52.5, 57.75, 63
    assert run_snapshots(tmp_path, max_steps=63, epochs=3, from_epoch=2) == [
        "step-42", "step-47", "step-53", "step-58", "step-63"]


def test_no_snapshots_when_the_run_ends_before_the_start_epoch(tmp_path):
    assert run_snapshots(tmp_path, max_steps=6, epochs=0.05, from_epoch=2) == []


def test_snapshot_flags_default_off():
    base = ["--train", "a.csv", "--val", "b.csv", "--out", "o"]
    assert tl.parse_args(base).snapshot_from_epoch is None
    args = tl.parse_args(base + ["--snapshot-from-epoch", "2", "--snapshots-per-epoch", "2"])
    assert args.snapshot_from_epoch == 2.0 and args.snapshots_per_epoch == 2


def test_augment_acoustic_flag_defaults_off():
    base = ["--train", "a.csv", "--val", "b.csv", "--out", "o"]
    assert tl.parse_args(base).augment_acoustic is False
    assert tl.parse_args(base + ["--augment-acoustic"]).augment_acoustic is True


def test_transform_applies_the_acoustic_augment(tmp_path):
    import soundfile as sf

    y = np.zeros(tl.SR, dtype=np.float32)
    y[::100] = 0.5  # clicks, so an mp3/reverb/babble change is visible
    sf.write(tmp_path / "a.wav", y, tl.SR)
    seen = []

    class Spy:
        def __call__(self, arr):
            seen.append(arr.copy())
            return arr * 0.5

    from types import SimpleNamespace

    processor = SimpleNamespace(
        feature_extractor=lambda arr, sampling_rate: SimpleNamespace(input_features=[arr]),
        tokenizer=lambda text: SimpleNamespace(input_ids=[1]),
    )
    transform = tl.make_transform(processor, augment=False, acoustic=Spy())
    out = transform({"path": [str(tmp_path / "a.wav")], "text": ["x"]})
    assert len(seen) == 1 and np.allclose(out["input_features"][0], seen[0] * 0.5)


def test_skipping_the_slow_down_still_consumes_one_draw():
    # the factor sequence for every other clip must not shift when one long clip is skipped
    rng = FixedChoice(0.9)
    tl.speed_perturb(clip(29.0), rng)
    assert rng.draws == 1
