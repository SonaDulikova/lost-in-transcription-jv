from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperConfig, WhisperForConditionalGeneration

from lit.soup import add_soup

TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def tiny_whisper() -> WhisperForConditionalGeneration:
    torch.manual_seed(0)
    cfg = WhisperConfig(vocab_size=64, d_model=16, encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=2, decoder_attention_heads=2,
                        encoder_ffn_dim=32, decoder_ffn_dim=32, num_mel_bins=8,
                        max_source_positions=16, max_target_positions=16,
                        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)
    return WhisperForConditionalGeneration(cfg).eval()


def make_adapter(path: Path, seed: int, targets=TARGETS, rank: int = 4, **extra) -> Path:
    torch.manual_seed(seed)
    # init_lora_weights=False draws B at random too, so the delta B@A is non-zero like a trained adapter
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=targets, init_lora_weights=False, **extra)
    get_peft_model(tiny_whisper(), cfg).save_pretrained(path)
    return path


def merged(adapter: Path) -> dict[str, torch.Tensor]:
    from peft import PeftModel
    return PeftModel.from_pretrained(tiny_whisper(), str(adapter)).merge_and_unload().state_dict()


def souped(adapters, **kw) -> dict[str, torch.Tensor]:
    model = tiny_whisper()
    add_soup(model, adapters, **kw)
    return model.state_dict()


def assert_same(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k, value in a.items():
        torch.testing.assert_close(value, b[k], rtol=0, atol=1e-6, msg=k)


def test_single_adapter_matches_peft_merge(tmp_path):
    a = make_adapter(tmp_path / "a", seed=1)
    assert_same(souped([a]), merged(a))


def test_soup_averages_merged_deltas_not_factors(tmp_path):
    a, b = make_adapter(tmp_path / "a", seed=1), make_adapter(tmp_path / "b", seed=2)
    base, ma, mb = tiny_whisper().state_dict(), merged(a), merged(b)
    expected = {k: base[k] + 0.5 * (ma[k] - base[k]) + 0.5 * (mb[k] - base[k]) for k in base}
    assert_same(souped([a, b]), expected)


def test_weights_and_scale(tmp_path):
    a, b = make_adapter(tmp_path / "a", seed=1), make_adapter(tmp_path / "b", seed=2)
    base, ma, mb = tiny_whisper().state_dict(), merged(a), merged(b)
    expected = {k: base[k] + 1.2 * (0.25 * (ma[k] - base[k]) + 0.75 * (mb[k] - base[k])) for k in base}
    assert_same(souped([a, b], weights=[1, 3], scale=1.2), expected)


def test_decoder_only_adapter_leaves_encoder_untouched(tmp_path):
    dec = make_adapter(tmp_path / "dec", seed=1, targets=rf".*\.decoder\..*\.({'|'.join(TARGETS)})")
    base, out = tiny_whisper().state_dict(), souped([dec])
    enc_keys = [k for k in base if ".encoder." in k]
    assert enc_keys and all(torch.equal(out[k], base[k]) for k in enc_keys)
    assert not torch.equal(out["model.decoder.layers.0.fc1.weight"], base["model.decoder.layers.0.fc1.weight"])


def test_rejects_dora(tmp_path):
    a = make_adapter(tmp_path / "a", seed=1, use_dora=True)
    with pytest.raises(ValueError, match="dora"):
        souped([a])


def test_rejects_delta_for_missing_module(tmp_path):
    a = make_adapter(tmp_path / "a", seed=1)
    weights = a / "adapter_model.safetensors"
    from safetensors.torch import load_file, save_file
    tensors = {k.replace("layers.0.self_attn.q_proj", "layers.9.self_attn.q_proj"): v
               for k, v in load_file(weights).items()}
    save_file(tensors, weights)
    with pytest.raises(KeyError, match="layers.9"):
        souped([a])


def test_rejects_mismatched_weights(tmp_path):
    a = make_adapter(tmp_path / "a", seed=1)
    with pytest.raises(ValueError, match="weights"):
        souped([a], weights=[1, 1])
