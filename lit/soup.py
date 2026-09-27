"""Model soups of LoRA adapters: add a weighted average of their merged deltas to a base model.

Each adapter contributes delta_i = (lora_alpha / r) * B_i @ A_i per targeted linear layer, and the base gets
W0 + scale * sum_i(w_i * delta_i) with the weights normalised to sum to 1. The factors A and B are never
averaged: mean(B) @ mean(A) contains cross terms B_i @ A_j, which are noise across seeds (PEFT's
add_weighted_adapter "linear" combination does exactly that). Works on adapter/ and checkpoint-*/ dirs alike.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

PREFIX = "base_model.model."


def lora_deltas(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Merged LoRA deltas of one adapter, keyed by the base model's state_dict name."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        raise ValueError(f"{adapter_dir}: dora adapters are not a plain B@A delta")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError(f"{adapter_dir}: per-module rank/alpha patterns are not supported")
    if cfg.get("bias", "none") != "none" or cfg.get("lora_bias") or cfg.get("modules_to_save"):
        raise ValueError(f"{adapter_dir}: trained biases or modules_to_save are not supported")
    r = cfg["r"]
    scaling = cfg["lora_alpha"] / (math.sqrt(r) if cfg.get("use_rslora") else r)

    tensors = load_file(adapter_dir / "adapter_model.safetensors")
    deltas = {}
    for key, a in tensors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        b = tensors[key.replace(".lora_A.", ".lora_B.")]
        name = key.removeprefix(PREFIX).replace(".lora_A.weight", ".weight")
        deltas[name] = scaling * (b.float() @ a.float())
    return deltas


@torch.no_grad()
def add_soup(model: torch.nn.Module, adapters: list[Path], weights: list[float] | None = None,
             scale: float = 1.0) -> None:
    """Add scale * weighted mean of the adapters' deltas to model's weights in place (one adapter in memory at a time)."""
    weights = weights or [1.0] * len(adapters)
    if len(weights) != len(adapters):
        raise ValueError(f"{len(weights)} weights for {len(adapters)} adapters")
    total = sum(weights)
    params = model.state_dict()
    for adapter, w in zip(adapters, weights):
        for name, delta in lora_deltas(Path(adapter)).items():
            if name not in params:
                raise KeyError(f"{adapter}: delta for {name} has no matching base weight")
            params[name].add_(delta.to(params[name].dtype), alpha=scale * w / total)
