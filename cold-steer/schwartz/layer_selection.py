"""Method-agnostic layer selection via normalized L2 separation.

Identical scoring rule to ``llm-steering-opt/pipeline/steering_pipeline.py``
so cold_fd's layer choice can be compared apples-to-apples against
``optimize_vector`` runs on the same dataset.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import data_utils
from .config import SCHWARTZ_CIRCUMPLEX_ORDER


def _normalized_mean_diff(
    pos: List[torch.Tensor], neg: List[torch.Tensor]
) -> float:
    n = min(len(pos), len(neg))
    if n == 0:
        return 0.0
    p = torch.stack(pos[:n]).float()
    q = torch.stack(neg[:n]).float()
    p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return float(torch.norm(p.mean(dim=0) - q.mean(dim=0), p=2).item())


@torch.no_grad()
def _last_token_activations(
    steerable_llm,
    text: str,
    layer_idx: int,
) -> torch.Tensor:
    """Forward ``text`` and return the last non-pad-token activation at ``layer_idx``.

    Uses cold-steer's existing steering-hook registration to capture the
    layer output without modifying it.
    """
    captured: Dict[int, torch.Tensor] = {}

    def hook(module, inp, out, lidx=layer_idx):
        # Hybrid architectures (Qwen3-Next / Qwen3.5) return a bare tensor
        # (B, T, D); standard transformer layers return a tuple whose first
        # element is the hidden state. Handle both.
        hidden = out[0] if isinstance(out, tuple) else out
        captured[lidx] = hidden.detach()
        return out

    # Temporarily expose only the candidate layer as a steering layer
    # (we restore it after the forward pass).
    from .method_adapters import set_steering_layers
    prev_layers = list(steerable_llm.steering_layer_indices)
    try:
        set_steering_layers(steerable_llm, [layer_idx])
        tok = steerable_llm.tokenizer(text, return_tensors="pt")
        input_ids = tok["input_ids"].to(steerable_llm.model.device)
        attention_mask = tok["attention_mask"].to(steerable_llm.model.device)
        handles = steerable_llm.register_steering_hooks(lambda l: hook)
        try:
            steerable_llm(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()
    finally:
        set_steering_layers(steerable_llm, prev_layers)

    z = captured[layer_idx][0]   # (T, D)
    attn = attention_mask[0]
    last_pos = int(attn.shape[0] - 1)
    while last_pos > 0 and attn[last_pos].item() == 0:
        last_pos -= 1
    return z[last_pos, :].detach().to("cpu").float()


def get_default_candidates(n_layers: int, n_cand: int = 12) -> List[int]:
    n_cand = min(n_cand, n_layers)
    start = max(1, int(n_layers * 0.15))
    end = int(n_layers * 0.85)
    step = max(1, (end - start) // (n_cand - 1)) if n_cand > 1 else 1
    return list(range(start, end + 1, step))[:n_cand]


def select_layer(
    steerable_llm,
    train_rows: List[dict],
    values: List[str],
    candidates: Optional[Sequence[int]] = None,
    n_samples_per_value: int = 10,
    seed: int = 42,
    use_chat_template: bool = True,
    prompt_template: str = "",
    verbose: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    """Pick the layer with the largest mean normalized L2 separation.

    Returns ``(best_layer, payload)`` where ``payload`` is a JSON-ready
    dict with all per-layer / per-value scores.
    """
    n_layers = steerable_llm.model.config.num_hidden_layers
    if candidates is None:
        candidates = get_default_candidates(n_layers)
    candidates = list(candidates)

    sweep_values = [v for v in values if v in SCHWARTZ_CIRCUMPLEX_ORDER]
    if not sweep_values:
        sweep_values = list(values)

    if verbose:
        print(f"Layer sweep (normalized L2 separation) over candidates: {candidates}")

    activations: Dict[str, Dict[int, Dict[str, List[torch.Tensor]]]] = {}
    rng = random.Random(seed)
    tokenizer = steerable_llm.tokenizer

    for value in sweep_values:
        value_rows = data_utils.get_rows_for_value(train_rows, value)
        if not value_rows:
            continue
        sample = value_rows if len(value_rows) <= n_samples_per_value \
            else rng.sample(value_rows, n_samples_per_value)
        activations[value] = {layer: {"pos": [], "neg": []} for layer in candidates}
        for row in sample:
            prompt = data_utils.format_prompt(
                row["question"], tokenizer, use_chat_template, prompt_template
            )
            pos_text = f"{prompt} {row['positive_answer']}"
            neg_text = f"{prompt} {row['negative_answer']}"
            for layer in candidates:
                activations[value][layer]["pos"].append(
                    _last_token_activations(steerable_llm, pos_text, layer)
                )
                activations[value][layer]["neg"].append(
                    _last_token_activations(steerable_llm, neg_text, layer)
                )

    mean_scores: Dict[int, float] = {}
    per_value_scores: Dict[int, Dict[str, float]] = {}
    for layer in candidates:
        layer_scores: Dict[str, float] = {}
        for value in sweep_values:
            if value not in activations:
                continue
            acts = activations[value][layer]
            layer_scores[value] = _normalized_mean_diff(acts["pos"], acts["neg"])
        per_value_scores[layer] = layer_scores
        vals = list(layer_scores.values())
        mean_scores[layer] = float(np.mean(vals)) if vals else 0.0

    if not mean_scores or all(v == 0.0 for v in mean_scores.values()):
        best_layer = candidates[len(candidates) // 2]
        if verbose:
            print(f"  Warning: all layers scored 0. Using middle layer {best_layer}")
    else:
        best_layer = max(candidates, key=lambda lyr: mean_scores.get(lyr, 0.0))
        if verbose:
            print(f"  Best layer: {best_layer} "
                  f"(mean normalized L2 sep = {mean_scores[best_layer]:.4f})")
            for layer in candidates:
                print(f"    Layer {layer}: mean normalized L2 sep = {mean_scores.get(layer, 0):.4f}")

    payload = {
        "candidates": candidates,
        "scores": {
            str(layer): {
                "mean_normalized_l2_separation": mean_scores.get(layer, 0.0),
                "per_value_normalized_l2_separation": {
                    k: round(v, 6) for k, v in per_value_scores.get(layer, {}).items()
                },
            }
            for layer in candidates
        },
        "best_layer": best_layer,
    }
    return best_layer, payload
