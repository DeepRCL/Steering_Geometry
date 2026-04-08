import json
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import SCHWARTZ_CIRCUMPLEX_ORDER
from .data_loader import DataLoader, PromptFormatter
from .model_loader import ModelInfo
from .steering.base import SteeringMethod


def compute_mean_activation_separation(
    activations_pos: Dict[str, torch.Tensor],
    activations_neg: Dict[str, torch.Tensor],
) -> float:
    """
    Compute a scale-normalized L2 separation for a single value at a single layer.

    Each sample activation is first normalized to unit norm, then we compute
    || mean(normalized_pos) - mean(normalized_neg) ||_2. This reduces the bias
    toward later layers whose residual stream magnitudes are larger overall.
    """
    shared_sample_ids = sorted(set(activations_pos.keys()) & set(activations_neg.keys()))
    if not shared_sample_ids:
        return 0.0

    pos_stack = torch.stack([activations_pos[sid] for sid in shared_sample_ids])
    neg_stack = torch.stack([activations_neg[sid] for sid in shared_sample_ids])

    pos_stack = pos_stack / pos_stack.norm(dim=1, keepdim=True).clamp_min(1e-12)
    neg_stack = neg_stack / neg_stack.norm(dim=1, keepdim=True).clamp_min(1e-12)

    pos_mean = pos_stack.mean(dim=0)
    neg_mean = neg_stack.mean(dim=0)
    return float(torch.norm(pos_mean - neg_mean, p=2).item())


def _save_selection_metadata(out_dir: str, selected_layer: int, selection_metric: str, scores_dict: Dict):
    with open(os.path.join(out_dir, "layer_scores.json"), "w") as f:
        json.dump(scores_dict, f, indent=2)

    with open(os.path.join(out_dir, "selected_layer.json"), "w") as f:
        json.dump(
            {
                "selected_layer": selected_layer,
                "selection_metric": selection_metric,
            },
            f,
            indent=2,
        )


def _select_layer_by_normalized_l2(config, layers, activations) -> int:
    mean_scores = {}
    per_value_scores = {}

    for layer_idx in layers:
        layer_value_scores = {}
        for value_name in SCHWARTZ_CIRCUMPLEX_ORDER:
            acts_pos = activations[value_name]["pos"][layer_idx]
            acts_neg = activations[value_name]["neg"][layer_idx]
            layer_value_scores[value_name] = compute_mean_activation_separation(acts_pos, acts_neg)

        per_value_scores[layer_idx] = layer_value_scores
        mean_scores[layer_idx] = float(np.mean(list(layer_value_scores.values())))

    selected_layer = max(layers, key=lambda layer_idx: mean_scores[layer_idx])
    out_dir = config.subdir("layer_selection")

    scores_dict = {
        layer_idx: {
            "mean_normalized_l2_separation": mean_scores[layer_idx],
            "per_value_normalized_l2_separation": per_value_scores[layer_idx],
        }
        for layer_idx in layers
    }
    _save_selection_metadata(out_dir, selected_layer, "normalized_l2", scores_dict)

    y_vals = np.array([mean_scores[layer_idx] for layer_idx in layers])
    plt.figure(figsize=(10, 6))
    plt.plot(layers, y_vals, marker="o", linewidth=2, label="Mean Normalized L2 Separation")
    plt.axvline(x=selected_layer, color="red", linestyle="--", label=f"Selected ({selected_layer})")
    plt.xlabel("Layer")
    plt.ylabel("Mean Normalized L2 Separation")
    plt.title("Layer Selection by Mean Normalized L2 Separation")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "mean_normalized_l2_separation.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Selected layer based on mean normalized L2 separation: {selected_layer}")
    return selected_layer


def _evaluate_value_accuracy_for_layer(
    layer_idx: int,
    alpha: float,
    value_name: str,
    vector: torch.Tensor,
    data_loader: DataLoader,
    model_info: ModelInfo,
    steering_method: SteeringMethod,
) -> float:
    eval_instances = data_loader.get_eval_instances(value_name)
    if not eval_instances:
        return 0.0

    formatter = PromptFormatter(model_info.tokenizer, model_info.is_instruct)
    handles = steering_method.apply(model_info, layer_idx, vector, alpha)

    try:
        correct_count = 0
        for inst in eval_instances:
            tokens, a_id, b_id = formatter.format_eval_prompt(inst)
            input_ids = torch.tensor([tokens]).to(model_info.device)

            with torch.no_grad():
                logits = model_info.model(input_ids).logits

            last_logits = logits[0, -1, :]
            prob_a = F.softmax(last_logits, dim=-1)[a_id].item()
            prob_b = F.softmax(last_logits, dim=-1)[b_id].item()
            chose_a = prob_a > prob_b
            if chose_a == inst.pos_is_a:
                correct_count += 1

        return correct_count / len(eval_instances)
    finally:
        steering_method.cleanup(handles)


def _select_layer_by_eval_accuracy(
    config,
    layers,
    vectors,
    data_loader: DataLoader,
    model_info: ModelInfo,
    steering_method: SteeringMethod,
) -> int:
    if data_loader is None or model_info is None or steering_method is None:
        raise ValueError("Evaluation-based layer selection requires data_loader, model_info, and steering_method.")

    model_info.model.eval()
    mean_scores = {}
    per_alpha_scores = {}

    for layer_idx in layers:
        alpha_scores = {}
        for alpha in config.alpha_values:
            per_value_acc = []
            for value_name in SCHWARTZ_CIRCUMPLEX_ORDER:
                acc = _evaluate_value_accuracy_for_layer(
                    layer_idx=layer_idx,
                    alpha=alpha,
                    value_name=value_name,
                    vector=vectors[value_name][layer_idx],
                    data_loader=data_loader,
                    model_info=model_info,
                    steering_method=steering_method,
                )
                per_value_acc.append(acc)

            alpha_scores[alpha] = float(np.mean(per_value_acc))

        per_alpha_scores[layer_idx] = alpha_scores
        mean_scores[layer_idx] = max(alpha_scores.values())

    selected_layer = max(layers, key=lambda layer_idx: mean_scores[layer_idx])
    best_alpha_per_layer = {
        layer_idx: max(per_alpha_scores[layer_idx], key=per_alpha_scores[layer_idx].get)
        for layer_idx in layers
    }

    out_dir = config.subdir("layer_selection")
    scores_dict = {
        layer_idx: {
            "best_mean_eval_accuracy": mean_scores[layer_idx],
            "best_alpha": best_alpha_per_layer[layer_idx],
            "per_alpha_mean_eval_accuracy": per_alpha_scores[layer_idx],
        }
        for layer_idx in layers
    }
    _save_selection_metadata(out_dir, selected_layer, "eval_accuracy", scores_dict)

    plt.figure(figsize=(10, 6))
    for alpha in config.alpha_values:
        y_vals = [per_alpha_scores[layer_idx][alpha] for layer_idx in layers]
        plt.plot(layers, y_vals, marker="o", linewidth=1.5, label=f"alpha={alpha}")

    plt.axvline(x=selected_layer, color="red", linestyle="--", label=f"Selected ({selected_layer})")
    plt.xlabel("Layer")
    plt.ylabel("Mean Eval Accuracy")
    plt.title("Layer Selection by Held-out Evaluation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "eval_accuracy_by_layer.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Selected layer based on held-out evaluation accuracy: {selected_layer}")
    return selected_layer


def select_layer(
    config,
    vectors: Dict[str, Dict[int, torch.Tensor]],
    activations: Dict[str, Dict[str, Dict[int, Dict[str, torch.Tensor]]]],
    data_loader: Optional[DataLoader] = None,
    model_info: Optional[ModelInfo] = None,
    steering_method: Optional[SteeringMethod] = None,
) -> int:
    print("Computing layer selection metrics...")

    layers = list(vectors[SCHWARTZ_CIRCUMPLEX_ORDER[0]].keys())
    layers.sort()

    if config.layer_selection_method == "normalized_l2":
        return _select_layer_by_normalized_l2(config, layers, activations)
    if config.layer_selection_method == "eval_accuracy":
        return _select_layer_by_eval_accuracy(config, layers, vectors, data_loader, model_info, steering_method)

    raise ValueError(
        f"Unknown layer selection method: {config.layer_selection_method}. "
        "Expected one of: normalized_l2, eval_accuracy."
    )
