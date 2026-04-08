import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from .config import PipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER
from .data_loader import DataLoader, PromptFormatter
from .model_loader import ModelInfo
from .steering.base import SteeringMethod


def _score_instance(model_info: ModelInfo, formatter: PromptFormatter, inst, handles: Optional[Any] = None) -> Dict[str, Any]:
    tokens, a_id, b_id = formatter.format_eval_prompt(inst)
    input_ids = torch.tensor([tokens]).to(model_info.device)

    with torch.no_grad():
        logits = model_info.model(input_ids).logits

    last_logits = logits[0, -1, :]
    probs = F.softmax(last_logits, dim=-1)

    prob_a = probs[a_id].item()
    prob_b = probs[b_id].item()
    prob_positive = prob_a if inst.pos_is_a else prob_b
    prob_negative = prob_b if inst.pos_is_a else prob_a
    positive_margin = prob_positive - prob_negative

    chose_a = prob_a > prob_b
    chose_positive = chose_a == inst.pos_is_a

    return {
        "sample_id": inst.sample_id,
        "prob_a": prob_a,
        "prob_b": prob_b,
        "prob_positive": prob_positive,
        "prob_negative": prob_negative,
        "positive_margin": positive_margin,
        "chose_a": chose_a,
        "pos_is_a": inst.pos_is_a,
        "is_correct": chose_positive,
    }


def _summarize_details(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    accuracy = float(np.mean([detail["is_correct"] for detail in details])) if details else 0.0
    mean_prob_positive = float(np.mean([detail["prob_positive"] for detail in details])) if details else 0.0
    mean_prob_negative = float(np.mean([detail["prob_negative"] for detail in details])) if details else 0.0
    mean_positive_margin = float(np.mean([detail["positive_margin"] for detail in details])) if details else 0.0

    return {
        "accuracy": accuracy,
        "num_eval": len(details),
        "mean_prob_positive": mean_prob_positive,
        "mean_prob_negative": mean_prob_negative,
        "mean_positive_margin": mean_positive_margin,
        "details": details,
    }


def evaluate_steering(
    config: PipelineConfig,
    data_loader: DataLoader,
    model_info: ModelInfo,
    steering_method: SteeringMethod,
    vectors: Dict[str, torch.Tensor],
    layer_idx: int,
):
    """
    Evaluates steering on the held-out split for each value and compares it to
    the unsteered baseline model.
    """
    print(f"Evaluating steering on layer {layer_idx} with alphas: {config.alpha_values}")
    formatter = PromptFormatter(model_info.tokenizer, model_info.is_instruct)

    out_dir = config.subdir("evaluation")
    results_all = {}

    model_info.model.eval()

    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        eval_instances = data_loader.get_eval_instances(val)
        if not eval_instances:
            continue

        # Baseline / non-steered model
        baseline_details = [_score_instance(model_info, formatter, inst) for inst in eval_instances]
        baseline_summary = _summarize_details(baseline_details)

        vector = vectors[val]
        steered_results = {}

        for alpha in config.alpha_values:
            handles = steering_method.apply(model_info, layer_idx, vector, alpha)
            try:
                print(f"Evaluating {val} (alpha={alpha})...")
                steered_details = []

                for inst, baseline_detail in zip(eval_instances, baseline_details):
                    detail = _score_instance(model_info, formatter, inst)
                    detail["baseline_prob_positive"] = baseline_detail["prob_positive"]
                    detail["baseline_prob_negative"] = baseline_detail["prob_negative"]
                    detail["baseline_positive_margin"] = baseline_detail["positive_margin"]
                    detail["baseline_is_correct"] = baseline_detail["is_correct"]
                    detail["delta_prob_positive"] = detail["prob_positive"] - baseline_detail["prob_positive"]
                    detail["delta_prob_negative"] = detail["prob_negative"] - baseline_detail["prob_negative"]
                    detail["delta_positive_margin"] = detail["positive_margin"] - baseline_detail["positive_margin"]
                    detail["became_correct"] = (not baseline_detail["is_correct"]) and detail["is_correct"]
                    detail["became_incorrect"] = baseline_detail["is_correct"] and (not detail["is_correct"])
                    steered_details.append(detail)
            finally:
                steering_method.cleanup(handles)

            steered_summary = _summarize_details(steered_details)
            steered_summary["accuracy_gain_vs_baseline"] = steered_summary["accuracy"] - baseline_summary["accuracy"]
            steered_summary["mean_prob_positive_gain_vs_baseline"] = (
                steered_summary["mean_prob_positive"] - baseline_summary["mean_prob_positive"]
            )
            steered_summary["mean_positive_margin_gain_vs_baseline"] = (
                steered_summary["mean_positive_margin"] - baseline_summary["mean_positive_margin"]
            )
            steered_results[str(alpha)] = steered_summary

        results_all[val] = {
            "baseline": baseline_summary,
            "steered": steered_results,
        }

    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(results_all, f, indent=2)

    value_order = [val for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in results_all]
    alpha_labels = [str(alpha) for alpha in config.alpha_values]

    if value_order:
        baseline_accuracy = np.array([results_all[val]["baseline"]["accuracy"] for val in value_order])
        steered_accuracy = np.array(
            [[results_all[val]["steered"][alpha]["accuracy"] for alpha in alpha_labels] for val in value_order]
        )
        accuracy_gain = steered_accuracy - baseline_accuracy[:, None]

        baseline_prob_positive = np.array([results_all[val]["baseline"]["mean_prob_positive"] for val in value_order])
        steered_prob_positive = np.array(
            [
                [results_all[val]["steered"][alpha]["mean_prob_positive"] for alpha in alpha_labels]
                for val in value_order
            ]
        )
        prob_positive_gain = steered_prob_positive - baseline_prob_positive[:, None]

        baseline_margin = np.array([results_all[val]["baseline"]["mean_positive_margin"] for val in value_order])
        steered_margin = np.array(
            [
                [results_all[val]["steered"][alpha]["mean_positive_margin"] for alpha in alpha_labels]
                for val in value_order
            ]
        )
        margin_gain = steered_margin - baseline_margin[:, None]

        mean_steered_accuracy = steered_accuracy.mean(axis=0)
        mean_accuracy_gain = accuracy_gain.mean(axis=0)
        mean_prob_positive_gain = prob_positive_gain.mean(axis=0)
        mean_margin_gain = margin_gain.mean(axis=0)

        plt.figure(figsize=(8, 5))
        plt.axhline(
            y=float(baseline_accuracy.mean()),
            color="gray",
            linestyle="--",
            linewidth=2,
            label="Baseline",
        )
        plt.plot(config.alpha_values, mean_steered_accuracy, marker="o", linewidth=2, label="Steered")
        plt.xlabel("Alpha")
        plt.ylabel("Mean Accuracy")
        plt.title("Baseline vs Steered Mean Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "baseline_vs_steered_accuracy.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(config.alpha_values, mean_accuracy_gain, marker="o", linewidth=2)
        plt.axhline(y=0.0, color="gray", linestyle="--")
        plt.xlabel("Alpha")
        plt.ylabel("Mean Accuracy Gain vs Baseline")
        plt.title("Mean Accuracy Gain from Steering")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "accuracy_gain_vs_baseline.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(config.alpha_values, mean_prob_positive_gain, marker="o", linewidth=2, label="Positive Probability Gain")
        plt.plot(config.alpha_values, mean_margin_gain, marker="s", linewidth=2, label="Positive Margin Gain")
        plt.axhline(y=0.0, color="gray", linestyle="--")
        plt.xlabel("Alpha")
        plt.ylabel("Mean Gain vs Baseline")
        plt.title("Probability and Margin Gain from Steering")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "probability_gain_vs_baseline.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 10))
        sns.heatmap(
            accuracy_gain,
            xticklabels=alpha_labels,
            yticklabels=value_order,
            cmap="coolwarm",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
        )
        plt.xlabel("Alpha")
        plt.ylabel("Value")
        plt.title("Accuracy Gain by Value and Alpha")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "accuracy_gain_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 10))
        sns.heatmap(
            prob_positive_gain,
            xticklabels=alpha_labels,
            yticklabels=value_order,
            cmap="coolwarm",
            center=0.0,
        )
        plt.xlabel("Alpha")
        plt.ylabel("Value")
        plt.title("Positive Probability Gain by Value and Alpha")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "positive_probability_gain_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Evaluation complete. Results saved to {out_dir}")
