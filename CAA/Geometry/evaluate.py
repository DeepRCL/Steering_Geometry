import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from .config import PipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER
from .data_loader import DataLoader, PromptFormatter
from .model_loader import ModelInfo
from .steering.base import SteeringMethod


def _mean_completion_logprob(
    model_info: ModelInfo,
    prompt: str,
    completion: str,
) -> float:
    """
    Compute mean per-token log-probability autoregressively.

    This is intentionally token-by-token rather than one full forward pass so
    last-token steering methods such as SphericalSteer affect the prediction of
    every completion token, matching generation-time behavior.
    """
    completion_text = " " + completion.lstrip()
    prompt_ids = model_info.tokenizer.encode(prompt, add_special_tokens=True)
    completion_ids = model_info.tokenizer.encode(completion_text, add_special_tokens=False)
    if not completion_ids:
        return 0.0

    prefix_ids = list(prompt_ids)
    token_logprobs = []
    with torch.no_grad():
        for token_id in completion_ids:
            input_ids = torch.tensor([prefix_ids]).to(model_info.device)
            logits = model_info.model(input_ids).logits[0, -1, :]
            logprob = F.log_softmax(logits, dim=-1)[token_id].item()
            token_logprobs.append(logprob)
            prefix_ids.append(token_id)

    return float(np.mean(token_logprobs)) if token_logprobs else 0.0


def _format_generation_prompt(formatter: PromptFormatter, question: str) -> str:
    return formatter._format_base_prompt(question)


def _score_full_logprob(model_info: ModelInfo, formatter: PromptFormatter, inst) -> Dict[str, Any]:
    prompt = _format_generation_prompt(formatter, inst.question)
    lp_positive = _mean_completion_logprob(model_info, prompt, inst.positive_answer)
    lp_negative = _mean_completion_logprob(model_info, prompt, inst.negative_answer)
    margin = lp_positive - lp_negative

    return {
        "sample_id": inst.sample_id,
        "mean_logprob_positive": lp_positive,
        "mean_logprob_negative": lp_negative,
        "logprob_positive_margin": margin,
        "is_correct": lp_positive > lp_negative,
    }


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


def _summarize_full_logprob(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    accuracy = float(np.mean([detail["is_correct"] for detail in details])) if details else 0.0
    mean_lp_positive = float(np.mean([detail["mean_logprob_positive"] for detail in details])) if details else 0.0
    mean_lp_negative = float(np.mean([detail["mean_logprob_negative"] for detail in details])) if details else 0.0
    mean_margin = float(np.mean([detail["logprob_positive_margin"] for detail in details])) if details else 0.0

    return {
        "accuracy": accuracy,
        "num_eval": len(details),
        "mean_logprob_positive": mean_lp_positive,
        "mean_logprob_negative": mean_lp_negative,
        "mean_logprob_positive_margin": mean_margin,
        "details": details,
    }


def _build_metric_summary(
    results_all: Dict[str, Any],
    alpha_labels: List[str],
    layer_idx: int,
    baseline_key: str,
    steered_key: str,
    metric_name: str,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    value_order = [val for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in results_all]
    baseline_accuracy = np.array([results_all[val][baseline_key]["accuracy"] for val in value_order])
    steered_accuracy = np.array(
        [[results_all[val][steered_key][alpha]["accuracy"] for alpha in alpha_labels] for val in value_order]
    )
    accuracy_gain = steered_accuracy - baseline_accuracy[:, None]

    mean_steered_accuracy = steered_accuracy.mean(axis=0)
    mean_accuracy_gain = accuracy_gain.mean(axis=0)
    best_alpha_idx = int(np.argmax(mean_accuracy_gain))
    best_alpha = alpha_labels[best_alpha_idx]
    best_accuracy_gain = float(mean_accuracy_gain[best_alpha_idx])
    best_mean_steered_accuracy = float(mean_steered_accuracy[best_alpha_idx])
    mean_baseline_accuracy = float(baseline_accuracy.mean())

    per_value_best_alpha = {}
    for row_idx, value_name in enumerate(value_order):
        value_best_idx = int(np.argmax(accuracy_gain[row_idx]))
        value_best_alpha = alpha_labels[value_best_idx]
        per_value_best_alpha[value_name] = {
            "baseline_accuracy": float(baseline_accuracy[row_idx]),
            "best_alpha": value_best_alpha,
            "best_accuracy": float(steered_accuracy[row_idx, value_best_idx]),
            "best_accuracy_gain_vs_baseline": float(accuracy_gain[row_idx, value_best_idx]),
        }

    summary = {
        "metric": metric_name,
        "layer_idx": int(layer_idx),
        "alpha_values": [float(alpha) for alpha in alpha_labels],
        "mean_baseline_accuracy": mean_baseline_accuracy,
        "mean_steered_accuracy_by_alpha": {
            alpha: float(mean_steered_accuracy[idx]) for idx, alpha in enumerate(alpha_labels)
        },
        "mean_accuracy_gain_by_alpha": {
            alpha: float(mean_accuracy_gain[idx]) for idx, alpha in enumerate(alpha_labels)
        },
        "best_overall_alpha_by_mean_accuracy_gain": {
            "alpha": best_alpha,
            "mean_accuracy_gain_vs_baseline": best_accuracy_gain,
            "mean_accuracy_at_alpha": best_mean_steered_accuracy,
            "mean_baseline_accuracy": mean_baseline_accuracy,
        },
        "per_value_best_alpha_by_accuracy_gain": per_value_best_alpha,
    }
    return summary, baseline_accuracy, steered_accuracy, accuracy_gain, value_order


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
        baseline_logprob_details = [
            _score_full_logprob(model_info, formatter, inst) for inst in eval_instances
        ]
        baseline_logprob_summary = _summarize_full_logprob(baseline_logprob_details)

        vector = vectors[val]
        steered_results = {}
        steered_logprob_results = {}

        for alpha in config.alpha_values:
            handles = steering_method.apply(model_info, layer_idx, vector, alpha)
            try:
                print(f"Evaluating {val} (alpha={alpha})...")
                steered_details = []
                steered_logprob_details = []

                for inst, baseline_detail, baseline_lp_detail in zip(
                    eval_instances,
                    baseline_details,
                    baseline_logprob_details,
                ):
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

                    lp_detail = _score_full_logprob(model_info, formatter, inst)
                    lp_detail["baseline_mean_logprob_positive"] = baseline_lp_detail["mean_logprob_positive"]
                    lp_detail["baseline_mean_logprob_negative"] = baseline_lp_detail["mean_logprob_negative"]
                    lp_detail["baseline_logprob_positive_margin"] = baseline_lp_detail["logprob_positive_margin"]
                    lp_detail["baseline_is_correct"] = baseline_lp_detail["is_correct"]
                    lp_detail["delta_logprob_positive"] = (
                        lp_detail["mean_logprob_positive"] - baseline_lp_detail["mean_logprob_positive"]
                    )
                    lp_detail["delta_logprob_negative"] = (
                        lp_detail["mean_logprob_negative"] - baseline_lp_detail["mean_logprob_negative"]
                    )
                    lp_detail["delta_logprob_positive_margin"] = (
                        lp_detail["logprob_positive_margin"] - baseline_lp_detail["logprob_positive_margin"]
                    )
                    lp_detail["became_correct"] = (
                        not baseline_lp_detail["is_correct"] and lp_detail["is_correct"]
                    )
                    lp_detail["became_incorrect"] = (
                        baseline_lp_detail["is_correct"] and not lp_detail["is_correct"]
                    )
                    steered_logprob_details.append(lp_detail)
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

            steered_logprob_summary = _summarize_full_logprob(steered_logprob_details)
            steered_logprob_summary["accuracy_gain_vs_baseline"] = (
                steered_logprob_summary["accuracy"] - baseline_logprob_summary["accuracy"]
            )
            steered_logprob_summary["mean_logprob_positive_gain_vs_baseline"] = (
                steered_logprob_summary["mean_logprob_positive"]
                - baseline_logprob_summary["mean_logprob_positive"]
            )
            steered_logprob_summary["mean_logprob_positive_margin_gain_vs_baseline"] = (
                steered_logprob_summary["mean_logprob_positive_margin"]
                - baseline_logprob_summary["mean_logprob_positive_margin"]
            )
            steered_logprob_results[str(alpha)] = steered_logprob_summary

        results_all[val] = {
            "baseline": baseline_summary,
            "steered": steered_results,
            "baseline_full_logprob": baseline_logprob_summary,
            "steered_full_logprob": steered_logprob_results,
        }

    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(results_all, f, indent=2)

    with open(os.path.join(out_dir, "eval_results_full_logprob.json"), "w") as f:
        json.dump(
            {
                val: {
                    "baseline": payload["baseline_full_logprob"],
                    "steered": payload["steered_full_logprob"],
                }
                for val, payload in results_all.items()
            },
            f,
            indent=2,
        )

    value_order = [val for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in results_all]
    alpha_labels = [str(alpha) for alpha in config.alpha_values]

    if value_order:
        (
            evaluation_summary,
            baseline_accuracy,
            steered_accuracy,
            accuracy_gain,
            value_order,
        ) = _build_metric_summary(
            results_all,
            alpha_labels,
            layer_idx,
            "baseline",
            "steered",
            "ab_next_token",
        )
        (
            full_logprob_summary,
            full_logprob_baseline_accuracy,
            full_logprob_steered_accuracy,
            full_logprob_accuracy_gain,
            _,
        ) = _build_metric_summary(
            results_all,
            alpha_labels,
            layer_idx,
            "baseline_full_logprob",
            "steered_full_logprob",
            "full_answer_mean_logprob",
        )

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
        evaluation_summary["mean_prob_positive_gain_by_alpha"] = {
            alpha: float(mean_prob_positive_gain[idx]) for idx, alpha in enumerate(alpha_labels)
        }
        evaluation_summary["mean_positive_margin_gain_by_alpha"] = {
            alpha: float(mean_margin_gain[idx]) for idx, alpha in enumerate(alpha_labels)
        }
        evaluation_summary["full_answer_mean_logprob"] = full_logprob_summary
        with open(os.path.join(out_dir, "evaluation_summary.json"), "w") as f:
            json.dump(evaluation_summary, f, indent=2)
        with open(os.path.join(out_dir, "evaluation_summary_full_logprob.json"), "w") as f:
            json.dump(full_logprob_summary, f, indent=2)

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
        plt.axhline(
            y=float(full_logprob_baseline_accuracy.mean()),
            color="gray",
            linestyle="--",
            linewidth=2,
            label="Baseline",
        )
        plt.plot(
            config.alpha_values,
            full_logprob_steered_accuracy.mean(axis=0),
            marker="o",
            linewidth=2,
            label="Steered",
        )
        plt.xlabel("Alpha")
        plt.ylabel("Mean Accuracy")
        plt.title("Baseline vs Steered Full-Answer Logprob Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(out_dir, "full_logprob_baseline_vs_steered_accuracy.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(config.alpha_values, full_logprob_accuracy_gain.mean(axis=0), marker="o", linewidth=2)
        plt.axhline(y=0.0, color="gray", linestyle="--")
        plt.xlabel("Alpha")
        plt.ylabel("Mean Accuracy Gain vs Baseline")
        plt.title("Full-Answer Logprob Accuracy Gain from Steering")
        plt.grid(True)
        plt.savefig(
            os.path.join(out_dir, "full_logprob_accuracy_gain_vs_baseline.png"),
            dpi=300,
            bbox_inches="tight",
        )
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

        print(
            "Best overall alpha by mean accuracy gain: "
            f"{evaluation_summary['best_overall_alpha_by_mean_accuracy_gain']['alpha']} "
            f"(gain={evaluation_summary['best_overall_alpha_by_mean_accuracy_gain']['mean_accuracy_gain_vs_baseline']:.4f}, "
            f"baseline={evaluation_summary['best_overall_alpha_by_mean_accuracy_gain']['mean_baseline_accuracy']:.4f}, "
            f"steered={evaluation_summary['best_overall_alpha_by_mean_accuracy_gain']['mean_accuracy_at_alpha']:.4f})"
        )

    print(f"Evaluation complete. Results saved to {out_dir}")
