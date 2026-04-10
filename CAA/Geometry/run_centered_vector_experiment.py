import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .config import PipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from .data_loader import DataLoader
from .evaluate import evaluate_steering
from .geometry import analyze_geometry
from .model_loader import load_model
from .steering.caa import CAASteeringMethod


def _load_vectors(run_output_dir: str, model_name_safe: str) -> Dict[str, Dict[int, torch.Tensor]]:
    vec_dir = os.path.join(run_output_dir, model_name_safe, "vectors")
    vectors_all = {}

    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        val_safe = safe_name(val)
        val_vec_dir = os.path.join(vec_dir, val_safe)
        vectors_all[val] = {}
        if os.path.exists(val_vec_dir):
            for f in os.listdir(val_vec_dir):
                if f.startswith("layer_") and f.endswith(".pt"):
                    l_idx = int(f.split("_")[1].split(".")[0])
                    vectors_all[val][l_idx] = torch.load(os.path.join(val_vec_dir, f))

    return vectors_all


def _load_selected_layer(run_output_dir: str, model_name_safe: str) -> int:
    path = os.path.join(run_output_dir, model_name_safe, "layer_selection", "selected_layer.json")
    with open(path) as f:
        return json.load(f)["selected_layer"]


def _center_vectors(vectors_all: Dict[str, Dict[int, torch.Tensor]]) -> Dict[str, Dict[int, torch.Tensor]]:
    centered = {val: {} for val in SCHWARTZ_CIRCUMPLEX_ORDER}
    layers = sorted(vectors_all[SCHWARTZ_CIRCUMPLEX_ORDER[0]].keys())

    for layer_idx in layers:
        layer_mean = torch.stack([vectors_all[val][layer_idx].float() for val in SCHWARTZ_CIRCUMPLEX_ORDER]).mean(dim=0)
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            centered[val][layer_idx] = vectors_all[val][layer_idx].float() - layer_mean

    return centered


def _renormalize_vectors(vectors_all: Dict[str, Dict[int, torch.Tensor]]) -> Dict[str, Dict[int, torch.Tensor]]:
    renormed = {val: {} for val in SCHWARTZ_CIRCUMPLEX_ORDER}
    layers = sorted(vectors_all[SCHWARTZ_CIRCUMPLEX_ORDER[0]].keys())

    for layer_idx in layers:
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            vec = vectors_all[val][layer_idx].float()
            norm = vec.norm().clamp_min(1e-12)
            renormed[val][layer_idx] = vec / norm

    return renormed


def _save_vectors(vectors_all: Dict[str, Dict[int, torch.Tensor]], output_dir: str, model_name_safe: str):
    vec_dir = os.path.join(output_dir, model_name_safe, "vectors")
    os.makedirs(vec_dir, exist_ok=True)

    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        val_safe = safe_name(val)
        val_vec_dir = os.path.join(vec_dir, val_safe)
        os.makedirs(val_vec_dir, exist_ok=True)
        for layer_idx, vec_tensor in vectors_all[val].items():
            torch.save(vec_tensor, os.path.join(val_vec_dir, f"layer_{layer_idx}.pt"))


def _load_eval_results(path: str):
    with open(path) as f:
        return json.load(f)


def _extract_baseline_and_steered(results: Dict):
    sample_value = next(iter(results.values()))

    if "baseline" in sample_value and "steered" in sample_value:
        baseline = {val: results[val]["baseline"]["accuracy"] for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in results}
        alpha_labels = sorted(sample_value["steered"].keys(), key=lambda x: float(x))
        steered = {
            alpha: {val: results[val]["steered"][alpha]["accuracy"] for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in results}
            for alpha in alpha_labels
        }
        return baseline, steered, alpha_labels

    # Backward compatibility with older evaluation schema: {value: {alpha: metrics}}
    alpha_labels = sorted(sample_value.keys(), key=lambda x: float(x))
    steered = {
        alpha: {val: results[val][alpha]["accuracy"] for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in results}
        for alpha in alpha_labels
    }
    baseline = {}
    return baseline, steered, alpha_labels


def _compare_results(previous_results: Dict, centered_results: Dict) -> Dict:
    prev_baseline, prev_steered, prev_alpha_labels = _extract_baseline_and_steered(previous_results)
    centered_baseline, centered_steered, centered_alpha_labels = _extract_baseline_and_steered(centered_results)
    alpha_labels = sorted(set(prev_alpha_labels) & set(centered_alpha_labels), key=lambda x: float(x))

    if not alpha_labels:
        raise ValueError(
            "No overlapping alpha values were found between the previous run and the centered run. "
            f"Previous alphas: {prev_alpha_labels}; centered alphas: {centered_alpha_labels}"
        )

    value_order = [val for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in centered_results]
    comparison = {
        "baseline_accuracy": {val: centered_baseline[val] for val in value_order},
        "previous_alpha_values": prev_alpha_labels,
        "centered_alpha_values": centered_alpha_labels,
        "compared_alpha_values": alpha_labels,
        "previous_steered_accuracy": {},
        "centered_steered_accuracy": {},
        "centered_minus_previous_accuracy": {},
        "mean_accuracy": {
            "baseline": float(np.mean([centered_baseline[val] for val in value_order])),
            "previous_steered": {},
            "centered_steered": {},
            "centered_minus_previous": {},
        },
    }

    if prev_baseline:
        comparison["previous_baseline_accuracy"] = {val: prev_baseline[val] for val in value_order}

    for alpha in alpha_labels:
        comparison["previous_steered_accuracy"][alpha] = {val: prev_steered[alpha][val] for val in value_order}
        comparison["centered_steered_accuracy"][alpha] = {val: centered_steered[alpha][val] for val in value_order}
        comparison["centered_minus_previous_accuracy"][alpha] = {
            val: centered_steered[alpha][val] - prev_steered[alpha][val] for val in value_order
        }
        comparison["mean_accuracy"]["previous_steered"][alpha] = float(
            np.mean([prev_steered[alpha][val] for val in value_order])
        )
        comparison["mean_accuracy"]["centered_steered"][alpha] = float(
            np.mean([centered_steered[alpha][val] for val in value_order])
        )
        comparison["mean_accuracy"]["centered_minus_previous"][alpha] = (
            comparison["mean_accuracy"]["centered_steered"][alpha]
            - comparison["mean_accuracy"]["previous_steered"][alpha]
        )

    return comparison


def _save_comparison_artifacts(comparison: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "comparison_eval_results.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    alpha_labels = sorted(comparison["previous_steered_accuracy"].keys(), key=lambda x: float(x))
    alpha_values = [float(alpha) for alpha in alpha_labels]
    value_order = list(comparison["baseline_accuracy"].keys())

    baseline_mean = comparison["mean_accuracy"]["baseline"]
    prev_mean = np.array([comparison["mean_accuracy"]["previous_steered"][alpha] for alpha in alpha_labels])
    centered_mean = np.array([comparison["mean_accuracy"]["centered_steered"][alpha] for alpha in alpha_labels])
    delta_mean = np.array([comparison["mean_accuracy"]["centered_minus_previous"][alpha] for alpha in alpha_labels])

    plt.figure(figsize=(8, 5))
    plt.axhline(y=baseline_mean, color="gray", linestyle="--", linewidth=2, label="Baseline")
    plt.plot(alpha_values, prev_mean, marker="o", linewidth=2, label="Previous Steered")
    plt.plot(alpha_values, centered_mean, marker="s", linewidth=2, label="Centered Steered")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Accuracy")
    plt.title("Baseline vs Previous vs Centered Steering")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mean_accuracy_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(alpha_values, delta_mean, marker="o", linewidth=2)
    plt.axhline(y=0.0, color="gray", linestyle="--")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Accuracy Delta")
    plt.title("Centered Minus Previous Steering Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mean_accuracy_delta.png"), dpi=300, bbox_inches="tight")
    plt.close()

    delta_matrix = np.array(
        [
            [comparison["centered_minus_previous_accuracy"][alpha][val] for alpha in alpha_labels]
            for val in value_order
        ]
    )
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        delta_matrix,
        xticklabels=alpha_labels,
        yticklabels=value_order,
        cmap="coolwarm",
        center=0.0,
    )
    plt.xlabel("Alpha")
    plt.ylabel("Value")
    plt.title("Centered Minus Previous Accuracy by Value and Alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_delta_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_centered_vector_experiment(
    model_name: str,
    dataset_path: str,
    relations_path: str,
    source_output_dir: str,
    experiment_output_dir: str,
    alpha_values,
    transform: str,
    run_geometry: bool,
    geometry_only: bool,
):
    source_config = PipelineConfig(
        model_name=model_name,
        dataset_path=dataset_path,
        relations_path=relations_path,
        output_dir=source_output_dir,
        alpha_values=alpha_values,
    )
    experiment_config = PipelineConfig(
        model_name=model_name,
        dataset_path=dataset_path,
        relations_path=relations_path,
        output_dir=experiment_output_dir,
        alpha_values=alpha_values,
    )

    model_name_safe = source_config.model_name_safe
    selected_layer = _load_selected_layer(source_output_dir, model_name_safe)
    original_vectors = _load_vectors(source_output_dir, model_name_safe)
    transformed_vectors = _center_vectors(original_vectors)
    if transform == "centered_renorm":
        transformed_vectors = _renormalize_vectors(transformed_vectors)
    elif transform != "centered":
        raise ValueError(f"Unknown transform: {transform}. Expected one of: centered, centered_renorm.")

    _save_vectors(transformed_vectors, experiment_output_dir, model_name_safe)
    experiment_config.save()
    metadata_path = os.path.join(experiment_output_dir, model_name_safe, "experiment_metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "transform": transform,
                "source_output_dir": source_output_dir,
                "selected_layer": selected_layer,
                "alpha_values": alpha_values,
                "geometry_only": geometry_only,
            },
            f,
            indent=2,
        )

    target_vectors = {val: transformed_vectors[val][selected_layer] for val in SCHWARTZ_CIRCUMPLEX_ORDER}
    if not geometry_only:
        model_info = load_model(model_name, device=experiment_config.device)
        data_loader = DataLoader(dataset_path, eval_split=experiment_config.eval_split, seed=experiment_config.seed)
        steering_method = CAASteeringMethod()
        evaluate_steering(experiment_config, data_loader, model_info, steering_method, target_vectors, selected_layer)

    if run_geometry:
        analyze_geometry(experiment_config, target_vectors)

    if not geometry_only:
        previous_eval_path = os.path.join(source_output_dir, model_name_safe, "evaluation", "eval_results.json")
        centered_eval_path = os.path.join(experiment_output_dir, model_name_safe, "evaluation", "eval_results.json")
        previous_results = _load_eval_results(previous_eval_path)
        centered_results = _load_eval_results(centered_eval_path)
        comparison = _compare_results(previous_results, centered_results)

        comparison_dir = os.path.join(experiment_output_dir, model_name_safe, "comparison")
        _save_comparison_artifacts(comparison, comparison_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--relations_path", type=str, required=True)
    parser.add_argument("--source_output_dir", type=str, required=True, help="Existing pipeline run root to load vectors/layer/eval from")
    parser.add_argument(
        "--experiment_output_dir",
        type=str,
        default=None,
        help="Separate output root for centered-vector experiment; defaults to <source_output_dir>_centered",
    )
    parser.add_argument("--alpha", type=str, default="0.5,1.0,2.0,4.0", help="Comma-separated alphas")
    parser.add_argument("--transform", type=str, default="centered", help="centered or centered_renorm")
    parser.add_argument("--run_geometry", action="store_true")
    parser.add_argument("--geometry_only", action="store_true", help="Recompute transformed vectors and geometry only; skip evaluation and comparison")
    args = parser.parse_args()

    default_suffix = "_centered_renorm" if args.transform == "centered_renorm" else "_centered"
    experiment_output_dir = args.experiment_output_dir or f"{args.source_output_dir}{default_suffix}"

    run_centered_vector_experiment(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        relations_path=args.relations_path,
        source_output_dir=args.source_output_dir,
        experiment_output_dir=experiment_output_dir,
        alpha_values=[float(a) for a in args.alpha.split(",")],
        transform=args.transform,
        run_geometry=args.run_geometry,
        geometry_only=args.geometry_only,
    )
