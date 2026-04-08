"""
Variance-Filter Experiment
==========================
Removes shared low-variance dimensions from steering vectors using a
permutation-based null distribution threshold.  Supports two modes:

  pre_steering   Filter vectors BEFORE steering.  Runs full evaluation +
                 optional geometry on the filtered vectors and compares
                 accuracy against the original run.

  geometry_only  Filter vectors AFTER extraction (no model needed).
                 Runs only geometry visualisations (UMAP, t-SNE, MDS
                 circumplex, similarity heatmap, Spearman report) on the
                 filtered vectors and saves them alongside the originals
                 for direct comparison.  Useful for inspecting geometric
                 structure without the cost of re-running steering eval.

Method
------
1. Stack all B value vectors into a matrix V of shape [B × d].
2. Compute real per-dimension variance across behaviors.
3. Build a null distribution: shuffle value labels randomly, recompute
   per-dimension variance, repeat n_permutations times.
4. Flag a dimension as shared/noise if its real variance < the
   null_percentile-th percentile of the null distribution for that dim.
5. Zero out those dimensions (project onto the complement subspace).
6. Proceed with evaluation (pre_steering) or geometry only (geometry_only).

Usage (CLI)
-----------
# Pre-steering mode (filter then evaluate)
python -m CAA.Geometry.run_variance_filter_experiment \\
    --mode pre_steering \\
    --model_name meta-llama/Llama-2-7b-chat-hf \\
    --dataset_path CAA/value_data/final_dataset_200.csv \\
    --relations_path CAA/Geometry/outputs/relations.json \\
    --source_output_dir CAA/Geometry/outputs/my_run \\
    --experiment_output_dir CAA/Geometry/outputs/my_run_variance_filtered \\
    --null_percentile 50 --n_permutations 1000 \\
    --alpha 0.5,1.0,2.0,4.0 --run_geometry

# Geometry-only mode (filter then visualise, no model needed)
python -m CAA.Geometry.run_variance_filter_experiment \\
    --mode geometry_only \\
    --relations_path CAA/Geometry/outputs/relations.json \\
    --source_output_dir CAA/Geometry/outputs/my_run \\
    --model_name meta-llama/Llama-2-7b-chat-hf \\
    --null_percentile 50 --n_permutations 1000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_vectors(run_output_dir: str, model_name_safe: str) -> Dict[str, Dict[int, torch.Tensor]]:
    vec_dir = os.path.join(run_output_dir, model_name_safe, "vectors")
    vectors_all: Dict[str, Dict[int, torch.Tensor]] = {}

    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        val_safe = safe_name(val)
        val_vec_dir = os.path.join(vec_dir, val_safe)
        vectors_all[val] = {}
        if os.path.exists(val_vec_dir):
            for fname in os.listdir(val_vec_dir):
                if fname.startswith("layer_") and fname.endswith(".pt"):
                    l_idx = int(fname.split("_")[1].split(".")[0])
                    vectors_all[val][l_idx] = torch.load(os.path.join(val_vec_dir, fname))

    return vectors_all


def _load_selected_layer(run_output_dir: str, model_name_safe: str) -> int:
    path = os.path.join(run_output_dir, model_name_safe, "layer_selection", "selected_layer.json")
    with open(path) as f:
        return json.load(f)["selected_layer"]


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


def _copy_layer_selection(source_output_dir: str, dest_output_dir: str, model_name_safe: str):
    src = os.path.join(source_output_dir, model_name_safe, "layer_selection", "selected_layer.json")
    dst_dir = os.path.join(dest_output_dir, model_name_safe, "layer_selection")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "selected_layer.json")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Core variance-filter logic
# ---------------------------------------------------------------------------

def _build_matrix(
    vectors: Dict[str, Dict[int, torch.Tensor]],
    layer_idx: int,
    values: List[str],
) -> np.ndarray:
    """Stack value vectors into a [B × d] float32 matrix."""
    return np.stack(
        [vectors[val][layer_idx].float().cpu().numpy() for val in values],
        axis=0,
    )


def _per_dim_variance(matrix: np.ndarray) -> np.ndarray:
    """Variance of each dimension across rows (behaviors). Returns shape [d]."""
    return matrix.var(axis=0)


def _null_variance_distribution(
    matrix: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Shuffle row order (value labels) n_permutations times and collect
    per-dimension variances. Returns shape [n_permutations × d].

    Each permutation reorders rows so marginal distributions are unchanged but
    inter-value structure is destroyed — giving a null for 'what variance looks
    like with no real behavioral signal'.
    """
    B, d = matrix.shape
    null_variances = np.empty((n_permutations, d), dtype=np.float32)
    for i in range(n_permutations):
        shuffled = matrix[rng.permutation(B), :]
        null_variances[i] = shuffled.var(axis=0)
    return null_variances


def compute_variance_filter_mask(
    matrix: np.ndarray,
    n_permutations: int = 1000,
    null_percentile: float = 50.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify low-variance dimensions indistinguishable from noise.

    Parameters
    ----------
    matrix : np.ndarray, shape [B × d]
    n_permutations : int
    null_percentile : float
        Remove a dimension if its real variance < this percentile of the null
        distribution. 50 = median (moderate), 95 = conservative.
    seed : int

    Returns
    -------
    keep_mask       : bool array [d]  — True = keep
    real_var        : float array [d]
    null_thresholds : float array [d]
    null_variances  : float array [n_permutations × d]
    """
    rng = np.random.default_rng(seed)
    real_var = _per_dim_variance(matrix)
    null_variances = _null_variance_distribution(matrix, n_permutations, rng)
    null_thresholds = np.percentile(null_variances, null_percentile, axis=0)
    keep_mask = real_var >= null_thresholds
    return keep_mask, real_var, null_thresholds, null_variances


def apply_variance_filter(
    vectors: Dict[str, Dict[int, torch.Tensor]],
    n_permutations: int = 1000,
    null_percentile: float = 50.0,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[int, dict]]:
    """
    Apply the variance filter to all layers.

    Returns filtered_vectors (low-variance dims zeroed out) and filter_stats
    (per-layer diagnostics).
    """
    layers = sorted(next(iter(vectors.values())).keys())
    filtered: Dict[str, Dict[int, torch.Tensor]] = {val: {} for val in SCHWARTZ_CIRCUMPLEX_ORDER}
    filter_stats: Dict[int, dict] = {}

    for layer_idx in layers:
        matrix = _build_matrix(vectors, layer_idx, SCHWARTZ_CIRCUMPLEX_ORDER)
        keep_mask, real_var, null_thresholds, null_variances = compute_variance_filter_mask(
            matrix, n_permutations=n_permutations, null_percentile=null_percentile, seed=seed
        )

        n_total = len(keep_mask)
        n_kept = int(keep_mask.sum())
        n_removed = n_total - n_kept

        filter_stats[layer_idx] = {
            "n_dims_total": n_total,
            "n_dims_kept": n_kept,
            "n_dims_removed": n_removed,
            "frac_removed": float(n_removed / n_total),
            "mean_real_var": float(real_var.mean()),
            "mean_null_threshold": float(null_thresholds.mean()),
            "keep_mask": keep_mask.tolist(),
        }

        print(
            f"  Layer {layer_idx}: removed {n_removed}/{n_total} dims "
            f"({100 * n_removed / n_total:.1f}%) below null-{null_percentile:.0f}th-pct threshold"
        )

        mask_tensor = torch.tensor(keep_mask, dtype=torch.float32)
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            original = vectors[val][layer_idx].float().cpu()
            filtered[val][layer_idx] = original * mask_tensor

    return filtered, filter_stats


# ---------------------------------------------------------------------------
# Diagnostic plots for the filter itself
# ---------------------------------------------------------------------------

def _save_filter_diagnostics(
    filter_stats: Dict[int, dict],
    vectors: Dict[str, Dict[int, torch.Tensor]],
    selected_layer: int,
    out_dir: str,
    n_permutations: int,
    null_percentile: float,
    seed: int,
):
    """Save diagnostic figures for the variance filter at the selected layer."""
    os.makedirs(out_dir, exist_ok=True)

    layer_ids = sorted(filter_stats.keys())
    frac_removed = [filter_stats[l]["frac_removed"] for l in layer_ids]

    # fraction of dims removed per layer
    plt.figure(figsize=(8, 4))
    plt.bar(layer_ids, frac_removed, color="#2196F3", alpha=0.8)
    plt.axvline(x=selected_layer, color="red", linestyle="--", linewidth=1.5,
                label=f"Selected layer {selected_layer}")
    plt.xlabel("Layer")
    plt.ylabel("Fraction of dims removed")
    plt.title(f"Variance-filter removal rate (null pct={null_percentile})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "removal_rate_per_layer.png"), dpi=300)
    plt.close()

    # real variance vs null threshold for selected layer
    matrix = _build_matrix(vectors, selected_layer, SCHWARTZ_CIRCUMPLEX_ORDER)
    keep_mask, real_var, null_thresholds, null_variances = compute_variance_filter_mask(
        matrix, n_permutations=n_permutations, null_percentile=null_percentile, seed=seed
    )

    d = len(real_var)
    sorted_idx = np.argsort(real_var)

    plt.figure(figsize=(10, 4))
    plt.plot(range(d), real_var[sorted_idx], color="#2196F3", linewidth=0.8, label="Real variance")
    plt.plot(range(d), null_thresholds[sorted_idx], color="#F44336", linewidth=0.8,
             linestyle="--", label=f"Null {null_percentile:.0f}th pct")
    plt.fill_between(range(d), null_thresholds[sorted_idx], real_var[sorted_idx],
                     where=(real_var[sorted_idx] >= null_thresholds[sorted_idx]),
                     alpha=0.2, color="#4CAF50", label="Kept dims")
    plt.fill_between(range(d), real_var[sorted_idx], null_thresholds[sorted_idx],
                     where=(real_var[sorted_idx] < null_thresholds[sorted_idx]),
                     alpha=0.2, color="#F44336", label="Removed dims")
    plt.xlabel("Dimension (sorted by real variance)")
    plt.ylabel("Variance")
    plt.title(f"Real vs Null Variance — Layer {selected_layer}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "real_vs_null_variance.png"), dpi=300)
    plt.close()

    # histogram: null distribution mean vs real mean
    null_means = null_variances.mean(axis=1)
    plt.figure(figsize=(6, 4))
    plt.hist(null_means, bins=40, color="#90CAF9", edgecolor="white", label="Null mean variance")
    plt.axvline(real_var.mean(), color="#F44336", linewidth=2,
                label=f"Real mean var = {real_var.mean():.4f}")
    plt.xlabel("Mean per-dim variance")
    plt.ylabel("Count")
    plt.title(f"Null distribution of mean variance — Layer {selected_layer}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "null_variance_histogram.png"), dpi=300)
    plt.close()

    # vector norms before / after filtering
    mask_tensor = torch.tensor(keep_mask, dtype=torch.float32)
    original_norms = np.array([
        vectors[val][selected_layer].float().cpu().norm().item()
        for val in SCHWARTZ_CIRCUMPLEX_ORDER
    ])
    filtered_norms = np.array([
        (vectors[val][selected_layer].float().cpu() * mask_tensor).norm().item()
        for val in SCHWARTZ_CIRCUMPLEX_ORDER
    ])
    norm_retention = filtered_norms / np.maximum(original_norms, 1e-12)

    x = np.arange(len(SCHWARTZ_CIRCUMPLEX_ORDER))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, original_norms, width, label="Original", color="#90CAF9")
    plt.bar(x + width / 2, filtered_norms, width, label="Filtered", color="#A5D6A7")
    plt.xticks(x, [v.split(":")[-1].strip() for v in SCHWARTZ_CIRCUMPLEX_ORDER],
               rotation=45, ha="right", fontsize=7)
    plt.ylabel("L2 Norm")
    plt.title(f"Vector norm before / after filtering — Layer {selected_layer}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vector_norms_before_after.png"), dpi=300)
    plt.close()

    summary = {
        "selected_layer": selected_layer,
        "null_percentile": null_percentile,
        "n_permutations": n_permutations,
        "seed": seed,
        "per_layer_stats": {str(l): filter_stats[l] for l in layer_ids},
        "selected_layer_norm_retention": {
            val: float(norm_retention[i]) for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER)
        },
        "selected_layer_mean_norm_retention": float(norm_retention.mean()),
    }
    with open(os.path.join(out_dir, "filter_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Filter diagnostics saved to {out_dir}")


# ---------------------------------------------------------------------------
# Evaluation comparison helpers
# ---------------------------------------------------------------------------

def _extract_baseline_and_steered(results: Dict):
    sample_value = next(iter(results.values()))

    if "baseline" in sample_value and "steered" in sample_value:
        baseline = {val: results[val]["baseline"]["accuracy"]
                    for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in results}
        alpha_labels = sorted(sample_value["steered"].keys(), key=lambda x: float(x))
        steered = {
            alpha: {val: results[val]["steered"][alpha]["accuracy"]
                    for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in results}
            for alpha in alpha_labels
        }
        return baseline, steered, alpha_labels

    alpha_labels = sorted(sample_value.keys(), key=lambda x: float(x))
    steered = {
        alpha: {val: results[val][alpha]["accuracy"]
                for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in results}
        for alpha in alpha_labels
    }
    return {}, steered, alpha_labels


def _compare_results(original_results: Dict, filtered_results: Dict) -> Dict:
    orig_baseline, orig_steered, orig_alphas = _extract_baseline_and_steered(original_results)
    filt_baseline, filt_steered, filt_alphas = _extract_baseline_and_steered(filtered_results)
    alpha_labels = sorted(set(orig_alphas) & set(filt_alphas), key=lambda x: float(x))

    if not alpha_labels:
        raise ValueError(
            "No overlapping alpha values between original and filtered runs. "
            f"Original: {orig_alphas}; Filtered: {filt_alphas}"
        )

    value_order = [val for val in SCHWARTZ_CIRCUMPLEX_ORDER if val in filtered_results]
    comparison = {
        "baseline_accuracy": {val: filt_baseline[val] for val in value_order if val in filt_baseline},
        "original_alpha_values": orig_alphas,
        "filtered_alpha_values": filt_alphas,
        "compared_alpha_values": alpha_labels,
        "original_steered_accuracy": {},
        "filtered_steered_accuracy": {},
        "filtered_minus_original_accuracy": {},
        "mean_accuracy": {
            "baseline": (
                float(np.mean([filt_baseline[val] for val in value_order if val in filt_baseline]))
                if filt_baseline else None
            ),
            "original_steered": {},
            "filtered_steered": {},
            "filtered_minus_original": {},
        },
    }

    if orig_baseline:
        comparison["original_baseline_accuracy"] = {
            val: orig_baseline[val] for val in value_order if val in orig_baseline
        }

    for alpha in alpha_labels:
        comparison["original_steered_accuracy"][alpha] = {
            val: orig_steered[alpha][val]
            for val in value_order if val in orig_steered.get(alpha, {})
        }
        comparison["filtered_steered_accuracy"][alpha] = {
            val: filt_steered[alpha][val]
            for val in value_order if val in filt_steered.get(alpha, {})
        }
        comparison["filtered_minus_original_accuracy"][alpha] = {
            val: comparison["filtered_steered_accuracy"][alpha].get(val, 0)
                 - comparison["original_steered_accuracy"][alpha].get(val, 0)
            for val in value_order
        }
        comparison["mean_accuracy"]["original_steered"][alpha] = float(
            np.mean(list(comparison["original_steered_accuracy"][alpha].values()))
        )
        comparison["mean_accuracy"]["filtered_steered"][alpha] = float(
            np.mean(list(comparison["filtered_steered_accuracy"][alpha].values()))
        )
        comparison["mean_accuracy"]["filtered_minus_original"][alpha] = (
            comparison["mean_accuracy"]["filtered_steered"][alpha]
            - comparison["mean_accuracy"]["original_steered"][alpha]
        )

    return comparison


def _save_comparison_artifacts(comparison: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "comparison_eval_results.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    alpha_labels = sorted(comparison["original_steered_accuracy"].keys(), key=lambda x: float(x))
    alpha_values = [float(a) for a in alpha_labels]
    value_order = list(comparison["filtered_minus_original_accuracy"][alpha_labels[0]].keys())

    orig_mean = np.array([comparison["mean_accuracy"]["original_steered"][a] for a in alpha_labels])
    filt_mean = np.array([comparison["mean_accuracy"]["filtered_steered"][a] for a in alpha_labels])
    delta_mean = np.array([comparison["mean_accuracy"]["filtered_minus_original"][a] for a in alpha_labels])

    plt.figure(figsize=(8, 5))
    if comparison["mean_accuracy"].get("baseline") is not None:
        plt.axhline(y=comparison["mean_accuracy"]["baseline"],
                    color="gray", linestyle="--", linewidth=2, label="Baseline")
    plt.plot(alpha_values, orig_mean, marker="o", linewidth=2, label="Original Steered")
    plt.plot(alpha_values, filt_mean, marker="s", linewidth=2, label="Variance-Filtered Steered")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Accuracy")
    plt.title("Original vs Variance-Filtered Steering")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mean_accuracy_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(alpha_values, delta_mean, marker="o", linewidth=2)
    plt.axhline(y=0.0, color="gray", linestyle="--")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Accuracy Delta")
    plt.title("Variance-Filtered Minus Original Steering Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mean_accuracy_delta.png"), dpi=300, bbox_inches="tight")
    plt.close()

    delta_matrix = np.array([
        [comparison["filtered_minus_original_accuracy"][a][val] for a in alpha_labels]
        for val in value_order
    ])
    plt.figure(figsize=(10, 10))
    sns.heatmap(delta_matrix, xticklabels=alpha_labels, yticklabels=value_order,
                cmap="coolwarm", center=0.0)
    plt.xlabel("Alpha")
    plt.ylabel("Value")
    plt.title("Variance-Filtered Minus Original Accuracy by Value and Alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_delta_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Mode A — pre_steering
# ---------------------------------------------------------------------------

def run_pre_steering_mode(
    model_name: str,
    dataset_path: str,
    relations_path: str,
    source_output_dir: str,
    experiment_output_dir: str,
    alpha_values: List[float],
    n_permutations: int,
    null_percentile: float,
    run_geometry: bool,
    seed: int,
):
    """
    Filter vectors → save → run steering evaluation → compare with original.
    Geometry visualisations on the filtered vectors are optional.
    """
    source_config = PipelineConfig(
        model_name=model_name, dataset_path=dataset_path,
        relations_path=relations_path, output_dir=source_output_dir,
        alpha_values=alpha_values, seed=seed,
    )
    experiment_config = PipelineConfig(
        model_name=model_name, dataset_path=dataset_path,
        relations_path=relations_path, output_dir=experiment_output_dir,
        alpha_values=alpha_values, seed=seed,
    )

    model_name_safe = source_config.model_name_safe
    selected_layer = _load_selected_layer(source_output_dir, model_name_safe)

    print(f"[pre_steering] Loading original vectors from {source_output_dir} ...")
    original_vectors = _load_vectors(source_output_dir, model_name_safe)

    print(f"[pre_steering] Applying variance filter "
          f"(n_permutations={n_permutations}, null_percentile={null_percentile}) ...")
    filtered_vectors, filter_stats = apply_variance_filter(
        original_vectors, n_permutations=n_permutations,
        null_percentile=null_percentile, seed=seed,
    )

    _save_vectors(filtered_vectors, experiment_output_dir, model_name_safe)
    experiment_config.save()
    _copy_layer_selection(source_output_dir, experiment_output_dir, model_name_safe)

    diag_dir = os.path.join(experiment_output_dir, model_name_safe, "filter_diagnostics")
    _save_filter_diagnostics(
        filter_stats=filter_stats, vectors=original_vectors,
        selected_layer=selected_layer, out_dir=diag_dir,
        n_permutations=n_permutations, null_percentile=null_percentile, seed=seed,
    )

    print("[pre_steering] Loading model for evaluation ...")
    model_info = load_model(model_name, device=experiment_config.device)
    data_loader = DataLoader(dataset_path, eval_split=experiment_config.eval_split, seed=seed)
    steering_method = CAASteeringMethod()

    target_vectors = {val: filtered_vectors[val][selected_layer] for val in SCHWARTZ_CIRCUMPLEX_ORDER}
    evaluate_steering(experiment_config, data_loader, model_info, steering_method,
                      target_vectors, selected_layer)

    if run_geometry:
        analyze_geometry(experiment_config, target_vectors)

    original_eval_path = os.path.join(source_output_dir, model_name_safe, "evaluation", "eval_results.json")
    filtered_eval_path = os.path.join(experiment_output_dir, model_name_safe, "evaluation", "eval_results.json")

    if os.path.exists(original_eval_path) and os.path.exists(filtered_eval_path):
        comparison = _compare_results(
            _load_eval_results(original_eval_path),
            _load_eval_results(filtered_eval_path),
        )
        comparison_dir = os.path.join(experiment_output_dir, model_name_safe, "comparison")
        _save_comparison_artifacts(comparison, comparison_dir)
        print(f"[pre_steering] Comparison artifacts saved to {comparison_dir}")
    else:
        print("[pre_steering] Original eval results not found — skipping comparison plots.")

    print("[pre_steering] Done.")


# ---------------------------------------------------------------------------
# Mode B — geometry_only
# ---------------------------------------------------------------------------

def run_geometry_only_mode(
    model_name: str,
    relations_path: str,
    source_output_dir: str,
    experiment_output_dir: str,
    n_permutations: int,
    null_percentile: float,
    seed: int,
    dataset_path: str = "",
):
    """
    Filter vectors → run geometry visualisations only (no model, no eval).

    Outputs land in <experiment_output_dir>/<model_safe>/geometry_variance_filtered/
    so they sit next to the original geometry/ folder and can be compared directly.
    """
    source_config = PipelineConfig(
        model_name=model_name, dataset_path=dataset_path,
        relations_path=relations_path, output_dir=source_output_dir,
        seed=seed,
    )
    # We reuse PipelineConfig only to get the geometry subdir path, but we
    # override output_dir so plots go into a clearly named subdirectory.
    experiment_config = PipelineConfig(
        model_name=model_name, dataset_path=dataset_path,
        relations_path=relations_path, output_dir=experiment_output_dir,
        seed=seed,
    )

    model_name_safe = source_config.model_name_safe
    selected_layer = _load_selected_layer(source_output_dir, model_name_safe)

    print(f"[geometry_only] Loading original vectors from {source_output_dir} ...")
    original_vectors = _load_vectors(source_output_dir, model_name_safe)

    print(f"[geometry_only] Applying variance filter "
          f"(n_permutations={n_permutations}, null_percentile={null_percentile}) ...")
    filtered_vectors, filter_stats = apply_variance_filter(
        original_vectors, n_permutations=n_permutations,
        null_percentile=null_percentile, seed=seed,
    )

    # Save filter diagnostics
    diag_dir = os.path.join(experiment_output_dir, model_name_safe, "filter_diagnostics")
    _save_filter_diagnostics(
        filter_stats=filter_stats, vectors=original_vectors,
        selected_layer=selected_layer, out_dir=diag_dir,
        n_permutations=n_permutations, null_percentile=null_percentile, seed=seed,
    )

    # Copy layer selection so analyze_geometry can find it if needed
    _copy_layer_selection(source_output_dir, experiment_output_dir, model_name_safe)

    # Run geometry on filtered vectors at selected layer
    target_vectors = {val: filtered_vectors[val][selected_layer] for val in SCHWARTZ_CIRCUMPLEX_ORDER}
    print(f"[geometry_only] Running geometry analysis on variance-filtered vectors "
          f"(layer {selected_layer}) ...")
    analyze_geometry(experiment_config, target_vectors)

    # Also run geometry on the original vectors and save into a parallel subdir
    # so results are side by side.
    original_target = {val: original_vectors[val][selected_layer] for val in SCHWARTZ_CIRCUMPLEX_ORDER}
    original_geom_config = PipelineConfig(
        model_name=model_name, dataset_path=dataset_path,
        relations_path=relations_path,
        output_dir=os.path.join(experiment_output_dir, "_original_geometry_reference"),
        seed=seed,
    )
    print("[geometry_only] Running geometry analysis on original (unfiltered) vectors for reference ...")
    analyze_geometry(original_geom_config, original_target)

    print(f"[geometry_only] Filtered geometry → "
          f"{os.path.join(experiment_output_dir, model_name_safe, 'geometry')}")
    print(f"[geometry_only] Original reference → "
          f"{os.path.join(experiment_output_dir, '_original_geometry_reference', model_name_safe, 'geometry')}")
    print("[geometry_only] Done.")


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def run_variance_filter_experiment(
    mode: str,
    model_name: str,
    relations_path: str,
    source_output_dir: str,
    experiment_output_dir: str,
    n_permutations: int = 1000,
    null_percentile: float = 50.0,
    seed: int = 42,
    # pre_steering-only args
    dataset_path: str = "",
    alpha_values: List[float] = None,
    run_geometry: bool = False,
):
    """
    mode: 'pre_steering' or 'geometry_only'
    """
    if mode == "pre_steering":
        if not dataset_path:
            raise ValueError("--dataset_path is required for pre_steering mode.")
        if alpha_values is None:
            alpha_values = [0.5, 1.0, 2.0, 4.0]
        run_pre_steering_mode(
            model_name=model_name, dataset_path=dataset_path,
            relations_path=relations_path,
            source_output_dir=source_output_dir,
            experiment_output_dir=experiment_output_dir,
            alpha_values=alpha_values,
            n_permutations=n_permutations, null_percentile=null_percentile,
            run_geometry=run_geometry, seed=seed,
        )
    elif mode == "geometry_only":
        run_geometry_only_mode(
            model_name=model_name, relations_path=relations_path,
            source_output_dir=source_output_dir,
            experiment_output_dir=experiment_output_dir,
            n_permutations=n_permutations, null_percentile=null_percentile,
            seed=seed, dataset_path=dataset_path,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'pre_steering' or 'geometry_only'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Variance-filter experiment. "
            "Mode 'pre_steering': filter → evaluate → compare. "
            "Mode 'geometry_only': filter → geometry visualisations only (no model needed)."
        )
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["pre_steering", "geometry_only"],
        help="'pre_steering' or 'geometry_only'",
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--relations_path", type=str, required=True)
    parser.add_argument(
        "--source_output_dir", type=str, required=True,
        help="Root of an existing completed pipeline run.",
    )
    parser.add_argument(
        "--experiment_output_dir", type=str, default=None,
        help="Output root; defaults to <source_output_dir>_variance_filtered.",
    )
    # pre_steering-only
    parser.add_argument(
        "--dataset_path", type=str, default="",
        help="Required for pre_steering mode.",
    )
    parser.add_argument(
        "--alpha", type=str, default="0.5,1.0,2.0,4.0",
        help="Comma-separated steering alphas (pre_steering only).",
    )
    parser.add_argument(
        "--run_geometry", action="store_true",
        help="Also run geometry analysis after steering eval (pre_steering only).",
    )
    # shared
    parser.add_argument(
        "--null_percentile", type=float, default=50.0,
        help="Percentile of null distribution used as removal threshold (default 50 = median).",
    )
    parser.add_argument(
        "--n_permutations", type=int, default=1000,
        help="Number of label-shuffle permutations for the null distribution.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    experiment_output_dir = args.experiment_output_dir or f"{args.source_output_dir}_variance_filtered"

    run_variance_filter_experiment(
        mode=args.mode,
        model_name=args.model_name,
        relations_path=args.relations_path,
        source_output_dir=args.source_output_dir,
        experiment_output_dir=experiment_output_dir,
        n_permutations=args.n_permutations,
        null_percentile=args.null_percentile,
        seed=args.seed,
        dataset_path=args.dataset_path,
        alpha_values=[float(a) for a in args.alpha.split(",")],
        run_geometry=args.run_geometry,
    )
