"""
Geometry analysis on sparse persona vectors.

Produces two complete sets of visualisations and a Spearman ρ report:

  geometry_raw/      — sparse CAA vectors as extracted
  geometry_centered/ — mean-centred: v_c[val] = v[val] - mean(all 20 vecs)
                       (mean-centering is applied to visualisation only;
                        the steering evaluation always uses the raw vectors)

Each set contains:
  spearman_report.json          — ρ and p-value vs Schwartz theoretical matrix
  empirical_similarity_heatmap.png
  theoretical_similarity_heatmap.png
  mds_circumplex.png            — MDS aligned to ideal Schwartz circumplex
  umap_2d.png
  tsne_2d.png
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import umap
from matplotlib.lines import Line2D
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS, TSNE

from .config import (
    HIGHER_ORDER_GROUPS,
    GROUP_COLORS,
    SCHWARTZ_CIRCUMPLEX_ORDER,
    SparseCAAPipelineConfig,
    value_to_group,
)

PLOT_LABEL_FONTSIZE = 13
PLOT_TITLE_FONTSIZE = 18
PLOT_LEGEND_FONTSIZE = 13
PLOT_MARKER_SIZE = 150


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _short(val: str) -> str:
    return val.split(":")[-1].strip()


def _legend_handles() -> list:
    return [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=c, markersize=12,
            label=g,
        )
        for g, c in GROUP_COLORS.items()
    ]


def _unit_vecs(vectors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = {}
    for val, vec in vectors.items():
        v = vec.detach().cpu().float()
        n = v.norm()
        result[val] = v / n if n > 0 else v
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Matrix computations
# ──────────────────────────────────────────────────────────────────────────────
def _empirical_sim(unit_vecs: Dict[str, torch.Tensor]) -> np.ndarray:
    n = len(SCHWARTZ_CIRCUMPLEX_ORDER)
    mat = np.zeros((n, n))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            mat[i, j] = F.cosine_similarity(
                unit_vecs[v1], unit_vecs[v2], dim=0
            ).item()
    return mat


def _theoretical_sim(relations_path: str) -> np.ndarray:
    with open(relations_path) as f:
        rel = json.load(f)["basic_value_relationship_matrix"]
    n = len(SCHWARTZ_CIRCUMPLEX_ORDER)
    mat = np.zeros((n, n))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            if v1 in rel and v2 in rel[v1]:
                mat[i, j] = rel[v1][v2]
    return mat


def _spearman(emp: np.ndarray, theo: np.ndarray) -> Tuple[float, float]:
    n = emp.shape[0]
    triu = np.triu_indices(n, k=1)
    rho, pval = spearmanr(emp[triu], theo[triu])
    return float(rho), float(pval)


def _circular_step_distance(i: int, j: int, n: int) -> int:
    return min(abs(i - j), n - abs(i - j))


def _theoretical_circle_points(num_values: int) -> np.ndarray:
    """
    Place Schwartz values on the canonical circumplex.

    The fixed order in SCHWARTZ_CIRCUMPLEX_ORDER is the important part.
    We therefore use the standard evenly spaced unit-circle construction and
    let Procrustes choose the best rigid alignment for the empirical MDS
    solution. This yields more stable and visually faithful plots than
    hard-coding presentation quadrants into the template itself.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_values, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _lower_order_family(value: str) -> str:
    return value.split(":")[0].strip() if ":" in value else value


def _higher_order_groups_for_value(value: str) -> set[str]:
    boundary_groups = {
        "Hedonism": {"Openness to Change", "Self-Enhancement"},
        "Face": {"Self-Enhancement", "Conservation"},
        "Humility": {"Conservation", "Self-Transcendence"},
    }
    if value in boundary_groups:
        return boundary_groups[value]

    groups = set()
    for group_name, members in HIGHER_ORDER_GROUPS.items():
        if value in members:
            groups.add(group_name)
    return groups


def _groups_are_opposite(group_a: str, group_b: str) -> bool:
    opposite_pairs = {
        frozenset({"Openness to Change", "Conservation"}),
        frozenset({"Self-Enhancement", "Self-Transcendence"}),
    }
    return frozenset({group_a, group_b}) in opposite_pairs


def _hierarchical_theory_distance(value_a: str, value_b: str) -> tuple[int, str]:
    if _lower_order_family(value_a) == _lower_order_family(value_b):
        return 1, "same_lower_order"

    groups_a = _higher_order_groups_for_value(value_a)
    groups_b = _higher_order_groups_for_value(value_b)

    if groups_a & groups_b:
        return 2, "same_higher_order"

    if any(_groups_are_opposite(group_a, group_b) for group_a in groups_a for group_b in groups_b):
        return 10, "opposite_higher_order"

    return 5, "no_relation"


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def _plot_heatmap(mat: np.ndarray, title: str, path: str) -> None:
    labels = [_short(v) for v in SCHWARTZ_CIRCUMPLEX_ORDER]
    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        mat,
        xticklabels=labels,
        yticklabels=labels,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.3,
        linecolor="lightgrey",
    )
    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_embedding(coords: np.ndarray, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 11))
    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        color = GROUP_COLORS.get(value_to_group(val), "black")
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=PLOT_MARKER_SIZE, edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(
            _short(val),
            (coords[i, 0], coords[i, 1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=PLOT_LABEL_FONTSIZE,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
        )
    ax.legend(handles=_legend_handles(), loc="best", fontsize=PLOT_LEGEND_FONTSIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_mds(emp_sim: np.ndarray, title: str, path: str, seed: int) -> None:
    n = len(SCHWARTZ_CIRCUMPLEX_ORDER)
    dist = np.clip(1.0 - emp_sim, 0, None)

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=4,
        normalized_stress="auto",
    )
    X_mds = mds.fit_transform(dist)

    X_circle = _theoretical_circle_points(n)
    R, _ = orthogonal_procrustes(X_mds, X_circle)
    X_aligned = X_mds @ R

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.add_patch(plt.Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--"))

    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        tx, ty = X_circle[i]
        ex, ey = X_aligned[i]
        color = GROUP_COLORS.get(value_to_group(val), "black")

        ax.plot(tx, ty, "x", color="gray", markersize=9)
        ax.plot(ex, ey, "o", color=color, markersize=10, markeredgecolor="white", markeredgewidth=1.0)
        ax.plot([tx, ex], [ty, ey], color="gray", alpha=0.3, linestyle=":")
        ax.annotate(
            _short(val),
            (ex, ey),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=PLOT_LABEL_FONTSIZE,
            fontweight="semibold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.8),
        )

    ax.legend(handles=_legend_handles(), fontsize=PLOT_LEGEND_FONTSIZE)
    ax.set_title(f"{title}\n(grey ×: Schwartz theory,  coloured ●: empirical)", fontsize=PLOT_TITLE_FONTSIZE)
    ax.set_aspect("equal")
    lim = max(np.abs(X_aligned).max(), 1.0) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Per-variant analysis
# ──────────────────────────────────────────────────────────────────────────────
def _run_geometry_variant(
    vectors: Dict[str, torch.Tensor],
    theo_sim: np.ndarray,
    config: SparseCAAPipelineConfig,
    out_dir: str,
    label: str,
) -> dict:
    """Run full geometry analysis on one variant of the vectors."""
    unit_vecs = _unit_vecs(vectors)
    emp_sim = _empirical_sim(unit_vecs)
    rho, pval = _spearman(emp_sim, theo_sim)
    n = len(SCHWARTZ_CIRCUMPLEX_ORDER)
    triu = np.triu_indices(n, k=1)
    emp_flat = emp_sim[triu]
    theo_flat = theo_sim[triu]
    pearson_r, pearson_p = pearsonr(emp_flat, theo_flat)

    # Spearman report
    report = {
        "label": label,
        "spearman_rho": rho,
        "p_value": pval,
        "n_pairs": int(np.triu_indices(len(SCHWARTZ_CIRCUMPLEX_ORDER), k=1)[0].shape[0]),
    }
    with open(os.path.join(out_dir, "spearman_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [{label}] Spearman ρ = {rho:+.4f}   p = {pval:.4g}")

    # Heatmaps
    _plot_heatmap(
        emp_sim,
        f"Empirical Cosine Similarity – {label}",
        os.path.join(out_dir, "empirical_similarity_heatmap.png"),
    )
    _plot_heatmap(
        theo_sim,
        "Theoretical Schwartz Relationships",
        os.path.join(out_dir, "theoretical_similarity_heatmap.png"),
    )

    # 2-D projections
    # Stack unit vectors as a matrix (20, d_sae)
    X = np.stack([unit_vecs[v].numpy() for v in SCHWARTZ_CIRCUMPLEX_ORDER])

    # UMAP — n_neighbors must be < n_samples (20)
    reducer = umap.UMAP(n_components=2, metric="cosine",
                        n_neighbors=5, random_state=config.seed)
    X_umap = reducer.fit_transform(X)
    _plot_embedding(X_umap, f"UMAP – {label}", os.path.join(out_dir, "umap_2d.png"))

    # t-SNE — perplexity must be < n_samples (20); 5 is safe
    perplexity = min(5, len(SCHWARTZ_CIRCUMPLEX_ORDER) - 1)
    X_tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=config.seed,
    ).fit_transform(X)
    _plot_embedding(X_tsne, f"t-SNE – {label}", os.path.join(out_dir, "tsne_2d.png"))

    # MDS circumplex
    _plot_mds(
        emp_sim,
        f"MDS Circumplex – {label}",
        os.path.join(out_dir, "mds_circumplex.png"),
        config.seed,
    )

    dist = np.clip(1.0 - emp_sim, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=config.seed,
        n_init=4,
        normalized_stress="auto",
    )
    X_mds = mds.fit_transform(dist)

    X_circle = _theoretical_circle_points(n)
    R, _ = orthogonal_procrustes(X_mds, X_circle)
    X_aligned = X_mds @ R

    group_labels = np.array([value_to_group(val) for val in SCHWARTZ_CIRCUMPLEX_ORDER])
    silhouette = silhouette_score(dist, group_labels, metric="precomputed")

    same_group_mask = []
    different_group_mask = []
    circular_step_flat = []
    neighbor_empirical = []
    opposite_empirical = []
    hierarchical_distance_flat = []
    same_lower_empirical = []
    same_higher_empirical = []
    no_relation_empirical = []
    opposite_higher_empirical = []
    for i in range(n):
        for j in range(i + 1, n):
            same_group = value_to_group(SCHWARTZ_CIRCUMPLEX_ORDER[i]) == value_to_group(SCHWARTZ_CIRCUMPLEX_ORDER[j])
            same_group_mask.append(same_group)
            different_group_mask.append(not same_group)

            step = _circular_step_distance(i, j, n)
            circular_step_flat.append(step)
            if step == 1:
                neighbor_empirical.append(emp_sim[i, j])
            if step == n // 2:
                opposite_empirical.append(emp_sim[i, j])

            hierarchical_distance, relation_bucket = _hierarchical_theory_distance(
                SCHWARTZ_CIRCUMPLEX_ORDER[i],
                SCHWARTZ_CIRCUMPLEX_ORDER[j],
            )
            hierarchical_distance_flat.append(hierarchical_distance)
            if relation_bucket == "same_lower_order":
                same_lower_empirical.append(emp_sim[i, j])
            elif relation_bucket == "same_higher_order":
                same_higher_empirical.append(emp_sim[i, j])
            elif relation_bucket == "opposite_higher_order":
                opposite_higher_empirical.append(emp_sim[i, j])
            else:
                no_relation_empirical.append(emp_sim[i, j])

    same_group_mask = np.array(same_group_mask, dtype=bool)
    different_group_mask = np.array(different_group_mask, dtype=bool)
    circular_step_flat = np.array(circular_step_flat, dtype=float)
    hierarchical_distance_flat = np.array(hierarchical_distance_flat, dtype=float)

    within_group_mean = float(emp_flat[same_group_mask].mean())
    across_group_mean = float(emp_flat[different_group_mask].mean())
    neighbor_mean = float(np.mean(neighbor_empirical))
    opposite_mean = float(np.mean(opposite_empirical))
    circular_distance_spearman, circular_distance_p = spearmanr(emp_flat, -circular_step_flat)
    hierarchical_distance_spearman, hierarchical_distance_p = spearmanr(emp_flat, -hierarchical_distance_flat)

    same_lower_mean = float(np.mean(same_lower_empirical)) if same_lower_empirical else float("nan")
    same_higher_mean = float(np.mean(same_higher_empirical)) if same_higher_empirical else float("nan")
    no_relation_mean = float(np.mean(no_relation_empirical)) if no_relation_empirical else float("nan")
    opposite_higher_mean = float(np.mean(opposite_higher_empirical)) if opposite_higher_empirical else float("nan")
    lower_minus_opposite = same_lower_mean - opposite_higher_mean

    _, _, procrustes_disparity = procrustes(X_circle, X_mds)
    procrustes_rmse = float(np.sqrt(np.mean(np.sum((X_aligned - X_circle) ** 2, axis=1))))

    geometry_metrics = {
        "label": label,
        "spearman_rho": float(rho),
        "spearman_p_value": float(pval),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "num_pairs": int(emp_flat.shape[0]),
        "silhouette_by_higher_order_group": float(silhouette),
        "within_group_mean_cosine": within_group_mean,
        "across_group_mean_cosine": across_group_mean,
        "within_minus_across_cosine": float(within_group_mean - across_group_mean),
        "neighbor_mean_cosine": neighbor_mean,
        "opposite_mean_cosine": opposite_mean,
        "neighbor_minus_opposite_cosine": float(neighbor_mean - opposite_mean),
        "circular_distance_spearman": float(circular_distance_spearman),
        "circular_distance_p_value": float(circular_distance_p),
        "hierarchical_distance_spearman": float(hierarchical_distance_spearman),
        "hierarchical_distance_p_value": float(hierarchical_distance_p),
        "same_lower_order_mean_cosine": same_lower_mean,
        "same_higher_order_mean_cosine": same_higher_mean,
        "no_relation_mean_cosine": no_relation_mean,
        "opposite_higher_order_mean_cosine": opposite_higher_mean,
        "lower_minus_opposite_cosine": lower_minus_opposite,
        "procrustes_disparity": float(procrustes_disparity),
        "procrustes_rmse_after_alignment": procrustes_rmse,
        "mds_stress": float(mds.stress_),
    }
    with open(os.path.join(out_dir, "geometry_metrics.json"), "w") as f:
        json.dump(geometry_metrics, f, indent=2)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def run_geometry(
    config: SparseCAAPipelineConfig,
    vectors: Dict[str, torch.Tensor],
) -> dict:
    """
    Run geometry analysis on raw and mean-centred sparse persona vectors.

    Saves all outputs under:
      <run_dir>/geometry_raw/       — raw sparse vectors
      <run_dir>/geometry_centered/  — mean-centred vectors (visualisation only)

    Returns a dict with Spearman ρ for both variants.
    """
    theo_sim = _theoretical_sim(config.relations_path)

    # ── Raw variant ───────────────────────────────────────────────────────────
    print("\n[Geometry] Raw sparse vectors …")
    raw_dir = config.subdir("geometry_raw")
    raw_report = _run_geometry_variant(vectors, theo_sim, config, raw_dir, "raw_sparse")

    # ── Mean-centred variant ──────────────────────────────────────────────────
    print("[Geometry] Mean-centred vectors …")
    mean_vec = torch.stack(
        [vectors[v].float() for v in SCHWARTZ_CIRCUMPLEX_ORDER]
    ).mean(dim=0)

    centered_vectors = {
        v: vectors[v].float() - mean_vec for v in SCHWARTZ_CIRCUMPLEX_ORDER
    }

    cen_dir = config.subdir("geometry_centered")
    cen_report = _run_geometry_variant(
        centered_vectors, theo_sim, config, cen_dir, "mean_centered"
    )

    # ── Comparison summary ────────────────────────────────────────────────────
    comparison = {
        "raw_spearman_rho": raw_report["spearman_rho"],
        "raw_p_value": raw_report["p_value"],
        "centered_spearman_rho": cen_report["spearman_rho"],
        "centered_p_value": cen_report["p_value"],
        "delta_rho": cen_report["spearman_rho"] - raw_report["spearman_rho"],
    }

    comp_path = os.path.join(config.run_dir, "geometry_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Side-by-side bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Raw sparse", "Mean-centred"],
        [raw_report["spearman_rho"], cen_report["spearman_rho"]],
        color=["#607D8B", "#4CAF50"],
    )
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, [raw_report["spearman_rho"], cen_report["spearman_rho"]]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004 * (1 if val >= 0 else -1),
            f"{val:+.4f}",
            ha="center",
            fontsize=11,
        )
    ax.set_ylabel("Spearman ρ vs. Schwartz theory")
    ax.set_title("Geometry Alignment: Raw vs Mean-Centred")
    ax.set_ylim(
        min(raw_report["spearman_rho"], cen_report["spearman_rho"], 0) - 0.05,
        max(raw_report["spearman_rho"], cen_report["spearman_rho"], 0) + 0.1,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config.run_dir, "rho_comparison.png"), dpi=200)
    plt.close()

    print("\n=== Geometry Summary ===")
    print(f"  Raw sparse vectors   ρ = {comparison['raw_spearman_rho']:+.4f}")
    print(f"  Mean-centred vectors ρ = {comparison['centered_spearman_rho']:+.4f}")
    print(f"  Δρ                     = {comparison['delta_rho']:+.4f}")

    return comparison
