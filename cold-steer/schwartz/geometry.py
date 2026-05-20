"""Schwartz circumplex geometry analysis for cold_fd steering vectors.

This is a verbatim port of
``odesteer/scripts/schwartz/geometry.py`` (which itself mirrors the
``llm-steering-opt`` pipeline) so cold_fd outputs use the *exact* same
metrics, plots, and filenames as the other methods. The only edit is the
import of ``SCHWARTZ_CIRCUMPLEX_ORDER`` / ``HIGHER_ORDER_GROUPS`` /
``GROUP_COLORS`` / ``value_to_group`` from this package's ``config``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    SCHWARTZ_CIRCUMPLEX_ORDER,
    HIGHER_ORDER_GROUPS,
    GROUP_COLORS,
    value_to_group,
)


def _circular_step_distance(i: int, j: int, n: int) -> int:
    return min(abs(i - j), n - abs(i - j))


def _lower_order_family(value: str) -> str:
    return value.split(":")[0].strip() if ":" in value else value


def _higher_order_groups_for_value(value: str) -> set:
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


def _hierarchical_theory_distance(value_a: str, value_b: str) -> Tuple[int, str]:
    if _lower_order_family(value_a) == _lower_order_family(value_b):
        return 1, "same_lower_order"
    groups_a = _higher_order_groups_for_value(value_a)
    groups_b = _higher_order_groups_for_value(value_b)
    if groups_a & groups_b:
        return 2, "same_higher_order"
    if any(_groups_are_opposite(ga, gb) for ga in groups_a for gb in groups_b):
        return 10, "opposite_higher_order"
    return 5, "no_relation"


def _plot_embedding_2d(out_path: str, title: str, coords: np.ndarray):
    plt.figure(figsize=(14, 11))
    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        group = value_to_group(val)
        color = GROUP_COLORS.get(group, "black")
        plt.scatter(coords[i, 0], coords[i, 1], c=color, s=150, edgecolors="white", linewidths=1.2)
        plt.annotate(
            val.split(":")[-1].strip(),
            (coords[i, 0], coords[i, 1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=13,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
        )
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=12, label=g)
        for g, c in GROUP_COLORS.items()
    ]
    plt.legend(handles=legend_els, loc="best", fontsize=13)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def analyze_geometry(
    vectors: Dict[str, torch.Tensor],
    relations_path: str,
    output_dir: str,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    """Run the full Schwartz geometry analysis on a dict of vectors.

    Args:
        vectors: mapping value name → 1-D torch tensor.
        relations_path: path to ``schwartz_relations.json``.
        output_dir: parent dir; geometry artefacts land in ``{output_dir}/geometry``.
        random_seed: seed passed to UMAP / PCA / t-SNE / MDS / jitter.
        verbose: print progress.
    Returns:
        Dict of geometry metrics (also saved to ``geometry_metrics.json``).
    """
    def _log(msg):
        if verbose:
            print(msg, flush=True)

    _log("Running geometry analysis...")
    out_dir = os.path.join(output_dir, "geometry")
    os.makedirs(out_dir, exist_ok=True)

    raw_vectors = {val: vectors[val].detach().cpu().float() for val in SCHWARTZ_CIRCUMPLEX_ORDER}
    mean_vec = torch.stack([raw_vectors[v] for v in SCHWARTZ_CIRCUMPLEX_ORDER]).mean(dim=0)
    centered = {v: raw_vectors[v] - mean_vec for v in SCHWARTZ_CIRCUMPLEX_ORDER}
    unit_vectors = {
        v: (vec / vec.norm().clamp_min(1e-12)) for v, vec in centered.items()
    }

    num_values = len(SCHWARTZ_CIRCUMPLEX_ORDER)

    empirical_sim = np.zeros((num_values, num_values))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            empirical_sim[i, j] = F.cosine_similarity(
                unit_vectors[v1], unit_vectors[v2], dim=0
            ).item()

    with open(relations_path, "r") as f:
        rel_data = json.load(f)
    rel_matrix = rel_data["basic_value_relationship_matrix"]
    theoretical_sim = np.zeros((num_values, num_values))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            if v1 in rel_matrix and v2 in rel_matrix[v1]:
                theoretical_sim[i, j] = rel_matrix[v1][v2]

    triu = np.triu_indices(num_values, k=1)
    emp_flat = empirical_sim[triu]
    theo_flat = theoretical_sim[triu]

    rho, p_val = spearmanr(emp_flat, theo_flat)
    pearson_r, pearson_p = pearsonr(emp_flat, theo_flat)
    with open(os.path.join(out_dir, "spearman_report.json"), "w") as f:
        json.dump({
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "num_pairs": len(emp_flat),
        }, f, indent=2)
    _log(f"Spearman: rho={rho:.4f}, p={p_val:.4g}")

    plt.figure(figsize=(14, 12))
    sns.heatmap(empirical_sim,
                xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
                cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Empirical Cosine Similarities", fontsize=18)
    plt.xticks(fontsize=10, rotation=45, ha="right"); plt.yticks(fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap.png"), dpi=300); plt.close()

    off_diag = ~np.eye(num_values, dtype=bool)
    off_vals = empirical_sim[off_diag]
    vmin_a, vmax_a = off_vals.min(), off_vals.max()
    plt.figure(figsize=(14, 12))
    sns.heatmap(empirical_sim,
                xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
                cmap="coolwarm", vmin=vmin_a, vmax=vmax_a)
    plt.title(f"Empirical Cosine Similarities (contrast-enhanced)\ncolor range: [{vmin_a:.3f}, {vmax_a:.3f}]", fontsize=18)
    plt.xticks(fontsize=10, rotation=45, ha="right"); plt.yticks(fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap_enhanced.png"), dpi=300); plt.close()

    rank_matrix = np.zeros_like(empirical_sim)
    rank_vals = rankdata(off_vals)
    rank_matrix[off_diag] = rank_vals / rank_vals.max()
    np.fill_diagonal(rank_matrix, 1.0)
    plt.figure(figsize=(14, 12))
    sns.heatmap(rank_matrix,
                xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
                cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Empirical Similarity (rank-transformed, 0=least similar, 1=most)", fontsize=18)
    plt.xticks(fontsize=10, rotation=45, ha="right"); plt.yticks(fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap_ranked.png"), dpi=300); plt.close()

    plt.figure(figsize=(14, 12))
    sns.heatmap(theoretical_sim,
                xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
                cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Theoretical Relationships", fontsize=18)
    plt.xticks(fontsize=10, rotation=45, ha="right"); plt.yticks(fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "theoretical_similarity_heatmap.png"), dpi=300); plt.close()

    X = np.stack([unit_vectors[v].numpy() for v in SCHWARTZ_CIRCUMPLEX_ORDER])
    reducer = umap.UMAP(n_components=2, metric="cosine", n_jobs=1, random_state=random_seed)
    X_umap = reducer.fit_transform(X)
    _plot_embedding_2d(os.path.join(out_dir, "umap_2d.png"),
                       "UMAP 2D Projection of Steering Vectors", X_umap)
    X_pca = PCA(n_components=2, random_state=random_seed).fit_transform(X)
    _plot_embedding_2d(os.path.join(out_dir, "pca_2d.png"),
                       "PCA 2D Projection of Steering Vectors", X_pca)
    perplexity = min(5, max(2, num_values - 1))
    X_tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  learning_rate="auto", random_state=random_seed).fit_transform(X)
    _plot_embedding_2d(os.path.join(out_dir, "tsne_2d.png"),
                       "t-SNE 2D Projection of Steering Vectors", X_tsne)

    dist_matrix = 1 - empirical_sim
    dist_matrix[dist_matrix < 0] = 0

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_seed,
        normalized_stress="auto",
        n_init=4,
    )
    X_mds = mds.fit_transform(dist_matrix)
    angles = np.linspace(0, 2 * np.pi, num_values, endpoint=False)
    X_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    R, _ = orthogonal_procrustes(X_mds, X_circle)
    X_mds_aligned = X_mds.dot(R)

    group_labels = np.array([value_to_group(v) for v in SCHWARTZ_CIRCUMPLEX_ORDER])
    clipped = np.maximum(0.0, 1.0 - empirical_sim)
    np.fill_diagonal(clipped, 0.0)
    sil = silhouette_score(clipped, group_labels, metric="precomputed")

    same_grp, diff_grp = [], []
    circ_steps, hier_dists = [], []
    neighbor, opposite = [], []
    same_lower, same_higher, no_rel, opp_higher = [], [], [], []
    for i in range(num_values):
        for j in range(i + 1, num_values):
            same = value_to_group(SCHWARTZ_CIRCUMPLEX_ORDER[i]) == value_to_group(SCHWARTZ_CIRCUMPLEX_ORDER[j])
            same_grp.append(same); diff_grp.append(not same)
            step = _circular_step_distance(i, j, num_values)
            circ_steps.append(step)
            if step == 1: neighbor.append(empirical_sim[i, j])
            if step == num_values // 2: opposite.append(empirical_sim[i, j])
            hd, bucket = _hierarchical_theory_distance(
                SCHWARTZ_CIRCUMPLEX_ORDER[i], SCHWARTZ_CIRCUMPLEX_ORDER[j])
            hier_dists.append(hd)
            if bucket == "same_lower_order": same_lower.append(empirical_sim[i, j])
            elif bucket == "same_higher_order": same_higher.append(empirical_sim[i, j])
            elif bucket == "opposite_higher_order": opp_higher.append(empirical_sim[i, j])
            else: no_rel.append(empirical_sim[i, j])

    same_grp_m = np.array(same_grp, dtype=bool)
    diff_grp_m = np.array(diff_grp, dtype=bool)
    circ_steps = np.array(circ_steps, dtype=float)
    hier_dists = np.array(hier_dists, dtype=float)

    within = float(emp_flat[same_grp_m].mean())
    across = float(emp_flat[diff_grp_m].mean())
    neighbor_mean = float(np.mean(neighbor))
    opposite_mean = float(np.mean(opposite))
    circ_rho, circ_p = spearmanr(emp_flat, -circ_steps)
    hier_rho, hier_p = spearmanr(emp_flat, -hier_dists)

    def _safe_mean(xs):
        return float(np.mean(xs)) if xs else float("nan")
    sl_mean = _safe_mean(same_lower)
    sh_mean = _safe_mean(same_higher)
    nr_mean = _safe_mean(no_rel)
    oh_mean = _safe_mean(opp_higher)

    _, _, proc_disp = procrustes(X_circle, X_mds)
    proc_rmse = float(np.sqrt(np.mean(np.sum((X_mds_aligned - X_circle) ** 2, axis=1))))

    geometry_metrics = {
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_val),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "num_pairs": len(emp_flat),
        "silhouette_by_higher_order_group": float(sil),
        "within_group_mean_cosine": within,
        "across_group_mean_cosine": across,
        "within_minus_across_cosine": within - across,
        "neighbor_mean_cosine": neighbor_mean,
        "opposite_mean_cosine": opposite_mean,
        "neighbor_minus_opposite_cosine": neighbor_mean - opposite_mean,
        "circular_distance_spearman": float(circ_rho),
        "circular_distance_p_value": float(circ_p),
        "hierarchical_distance_spearman": float(hier_rho),
        "hierarchical_distance_p_value": float(hier_p),
        "same_lower_order_mean_cosine": sl_mean,
        "same_higher_order_mean_cosine": sh_mean,
        "no_relation_mean_cosine": nr_mean,
        "opposite_higher_order_mean_cosine": oh_mean,
        "lower_minus_opposite_cosine": sl_mean - oh_mean,
        "procrustes_disparity": float(proc_disp),
        "procrustes_rmse_after_alignment": proc_rmse,
        "mds_stress": float(mds.stress_),
    }
    with open(os.path.join(out_dir, "geometry_metrics.json"), "w") as f:
        json.dump(geometry_metrics, f, indent=2)

    plt.figure(figsize=(15, 15))
    plt.gca().add_patch(plt.Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--"))
    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        tx, ty = X_circle[i]
        plt.plot(tx, ty, "x", color="gray", markersize=9)
        ex, ey = X_mds_aligned[i]
        color = GROUP_COLORS.get(value_to_group(val), "black")
        plt.plot(ex, ey, "o", color=color, markersize=10, markeredgecolor="white", markeredgewidth=1.0)
        plt.plot([tx, ex], [ty, ey], color="gray", alpha=0.3, linestyle=":")
        plt.annotate(
            val.split(":")[-1].strip(),
            (ex, ey),
            xytext=(8, 8), textcoords="offset points",
            fontsize=13, fontweight="semibold", color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.8),
        )
    plt.title("2D MDS Aligned to Theoretical Circumplex", fontsize=18)
    plt.axis("equal")
    lim = max(1.2, float(np.max(np.abs(X_mds_aligned))) * 1.2)
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.grid(alpha=0.2); plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "mds_circumplex.png"), dpi=300); plt.close()

    plt.figure(figsize=(8, 5))
    jitter = np.random.default_rng(random_seed).normal(0.0, 0.03, size=len(theo_flat))
    plt.scatter(theo_flat + jitter, emp_flat, alpha=0.7, s=40)
    plt.xticks([-1, 0, 1])
    plt.xlabel("Theoretical Relationship"); plt.ylabel("Empirical Cosine Similarity")
    plt.title("Empirical Similarity vs Theory")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "theory_vs_empirical_scatter.png"), dpi=300); plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(empirical_sim - theoretical_sim,
                xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
                cmap="coolwarm", center=0.0)
    plt.title("Empirical Minus Theoretical Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "empirical_minus_theoretical_heatmap.png"), dpi=300); plt.close()

    _log("Geometry analysis complete.")
    return geometry_metrics
