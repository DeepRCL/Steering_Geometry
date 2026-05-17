"""
Geometry analysis for steering vectors against the Schwartz circumplex.

Self-contained module — identical metrics and visualizations to
the llm-steering-opt pipeline, but with no cross-package imports.
"""

import json
import os
from typing import Dict, List, Tuple

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

from config import (
    SCHWARTZ_CIRCUMPLEX_ORDER,
    HIGHER_ORDER_GROUPS,
    GROUP_COLORS,
    value_to_group,
)


# ─── Helper Functions ────────────────────────────────────────────────────────

def _circular_step_distance(i: int, j: int, n: int) -> int:
    """Shortest step distance between positions *i* and *j* on a circle of size *n*."""
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
    """Scatter plot of a 2-D embedding, coloured by Schwartz higher-order group."""
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


# ─── Main Geometry Analysis ──────────────────────────────────────────────────

def analyze_geometry(
    vectors: Dict[str, torch.Tensor],
    relations_path: str,
    output_dir: str,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Full geometry analysis of steering vectors against the theoretical
    Schwartz circumplex.

    Computes Spearman/Pearson correlations, silhouette scores, within-
    vs across-group cosine similarities, Procrustes alignment to the
    theoretical circle, and generates heatmaps plus dimensionality-
    reduction plots (UMAP, PCA, t-SNE, MDS with circumplex overlay).

    This is identical to SteeringPipeline.analyze_geometry() in the
    llm-steering-opt package, ensuring cross-method comparability.

    Args:
        vectors: Dict mapping value name -> (d_model,) tensor.
        relations_path: Path to schwartz_relations.json.
        output_dir: Directory to save geometry outputs.
        random_seed: Random seed for reproducibility.
        verbose: Whether to print progress.

    Returns:
        Dict of geometry metrics (also saved as geometry_metrics.json).
    """
    def _log(msg):
        if verbose:
            print(msg, flush=True)

    _log("Running geometry analysis...")
    out_dir = os.path.join(output_dir, "geometry")
    os.makedirs(out_dir, exist_ok=True)

    # ── 0. Mean-center then renormalize (consistent with CAA pipeline) ──
    # Step 1: collect raw vectors as float
    raw_vectors: Dict[str, torch.Tensor] = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        raw_vectors[val] = vectors[val].detach().cpu().float()

    # Step 2: center — subtract the mean vector across all values
    mean_vec = torch.stack(
        [raw_vectors[val] for val in SCHWARTZ_CIRCUMPLEX_ORDER]
    ).mean(dim=0)
    centered_vectors: Dict[str, torch.Tensor] = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        centered_vectors[val] = raw_vectors[val] - mean_vec

    # Step 3: renormalize each centered vector to unit norm
    unit_vectors: Dict[str, torch.Tensor] = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        vec = centered_vectors[val]
        norm = vec.norm().clamp_min(1e-12)
        unit_vectors[val] = vec / norm

    num_values = len(SCHWARTZ_CIRCUMPLEX_ORDER)

    # ── 1. Empirical similarity matrix ────────────────────────────
    empirical_sim = np.zeros((num_values, num_values))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            empirical_sim[i, j] = F.cosine_similarity(
                unit_vectors[v1], unit_vectors[v2], dim=0
            ).item()

    # ── 2. Theoretical similarity matrix ──────────────────────────
    with open(relations_path, "r") as f:
        rel_data = json.load(f)
    rel_matrix = rel_data["basic_value_relationship_matrix"]

    theoretical_sim = np.zeros((num_values, num_values))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            if v1 in rel_matrix and v2 in rel_matrix[v1]:
                theoretical_sim[i, j] = rel_matrix[v1][v2]

    # ── 3. Correlation (upper triangle, no diagonal) ─────────────
    triu_indices = np.triu_indices(num_values, k=1)
    emp_flat = empirical_sim[triu_indices]
    theo_flat = theoretical_sim[triu_indices]

    rho, p_val = spearmanr(emp_flat, theo_flat)
    pearson_r, pearson_p = pearsonr(emp_flat, theo_flat)

    with open(os.path.join(out_dir, "spearman_report.json"), "w") as f:
        json.dump({
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "num_pairs": len(emp_flat),
        }, f, indent=2)

    _log(
        f"Spearman correlation between theoretical and empirical "
        f"similarities: rho={rho:.4f}, p={p_val:.4g}"
    )

    # ── 4. Heatmaps ──────────────────────────────────────────────

    # 4a. Original empirical heatmap (fixed range [-1, 1])
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        empirical_sim,
        xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
        yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
        cmap="coolwarm", vmin=-1, vmax=1,
    )
    plt.title("Empirical Cosine Similarities", fontsize=18)
    plt.xticks(fontsize=10, rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap.png"), dpi=300)
    plt.close()

    # 4b. Contrast-enhanced: auto-scale to off-diagonal range
    off_diag_mask = ~np.eye(num_values, dtype=bool)
    off_diag_vals = empirical_sim[off_diag_mask]
    vmin_auto = off_diag_vals.min()
    vmax_auto = off_diag_vals.max()

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        empirical_sim,
        xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
        yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
        cmap="coolwarm",
        vmin=vmin_auto, vmax=vmax_auto,
    )
    plt.title(
        f"Empirical Cosine Similarities (contrast-enhanced)\n"
        f"color range: [{vmin_auto:.3f}, {vmax_auto:.3f}]",
        fontsize=18,
    )
    plt.xticks(fontsize=10, rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap_enhanced.png"), dpi=300)
    plt.close()

    # 4c. Rank-transformed heatmap for maximum contrast
    rank_matrix = np.zeros_like(empirical_sim)
    rank_vals = rankdata(off_diag_vals)  # rank the off-diagonal values
    rank_matrix[off_diag_mask] = rank_vals / rank_vals.max()  # normalize to [0,1]
    np.fill_diagonal(rank_matrix, 1.0)  # diagonal = max similarity

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        rank_matrix,
        xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
        yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
        cmap="coolwarm",
        vmin=0, vmax=1,
    )
    plt.title("Empirical Similarity (rank-transformed, 0=least similar, 1=most)", fontsize=18)
    plt.xticks(fontsize=10, rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap_ranked.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        theoretical_sim,
        xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
        yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
        cmap="coolwarm", vmin=-1, vmax=1,
    )
    plt.title("Theoretical Relationships", fontsize=18)
    plt.xticks(fontsize=10, rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "theoretical_similarity_heatmap.png"), dpi=300)
    plt.close()

    # ── 5. Dimensionality-reduction projections ───────────────────
    X = np.stack([unit_vectors[v].numpy() for v in SCHWARTZ_CIRCUMPLEX_ORDER])

    reducer = umap.UMAP(n_components=2, metric="cosine",
                        n_jobs=1, random_state=random_seed)
    X_umap = reducer.fit_transform(X)
    _plot_embedding_2d(
        os.path.join(out_dir, "umap_2d.png"),
        "UMAP 2D Projection of Steering Vectors", X_umap,
    )

    X_pca = PCA(n_components=2, random_state=random_seed).fit_transform(X)
    _plot_embedding_2d(
        os.path.join(out_dir, "pca_2d.png"),
        "PCA 2D Projection of Steering Vectors", X_pca,
    )

    perplexity = min(5, max(2, num_values - 1))
    X_tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_seed,
    ).fit_transform(X)
    _plot_embedding_2d(
        os.path.join(out_dir, "tsne_2d.png"),
        "t-SNE 2D Projection of Steering Vectors", X_tsne,
    )

    # ── 6. MDS with circumplex overlay ────────────────────────────
    dist_matrix = 1 - empirical_sim
    dist_matrix[dist_matrix < 0] = 0

    mds = MDS(
        n_components=2,
        metric="precomputed",
        init="random",
        random_state=random_seed,
        normalized_stress="auto",
        n_init=4,
    )
    X_mds = mds.fit_transform(dist_matrix)

    angles = np.linspace(0, 2 * np.pi, num_values, endpoint=False)
    X_circle = np.column_stack([np.cos(angles), np.sin(angles)])

    R, _sca = orthogonal_procrustes(X_mds, X_circle)
    X_mds_aligned = X_mds.dot(R)

    # ── 7. Quantitative geometry metrics ──────────────────────────
    group_labels = np.array(
        [value_to_group(val) for val in SCHWARTZ_CIRCUMPLEX_ORDER]
    )
    clipped_dist_matrix = np.maximum(0.0, 1.0 - empirical_sim)
    np.fill_diagonal(clipped_dist_matrix, 0.0)
    sil = silhouette_score(clipped_dist_matrix, group_labels, metric="precomputed")

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
    for i in range(num_values):
        for j in range(i + 1, num_values):
            same = (
                value_to_group(SCHWARTZ_CIRCUMPLEX_ORDER[i])
                == value_to_group(SCHWARTZ_CIRCUMPLEX_ORDER[j])
            )
            same_group_mask.append(same)
            different_group_mask.append(not same)

            step = _circular_step_distance(i, j, num_values)
            circular_step_flat.append(step)
            if step == 1:
                neighbor_empirical.append(empirical_sim[i, j])
            if step == num_values // 2:
                opposite_empirical.append(empirical_sim[i, j])

            hierarchical_distance, relation_bucket = _hierarchical_theory_distance(
                SCHWARTZ_CIRCUMPLEX_ORDER[i],
                SCHWARTZ_CIRCUMPLEX_ORDER[j],
            )
            hierarchical_distance_flat.append(hierarchical_distance)
            if relation_bucket == "same_lower_order":
                same_lower_empirical.append(empirical_sim[i, j])
            elif relation_bucket == "same_higher_order":
                same_higher_empirical.append(empirical_sim[i, j])
            elif relation_bucket == "opposite_higher_order":
                opposite_higher_empirical.append(empirical_sim[i, j])
            else:
                no_relation_empirical.append(empirical_sim[i, j])

    same_group_mask = np.array(same_group_mask, dtype=bool)
    different_group_mask = np.array(different_group_mask, dtype=bool)
    circular_step_flat = np.array(circular_step_flat, dtype=float)
    hierarchical_distance_flat = np.array(hierarchical_distance_flat, dtype=float)

    within_group_mean = float(emp_flat[same_group_mask].mean())
    across_group_mean = float(emp_flat[different_group_mask].mean())
    within_minus_across = within_group_mean - across_group_mean

    neighbor_mean = float(np.mean(neighbor_empirical))
    opposite_mean = float(np.mean(opposite_empirical))
    neighbor_minus_opposite = neighbor_mean - opposite_mean
    circular_distance_spearman, circular_distance_p = spearmanr(
        emp_flat, -circular_step_flat
    )
    hierarchical_distance_spearman, hierarchical_distance_p = spearmanr(
        emp_flat, -hierarchical_distance_flat
    )

    same_lower_mean = float(np.mean(same_lower_empirical)) if same_lower_empirical else float("nan")
    same_higher_mean = float(np.mean(same_higher_empirical)) if same_higher_empirical else float("nan")
    no_relation_mean = float(np.mean(no_relation_empirical)) if no_relation_empirical else float("nan")
    opposite_higher_mean = float(np.mean(opposite_higher_empirical)) if opposite_higher_empirical else float("nan")
    lower_minus_opposite = same_lower_mean - opposite_higher_mean

    _, _, procrustes_disparity = procrustes(X_circle, X_mds)
    procrustes_rmse = float(
        np.sqrt(np.mean(np.sum((X_mds_aligned - X_circle) ** 2, axis=1)))
    )

    geometry_metrics = {
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_val),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "num_pairs": len(emp_flat),
        "silhouette_by_higher_order_group": float(sil),
        "within_group_mean_cosine": within_group_mean,
        "across_group_mean_cosine": across_group_mean,
        "within_minus_across_cosine": within_minus_across,
        "neighbor_mean_cosine": neighbor_mean,
        "opposite_mean_cosine": opposite_mean,
        "neighbor_minus_opposite_cosine": neighbor_minus_opposite,
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

    # ── 8. MDS circumplex overlay plot ────────────────────────────
    plt.figure(figsize=(15, 15))
    circle_patch = plt.Circle((0, 0), 1, color="lightgray",
                              fill=False, linestyle="--")
    plt.gca().add_patch(circle_patch)

    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        tx, ty = X_circle[i]
        plt.plot(tx, ty, "x", color="gray", markersize=9)

        ex, ey = X_mds_aligned[i]
        group = value_to_group(val)
        color = GROUP_COLORS.get(group, "black")

        plt.plot(ex, ey, "o", color=color, markersize=10, markeredgecolor="white", markeredgewidth=1.0)
        plt.plot([tx, ex], [ty, ey], color="gray", alpha=0.3, linestyle=":")

        label = val.split(":")[-1].strip()
        plt.annotate(
            label,
            (ex, ey),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=13,
            fontweight="semibold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.8),
        )

    plt.title("2D MDS Aligned to Theoretical Circumplex", fontsize=18)
    plt.axis("equal")
    scale = np.max(np.abs(X_mds_aligned))
    lim = max(1.2, scale * 1.2)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.grid(alpha=0.2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mds_circumplex.png"), dpi=300)
    plt.close()

    # ── 9. Theory vs empirical scatter ────────────────────────────
    plt.figure(figsize=(8, 5))
    jitter = np.random.default_rng(random_seed).normal(
        0.0, 0.03, size=len(theo_flat)
    )
    plt.scatter(theo_flat + jitter, emp_flat, alpha=0.7, s=40)
    plt.xticks([-1, 0, 1])
    plt.xlabel("Theoretical Relationship")
    plt.ylabel("Empirical Cosine Similarity")
    plt.title("Empirical Similarity vs Theory")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "theory_vs_empirical_scatter.png"), dpi=300)
    plt.close()

    # ── 10. Difference heatmap ────────────────────────────────────
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        empirical_sim - theoretical_sim,
        xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
        yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
        cmap="coolwarm",
        center=0.0,
    )
    plt.title("Empirical Minus Theoretical Similarity")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "empirical_minus_theoretical_heatmap.png"), dpi=300
    )
    plt.close()

    _log("Geometry analysis complete!")
    return geometry_metrics
