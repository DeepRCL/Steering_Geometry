import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from typing import Dict

from .config import PipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, HIGHER_ORDER_GROUPS, value_to_group, GROUP_COLORS, safe_name

PLOT_LABEL_FONTSIZE = 13
PLOT_TITLE_FONTSIZE = 18
PLOT_LEGEND_FONTSIZE = 13
PLOT_MARKER_SIZE = 150
BOUNDARY_GROUPS = {
    "Hedonism": ("Openness to Change", "Self-Enhancement"),
    "Face": ("Self-Enhancement", "Conservation"),
    "Humility": ("Conservation", "Self-Transcendence"),
}


def _short_value_label(value: str) -> str:
    return value.split(":")[-1].strip()


def _group_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=GROUP_COLORS[group_name],
            markeredgecolor="white",
            markersize=11,
            linewidth=0,
            label=group_name,
        )
        for group_name in HIGHER_ORDER_GROUPS
    ]


def _add_group_legend(ax) -> None:
    legend = ax.legend(
        handles=_group_legend_handles(),
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="lightgray",
        fontsize=12,
        borderpad=0.5,
        labelspacing=0.45,
        handletextpad=0.6,
    )
    ax.add_artist(legend)


def _draw_value_marker(ax, x: float, y: float, value: str, radius: float = 0.055) -> None:
    if value in BOUNDARY_GROUPS:
        left_group, right_group = BOUNDARY_GROUPS[value]
        ax.add_patch(Wedge((x, y), radius, 90, 270, facecolor=GROUP_COLORS[left_group], edgecolor="none", zorder=3))
        ax.add_patch(Wedge((x, y), radius, -90, 90, facecolor=GROUP_COLORS[right_group], edgecolor="none", zorder=3))
        ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="white", linewidth=1.2, zorder=4))
        ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="black", linewidth=0.3, alpha=0.35, zorder=4))
        return

    ax.add_patch(
        Circle(
            (x, y),
            radius,
            facecolor=GROUP_COLORS.get(value_to_group(value), "black"),
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
    )


def _plot_embedding_2d(out_path: str, title: str, coords: np.ndarray):
    plt.figure(figsize=(14, 11))
    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        group = value_to_group(val)
        color = GROUP_COLORS.get(group, "black")
        plt.scatter(coords[i, 0], coords[i, 1], c=color, s=PLOT_MARKER_SIZE, edgecolors="white", linewidths=1.2)
        plt.annotate(
            _short_value_label(val),
            (coords[i, 0], coords[i, 1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=PLOT_LABEL_FONTSIZE,
            fontweight="semibold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
        )

    plt.legend(handles=_group_legend_handles(), loc="best", fontsize=PLOT_LEGEND_FONTSIZE)
    plt.title(title, fontsize=PLOT_TITLE_FONTSIZE)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


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


def analyze_geometry(config: PipelineConfig, vectors: Dict[str, torch.Tensor]):
    print("Running geometry analysis...")
    out_dir = config.subdir("geometry")
    
    # Ensure vectors are normalized
    unit_vectors = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        vec = vectors[val].detach().cpu().float()
        norm = vec.norm()
        if norm > 0:
            unit_vectors[val] = vec / norm
        else:
            unit_vectors[val] = vec
            
    num_values = len(SCHWARTZ_CIRCUMPLEX_ORDER)
    
    # 1. Empirical Similarity Matrix
    empirical_sim = np.zeros((num_values, num_values))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            cos_sim = F.cosine_similarity(unit_vectors[v1], unit_vectors[v2], dim=0).item()
            empirical_sim[i, j] = cos_sim
            
    # 2. Theoretical Matrix
    with open(config.relations_path, 'r') as f:
        rel_data = json.load(f)
    rel_matrix = rel_data['basic_value_relationship_matrix']
    
    theoretical_sim = np.zeros((num_values, num_values))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            if v1 in rel_matrix and v2 in rel_matrix[v1]:
                theoretical_sim[i, j] = rel_matrix[v1][v2]
                
    # 3. Correlation
    # Get upper triangles without diagonal
    triu_indices = np.triu_indices(num_values, k=1)
    emp_flat = empirical_sim[triu_indices]
    theo_flat = theoretical_sim[triu_indices]
    
    rho, p_val = spearmanr(emp_flat, theo_flat)
    pearson_r, pearson_p = pearsonr(emp_flat, theo_flat)
    
    with open(os.path.join(out_dir, "spearman_report.json"), "w") as f:
        json.dump({
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "num_pairs": len(emp_flat)
        }, f, indent=2)
        
    print(f"Spearman correlation between theoretical and empirical similarities: rho={rho:.4f}, p={p_val:.4g}")
    
    # 4. Visualizations
    
    # Heatmaps
    plt.figure(figsize=(14, 12))
    sns.heatmap(empirical_sim, xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Empirical Cosine Similarities', fontsize=PLOT_TITLE_FONTSIZE)
    plt.xticks(fontsize=10, rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(theoretical_sim, xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Theoretical Relationships', fontsize=PLOT_TITLE_FONTSIZE)
    plt.xticks(fontsize=10, rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "theoretical_similarity_heatmap.png"), dpi=300)
    plt.close()
    
    X = np.stack([unit_vectors[v].numpy() for v in SCHWARTZ_CIRCUMPLEX_ORDER])

    # UMAP 2D
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=config.seed)
    X_umap = reducer.fit_transform(X)
    _plot_embedding_2d(os.path.join(out_dir, "umap_2d.png"), "UMAP 2D Projection of Steering Vectors", X_umap)

    # PCA 2D
    X_pca = PCA(n_components=2, random_state=config.seed).fit_transform(X)
    _plot_embedding_2d(os.path.join(out_dir, "pca_2d.png"), "PCA 2D Projection of Steering Vectors", X_pca)

    # t-SNE 2D
    perplexity = min(5, max(2, len(SCHWARTZ_CIRCUMPLEX_ORDER) - 1))
    X_tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=config.seed,
    ).fit_transform(X)
    _plot_embedding_2d(os.path.join(out_dir, "tsne_2d.png"), "t-SNE 2D Projection of Steering Vectors", X_tsne)
    
    # MDS with Circumplex Overlay
    # Distance matrix = 1 - cosine similarity
    dist_matrix = 1 - empirical_sim
    # Replace negative distances with 0 just in case
    dist_matrix[dist_matrix < 0] = 0
    
    mds = MDS(
        n_components=2,
        dissimilarity='precomputed',
        random_state=config.seed,
        normalized_stress='auto',
        n_init=4,
    )
    X_mds = mds.fit_transform(dist_matrix)
    
    # Theoretical points on a circle based on order
    X_circle = _theoretical_circle_points(num_values)
    
    R, sca = orthogonal_procrustes(X_mds, X_circle)
    X_mds_aligned = X_mds.dot(R)

    # Additional quantitative geometry metrics
    group_labels = np.array([value_to_group(val) for val in SCHWARTZ_CIRCUMPLEX_ORDER])
    clipped_dist_matrix = np.maximum(0.0, 1.0 - empirical_sim)
    np.fill_diagonal(clipped_dist_matrix, 0.0)
    silhouette = silhouette_score(clipped_dist_matrix, group_labels, metric="precomputed")

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
            same_group = value_to_group(SCHWARTZ_CIRCUMPLEX_ORDER[i]) == value_to_group(SCHWARTZ_CIRCUMPLEX_ORDER[j])
            same_group_mask.append(same_group)
            different_group_mask.append(not same_group)

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
    circular_distance_spearman, circular_distance_p = spearmanr(emp_flat, -circular_step_flat)
    hierarchical_distance_spearman, hierarchical_distance_p = spearmanr(emp_flat, -hierarchical_distance_flat)

    same_lower_mean = float(np.mean(same_lower_empirical)) if same_lower_empirical else float("nan")
    same_higher_mean = float(np.mean(same_higher_empirical)) if same_higher_empirical else float("nan")
    no_relation_mean = float(np.mean(no_relation_empirical)) if no_relation_empirical else float("nan")
    opposite_higher_mean = float(np.mean(opposite_higher_empirical)) if opposite_higher_empirical else float("nan")
    lower_minus_opposite = same_lower_mean - opposite_higher_mean

    _, _, procrustes_disparity = procrustes(X_circle, X_mds)
    procrustes_rmse = float(np.sqrt(np.mean(np.sum((X_mds_aligned - X_circle) ** 2, axis=1))))

    geometry_metrics = {
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_val),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "num_pairs": len(emp_flat),
        "silhouette_by_higher_order_group": float(silhouette),
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
    
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    ax.add_patch(Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--"))

    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        # Theoretical pos
        tx, ty = X_circle[i]
        ax.plot(tx, ty, "x", color="gray", markersize=9)

        # Empirical pos
        ex, ey = X_mds_aligned[i]
        group = value_to_group(val)
        color = GROUP_COLORS.get(group, "black")

        _draw_value_marker(ax, ex, ey, val)

        # Draw line connecting theoretical to empirical
        ax.plot([tx, ex], [ty, ey], color="gray", alpha=0.3, linestyle=":")

        label = _short_value_label(val)
        ax.annotate(
            label,
            (ex, ey),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=PLOT_LABEL_FONTSIZE,
            fontweight="semibold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.8),
        )

    _add_group_legend(ax)
    plt.title("2D MDS Aligned to Theoretical Circumplex", fontsize=PLOT_TITLE_FONTSIZE)
    plt.axis("equal")
    # Set limits clearly showing unit circle
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

    # Scatter plot comparing empirical similarities to theory labels directly.
    plt.figure(figsize=(8, 5))
    jitter = np.random.default_rng(config.seed).normal(0.0, 0.03, size=len(theo_flat))
    plt.scatter(theo_flat + jitter, emp_flat, alpha=0.7, s=40)
    plt.xticks([-1, 0, 1])
    plt.xlabel("Theoretical Relationship")
    plt.ylabel("Empirical Cosine Similarity")
    plt.title("Empirical Similarity vs Theory")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "theory_vs_empirical_scatter.png"), dpi=300)
    plt.close()

    # Pairwise difference heatmap to see where empirical structure overshoots or undershoots theory.
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
    plt.savefig(os.path.join(out_dir, "empirical_minus_theoretical_heatmap.png"), dpi=300)
    plt.close()
    
    print("Geometry analysis complete!")
