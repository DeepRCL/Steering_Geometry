import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from sklearn.manifold import MDS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAA.Geometry.config import GROUP_COLORS, HIGHER_ORDER_GROUPS, SCHWARTZ_CIRCUMPLEX_ORDER, value_to_group


DEFAULT_RELATIONS_PATH = PROJECT_ROOT / "CAA" / "value_data" / "schwartz_relations-new.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "CAA" / "Geometry" / "theoretical_source"
PLOT_LABEL_FONTSIZE = 13
PLOT_TITLE_FONTSIZE = 18
BOUNDARY_GROUPS = {
    "Hedonism": ("Openness to Change", "Self-Enhancement"),
    "Face": ("Self-Enhancement", "Conservation"),
    "Humility": ("Conservation", "Self-Transcendence"),
}


def theoretical_circle_points(num_values: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num_values, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def short_value_label(value: str) -> str:
    return value.split(":")[-1].strip()


def groups_for_display(value: str) -> tuple[str, ...]:
    if value in BOUNDARY_GROUPS:
        return BOUNDARY_GROUPS[value]
    return (value_to_group(value),)


def add_group_legend(ax) -> None:
    handles = []
    labels = []
    for group_name in HIGHER_ORDER_GROUPS:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=GROUP_COLORS[group_name],
                markeredgecolor="white",
                markersize=11,
                linewidth=0,
            )
        )
        labels.append(group_name)

    legend = ax.legend(
        handles,
        labels,
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


def draw_value_marker(ax, x: float, y: float, value: str, radius: float = 0.055) -> None:
    if value in BOUNDARY_GROUPS:
        left_group, right_group = BOUNDARY_GROUPS[value]
        ax.add_patch(Wedge((x, y), radius, 90, 270, facecolor=GROUP_COLORS[left_group], edgecolor="none", zorder=3))
        ax.add_patch(Wedge((x, y), radius, -90, 90, facecolor=GROUP_COLORS[right_group], edgecolor="none", zorder=3))
        ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="white", linewidth=1.2, zorder=4))
        ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="black", linewidth=0.3, alpha=0.35, zorder=4))
        return

    group = value_to_group(value)
    ax.add_patch(
        Circle(
            (x, y),
            radius,
            facecolor=GROUP_COLORS.get(group, "black"),
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
    )


def build_theoretical_similarity_matrix(relations_path: Path) -> np.ndarray:
    with relations_path.open() as f:
        rel_data = json.load(f)

    rel_matrix = rel_data["basic_value_relationship_matrix"]
    num_values = len(SCHWARTZ_CIRCUMPLEX_ORDER)
    theoretical_sim = np.zeros((num_values, num_values), dtype=float)

    for i, value_i in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, value_j in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            if value_i in rel_matrix and value_j in rel_matrix[value_i]:
                theoretical_sim[i, j] = rel_matrix[value_i][value_j]

    return theoretical_sim


def save_similarity_heatmap(theoretical_sim: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(14, 12))
    image = plt.imshow(theoretical_sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.title("Theoretical Relationships", fontsize=PLOT_TITLE_FONTSIZE)
    ticks = np.arange(len(SCHWARTZ_CIRCUMPLEX_ORDER))
    plt.xticks(ticks, SCHWARTZ_CIRCUMPLEX_ORDER, fontsize=10, rotation=45, ha="right")
    plt.yticks(ticks, SCHWARTZ_CIRCUMPLEX_ORDER, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "theoretical_similarity_heatmap.png", dpi=300)
    plt.close()


def plot_canonical_circumplex(x_circle: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--")
    ax.add_patch(circle)

    for i, value in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        x, y = x_circle[i]
        group = value_to_group(value)
        color = GROUP_COLORS.get(group, "black")
        label = value.split(":")[-1].strip()

        draw_value_marker(ax, x, y, value)
        ax.annotate(
            label,
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=PLOT_LABEL_FONTSIZE,
            fontweight="semibold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.8),
        )

    add_group_legend(ax)
    plt.title("Canonical Schwartz Circumplex", fontsize=PLOT_TITLE_FONTSIZE)
    plt.axis("equal")
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    plt.grid(alpha=0.2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "canonical_circumplex.png", dpi=300)
    plt.close()


def plot_theoretical_mds(theoretical_sim: np.ndarray, x_circle: np.ndarray, seed: int, output_dir: Path) -> dict:
    dist_matrix = np.maximum(0.0, 1.0 - theoretical_sim)
    np.fill_diagonal(dist_matrix, 0.0)

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        normalized_stress="auto",
        n_init=4,
    )
    x_mds = mds.fit_transform(dist_matrix)
    rotation, _ = orthogonal_procrustes(x_mds, x_circle)
    x_mds_aligned = x_mds.dot(rotation)

    _, _, procrustes_disparity = procrustes(x_circle, x_mds)
    procrustes_rmse = float(np.sqrt(np.mean(np.sum((x_mds_aligned - x_circle) ** 2, axis=1))))

    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--")
    ax.add_patch(circle)

    for i, value in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        tx, ty = x_circle[i]
        ex, ey = x_mds_aligned[i]
        group = value_to_group(value)
        color = GROUP_COLORS.get(group, "black")
        label = value.split(":")[-1].strip()

        ax.plot(tx, ty, "x", color="gray", markersize=9)
        draw_value_marker(ax, ex, ey, value)
        ax.plot([tx, ex], [ty, ey], color="gray", alpha=0.3, linestyle=":")
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

    add_group_legend(ax)
    plt.title("Theoretical MDS Aligned to Canonical Circumplex", fontsize=PLOT_TITLE_FONTSIZE)
    plt.axis("equal")
    scale = np.max(np.abs(x_mds_aligned))
    lim = max(1.2, scale * 1.2)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.grid(alpha=0.2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "theoretical_mds_circumplex.png", dpi=300)
    plt.close()

    return {
        "mds_stress": float(mds.stress_),
        "procrustes_disparity": float(procrustes_disparity),
        "procrustes_rmse_after_alignment": procrustes_rmse,
        "canonical_coordinates": {
            value: {"x": float(x_circle[i, 0]), "y": float(x_circle[i, 1])}
            for i, value in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER)
        },
        "theoretical_mds_aligned_coordinates": {
            value: {"x": float(x_mds_aligned[i, 0]), "y": float(x_mds_aligned[i, 1])}
            for i, value in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a theory-only Schwartz circumplex figure with the same anchor used in model MDS plots."
    )
    parser.add_argument("--relations_path", type=Path, default=DEFAULT_RELATIONS_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    theoretical_sim = build_theoretical_similarity_matrix(args.relations_path)
    x_circle = theoretical_circle_points(len(SCHWARTZ_CIRCUMPLEX_ORDER))

    save_similarity_heatmap(theoretical_sim, output_dir)
    plot_canonical_circumplex(x_circle, output_dir)
    metrics = plot_theoretical_mds(theoretical_sim, x_circle, args.seed, output_dir)

    summary = {
        "relations_path": str(args.relations_path),
        "anchor": "canonical_circle_from_schwartz_order",
        "order": SCHWARTZ_CIRCUMPLEX_ORDER,
        **metrics,
    }
    with (output_dir / "theoretical_circumplex_source.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Wrote {output_dir / 'theoretical_similarity_heatmap.png'}")
    print(f"Wrote {output_dir / 'canonical_circumplex.png'}")
    print(f"Wrote {output_dir / 'theoretical_mds_circumplex.png'}")
    print(f"Wrote {output_dir / 'theoretical_circumplex_source.json'}")


if __name__ == "__main__":
    main()
