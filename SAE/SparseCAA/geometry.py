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
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

from .config import (
    GROUP_COLORS,
    SCHWARTZ_CIRCUMPLEX_ORDER,
    SparseCAAPipelineConfig,
    value_to_group,
)


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
            markerfacecolor=c, markersize=10,
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


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def _plot_heatmap(mat: np.ndarray, title: str, path: str) -> None:
    labels = [_short(v) for v in SCHWARTZ_CIRCUMPLEX_ORDER]
    fig, ax = plt.subplots(figsize=(14, 12))
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
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_embedding(coords: np.ndarray, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        color = GROUP_COLORS.get(value_to_group(val), "black")
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=100, zorder=3)
        ax.annotate(
            _short(val),
            (coords[i, 0], coords[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.legend(handles=_legend_handles(), loc="best")
    ax.set_title(title)
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

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    R, _ = orthogonal_procrustes(X_mds, X_circle)
    X_aligned = X_mds @ R

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.add_patch(plt.Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--"))

    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        tx, ty = X_circle[i]
        ex, ey = X_aligned[i]
        color = GROUP_COLORS.get(value_to_group(val), "black")

        ax.plot(tx, ty, "x", color="gray", markersize=8)
        ax.plot(ex, ey, "o", color=color, markersize=8)
        ax.plot([tx, ex], [ty, ey], color="gray", alpha=0.3, linestyle=":")
        ax.annotate(
            _short(val),
            (ex, ey),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

    ax.legend(handles=_legend_handles())
    ax.set_title(f"{title}\n(grey ×: Schwartz theory,  coloured ●: empirical)")
    ax.set_aspect("equal")
    lim = max(np.abs(X_aligned).max(), 1.0) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.2)
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
