"""
SAE-based analysis of Schwartz value steering vectors.

Pipeline
────────
1. Load the pre-extracted MLP CAA vectors (one per Schwartz value).
2. Project each vector through the SAE encoder → sparse feature activations
   of shape (d_sae,) = (16 384,).
3. Identify "common" features – features that are highly active across *every*
   value vector.  These capture generic language / topic signals that are not
   value-specific, the equivalent of "polysemantic" directions in the raw space.
4. Purify each vector by zeroing out the common features and decoding back to
   the original activation space.
5. Run geometry analysis (Spearman rho vs. Schwartz theory) on both the raw
   and purified vectors and compare.
6. Disjointness test: do opposing higher-order groups (e.g. Conservation vs.
   Openness-to-Change) activate disjoint feature sets?  Measured via pairwise
   Jaccard similarity on each group's top-K feature union.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

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
from sklearn.manifold import MDS

from .config import (
    GROUP_COLORS,
    HIGHER_ORDER_GROUPS,
    OPPOSING_PAIRS,
    SCHWARTZ_CIRCUMPLEX_ORDER,
    SAEConfig,
    safe_name,
    value_to_group,
)
from .sae_model import SparseAutoencoder, load_sae


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _short_label(val: str) -> str:
    """'Conformity: rules' → 'rules'"""
    return val.split(":")[-1].strip()


def _legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=g)
        for g, c in GROUP_COLORS.items()
    ]


def _plot_embedding(coords: np.ndarray, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        color = GROUP_COLORS.get(value_to_group(val), "black")
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=100, zorder=3)
        ax.annotate(
            _short_label(val),
            (coords[i, 0], coords[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.legend(handles=_legend_handles(), loc="best")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_heatmap(matrix: np.ndarray, title: str, out_path: str) -> None:
    labels = [_short_label(v) for v in SCHWARTZ_CIRCUMPLEX_ORDER]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        matrix,
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
    plt.savefig(out_path, dpi=200)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────
def _empirical_sim_matrix(
    unit_vecs: Dict[str, torch.Tensor],
) -> np.ndarray:
    n = len(SCHWARTZ_CIRCUMPLEX_ORDER)
    mat = np.zeros((n, n))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            mat[i, j] = F.cosine_similarity(unit_vecs[v1], unit_vecs[v2], dim=0).item()
    return mat


def _theoretical_matrix(relations_path: str) -> np.ndarray:
    with open(relations_path) as f:
        rel = json.load(f)["basic_value_relationship_matrix"]
    n = len(SCHWARTZ_CIRCUMPLEX_ORDER)
    mat = np.zeros((n, n))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            if v1 in rel and v2 in rel[v1]:
                mat[i, j] = rel[v1][v2]
    return mat


def _unit(vectors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = {}
    for val, vec in vectors.items():
        v = vec.detach().cpu().float()
        n = v.norm()
        result[val] = v / n if n > 0 else v
    return result


# ──────────────────────────────────────────────────────────────────────────────
# SAEAnalyzer
# ──────────────────────────────────────────────────────────────────────────────
class SAEAnalyzer:
    """Orchestrates the full SAE analysis pipeline."""

    def __init__(self, config: SAEConfig):
        self.config = config
        self.sae: Optional[SparseAutoencoder] = None
        self.mlp_vectors: Optional[Dict[str, torch.Tensor]] = None
        # (n_values, d_sae) sparse feature activations for each value's CAA vector
        self.feature_matrix: Optional[torch.Tensor] = None
        self.common_features: Optional[List[int]] = None
        self.purified_vectors: Optional[Dict[str, torch.Tensor]] = None

    # ── Loading ───────────────────────────────────────────────────────────────
    def load_sae(self) -> None:
        self.sae = load_sae(
            self.config.sae_checkpoint,
            d_in=self.config.d_in,
            d_sae=self.config.d_sae,
        )

    def load_vectors(
        self, vectors: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """Accept vectors from memory or load them from the cached extraction."""
        if vectors is not None:
            self.mlp_vectors = {k: v.float() for k, v in vectors.items()}
            return

        vec_dir = os.path.join(
            self.config.output_dir, self.config.model_name_safe, "mlp_vectors"
        )
        loaded: Dict[str, torch.Tensor] = {}
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            path = os.path.join(vec_dir, safe_name(val), f"layer_{self.config.mlp_layer}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"MLP vector not found: {path}\n"
                    "Run the 'extract' step first:\n"
                    "  python -m SAE.run_sae_pipeline ... --modules extract"
                )
            loaded[val] = torch.load(path, map_location="cpu").float()

        self.mlp_vectors = loaded
        print(f"Loaded {len(self.mlp_vectors)} MLP vectors from disk.")

    # ── Step 1: Project through SAE ───────────────────────────────────────────
    def project_through_sae(self) -> None:
        """
        Encode every value's CAA vector through the SAE encoder.

        Each CAA vector is a *difference* vector (mean_pos − mean_neg) in the
        MLP activation space.  Passing it through the SAE encoder reveals which
        sparse features are differentially activated by the positive vs. negative
        side – i.e. which human-interpretable concepts the model associates with
        each Schwartz value.
        """
        assert self.sae is not None, "Call load_sae() first."
        assert self.mlp_vectors is not None, "Call load_vectors() first."

        feat_dir = self.config.subdir("features")
        feat_path = os.path.join(feat_dir, "feature_matrix.pt")

        if os.path.exists(feat_path):
            print("Loading cached feature matrix …")
            self.feature_matrix = torch.load(feat_path, map_location="cpu")
        else:
            features = []
            for val in SCHWARTZ_CIRCUMPLEX_ORDER:
                vec = self.mlp_vectors[val]       # (d_in,)
                feat = self.sae.encode(vec)        # (d_sae,)
                features.append(feat.detach())

            self.feature_matrix = torch.stack(features)  # (n_values, d_sae)
            torch.save(self.feature_matrix, feat_path)

        n_values, d_sae = self.feature_matrix.shape
        sparsity = (self.feature_matrix > self.config.activation_threshold).float().mean().item()
        print(f"Feature matrix: {n_values} values × {d_sae} features")
        print(f"  Mean fraction of active features per value: {sparsity:.3f}")

        self._save_feature_stats(feat_dir)

    def _save_feature_stats(self, feat_dir: str) -> None:
        stats: Dict = {}
        for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            feat = self.feature_matrix[i]   # type: ignore[index]
            active_mask = feat > self.config.activation_threshold
            stats[val] = {
                "n_active_features": int(active_mask.sum().item()),
                "mean_activation": float(feat.mean().item()),
                "max_activation": float(feat.max().item()),
                "top_10_features": feat.topk(10).indices.tolist(),
                "top_10_activations": feat.topk(10).values.tolist(),
            }
        with open(os.path.join(feat_dir, "value_feature_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

    # ── Step 2: Find common features ──────────────────────────────────────────
    def find_common_features(self) -> None:
        """
        Identify features that are high across *all* Schwartz values.

        Method: rank features by their *minimum* activation across all values
        (features with a high minimum are active for every value, not just some).
        Take the top-K such features.

        These "common" features are likely encoding general argumentative or
        language style patterns rather than value-specific content.  Removing
        them during purification sharpens the value-specific signal.
        """
        assert self.feature_matrix is not None

        feat_dir = self.config.subdir("features")

        min_act = self.feature_matrix.min(dim=0).values      # (d_sae,)
        top = min_act.topk(self.config.common_feature_top_k)
        self.common_features = top.indices.tolist()

        # What fraction of total activation mass do these features account for?
        common_mass = self.feature_matrix[:, self.common_features].sum().item()
        total_mass = self.feature_matrix.sum().item()
        frac = 100 * common_mass / total_mass if total_mass > 0 else 0.0

        result = {
            "method": "top_k_by_min_activation",
            "top_k": self.config.common_feature_top_k,
            "common_feature_ids": self.common_features,
            "min_activations_of_common": top.values.tolist(),
            "common_mass_fraction_pct": round(frac, 2),
        }
        with open(os.path.join(feat_dir, "common_features.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"Common features: {len(self.common_features)} identified")
        print(f"  They account for {frac:.1f}% of total activation mass")

        # Visualise common feature activation profile across values
        self._plot_common_feature_profile(feat_dir)

    def _plot_common_feature_profile(self, feat_dir: str) -> None:
        """Heatmap: values × top-20 common features (mean activation)."""
        top20_ids = self.common_features[:20]  # type: ignore[index]
        data = self.feature_matrix[:, top20_ids].numpy()  # (n_values, 20)

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            data,
            yticklabels=[_short_label(v) for v in SCHWARTZ_CIRCUMPLEX_ORDER],
            xticklabels=[f"F{i}" for i in top20_ids],
            cmap="YlOrRd",
            ax=ax,
        )
        ax.set_title("Top-20 Common Features: Activation per Schwartz Value")
        ax.set_xlabel("Feature ID")
        ax.set_ylabel("Schwartz Value")
        plt.tight_layout()
        plt.savefig(os.path.join(feat_dir, "common_feature_profile.png"), dpi=200)
        plt.close()

    # ── Step 3: Purify vectors ────────────────────────────────────────────────
    def purify_vectors(self) -> None:
        """
        Zero out the common features in the sparse code and decode back to the
        original (d_in,) activation space.

        The result is a "purified" CAA vector that should be more specific to
        the target value's content and less polluted by generic language signals.
        """
        assert self.sae is not None
        assert self.feature_matrix is not None
        assert self.common_features is not None
        assert self.mlp_vectors is not None

        feats_purified = self.feature_matrix.clone()
        feats_purified[:, self.common_features] = 0.0

        purified: Dict[str, torch.Tensor] = {}
        for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            z = feats_purified[i]               # (d_sae,)
            recon = self.sae.decode(z).detach() # (d_in,)
            purified[val] = recon

        self.purified_vectors = purified

        # Cache to disk
        pur_dir = os.path.join(
            self.config.output_dir, self.config.model_name_safe, "purified_vectors"
        )
        os.makedirs(pur_dir, exist_ok=True)
        for val, vec in purified.items():
            torch.save(vec, os.path.join(pur_dir, f"{safe_name(val)}.pt"))

        print(f"Purified {len(purified)} vectors ({len(self.common_features)} common features zeroed)")
        print("  Norm change (first 5 values):")
        for val in SCHWARTZ_CIRCUMPLEX_ORDER[:5]:
            raw_n = self.mlp_vectors[val].norm().item()
            pur_n = purified[val].norm().item()
            print(f"    {val}: {raw_n:.4f} → {pur_n:.4f}")

    # ── Step 4: Geometry analysis ─────────────────────────────────────────────
    def run_geometry(
        self,
        vectors: Dict[str, torch.Tensor],
        label: str,
    ) -> dict:
        """
        Compute Spearman ρ between empirical cosine similarity and the
        theoretical Schwartz relation matrix.  Also produces heatmaps, UMAP,
        PCA, and MDS circumplex plots.

        Args:
            vectors: Dict[value → (d_in,) tensor]
            label:   Short label used for output subdirectory and plot titles
                     (e.g. ``"raw_mlp"`` or ``"purified"``).

        Returns:
            Dict with ``spearman_rho``, ``p_value``, ``label``.
        """
        out_dir = self.config.subdir(f"geometry_{label}")
        unit_vecs = _unit(vectors)

        emp_sim = _empirical_sim_matrix(unit_vecs)
        theo_sim = _theoretical_matrix(self.config.relations_path)

        triu = np.triu_indices(len(SCHWARTZ_CIRCUMPLEX_ORDER), k=1)
        rho, pval = spearmanr(emp_sim[triu], theo_sim[triu])

        metrics = {
            "label": label,
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "n_pairs": int(len(emp_sim[triu])),
        }
        with open(os.path.join(out_dir, "spearman_report.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  [{label}] Spearman ρ = {rho:+.4f}   p = {pval:.4g}")

        # Heatmaps
        _plot_heatmap(
            emp_sim,
            f"Empirical Cosine Similarity [{label}]",
            os.path.join(out_dir, "empirical_heatmap.png"),
        )
        _plot_heatmap(
            theo_sim,
            "Theoretical Schwartz Relationships",
            os.path.join(out_dir, "theoretical_heatmap.png"),
        )

        # Low-dimensional projections
        X = np.stack([unit_vecs[v].numpy() for v in SCHWARTZ_CIRCUMPLEX_ORDER])
        self._plot_umap(X, out_dir, label)
        self._plot_pca(X, out_dir, label)
        self._plot_mds(emp_sim, out_dir, label)

        return metrics

    # ── Projection plots ──────────────────────────────────────────────────────
    def _plot_umap(self, X: np.ndarray, out_dir: str, label: str) -> None:
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=self.config.seed)
        coords = reducer.fit_transform(X)
        _plot_embedding(coords, f"UMAP – {label}", os.path.join(out_dir, "umap_2d.png"))

    def _plot_pca(self, X: np.ndarray, out_dir: str, label: str) -> None:
        coords = PCA(n_components=2, random_state=self.config.seed).fit_transform(X)
        _plot_embedding(coords, f"PCA – {label}", os.path.join(out_dir, "pca_2d.png"))

    def _plot_mds(self, emp_sim: np.ndarray, out_dir: str, label: str) -> None:
        n = len(SCHWARTZ_CIRCUMPLEX_ORDER)
        dist = np.clip(1.0 - emp_sim, 0, None)

        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=self.config.seed,
            n_init=4,
            normalized_stress="auto",
        )
        X_mds = mds.fit_transform(dist)

        # Align empirical MDS to the ideal theoretical circumplex
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        X_circle = np.column_stack([np.cos(angles), np.sin(angles)])
        R, _ = orthogonal_procrustes(X_mds, X_circle)
        X_mds_aligned = X_mds @ R

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.add_patch(plt.Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--"))

        for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            tx, ty = X_circle[i]
            ex, ey = X_mds_aligned[i]
            color = GROUP_COLORS.get(value_to_group(val), "black")

            ax.plot(tx, ty, "x", color="gray", markersize=8)
            ax.plot(ex, ey, "o", color=color, markersize=8)
            ax.plot([tx, ex], [ty, ey], color="gray", alpha=0.3, linestyle=":")
            ax.annotate(
                _short_label(val),
                (ex, ey),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )

        ax.legend(handles=_legend_handles())
        ax.set_title(f"MDS Circumplex – {label}\n(grey ×: theory,  coloured ●: empirical)")
        ax.set_aspect("equal")
        lim = max(np.abs(X_mds_aligned).max(), 1.0) * 1.25
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mds_circumplex.png"), dpi=200)
        plt.close()

    # ── Step 5: Geometry comparison ───────────────────────────────────────────
    def geometry_comparison(self) -> dict:
        """Run geometry for raw and purified vectors and compare Spearman ρ."""
        assert self.mlp_vectors is not None and self.purified_vectors is not None

        print("\n[Geometry] Raw MLP vectors …")
        raw_m = self.run_geometry(self.mlp_vectors, "raw_mlp")

        print("[Geometry] Purified vectors …")
        pur_m = self.run_geometry(self.purified_vectors, "purified")

        comparison = {
            "raw_spearman_rho": raw_m["spearman_rho"],
            "raw_p_value": raw_m["p_value"],
            "purified_spearman_rho": pur_m["spearman_rho"],
            "purified_p_value": pur_m["p_value"],
            "delta_rho": pur_m["spearman_rho"] - raw_m["spearman_rho"],
        }

        out_path = os.path.join(
            self.config.output_dir, self.config.model_name_safe, "geometry_comparison.json"
        )
        with open(out_path, "w") as f:
            json.dump(comparison, f, indent=2)

        # Side-by-side bar chart
        self._plot_rho_comparison(raw_m["spearman_rho"], pur_m["spearman_rho"])

        print("\n=== Geometry Comparison ===")
        print(f"  Raw MLP vectors  ρ = {comparison['raw_spearman_rho']:+.4f}")
        print(f"  Purified vectors ρ = {comparison['purified_spearman_rho']:+.4f}")
        print(f"  Δρ               = {comparison['delta_rho']:+.4f}")

        return comparison

    def _plot_rho_comparison(self, raw_rho: float, pur_rho: float) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(["Raw MLP", "Purified"], [raw_rho, pur_rho], color=["#607D8B", "#4CAF50"])
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, val in zip(bars, [raw_rho, pur_rho]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * (1 if val >= 0 else -1),
                f"{val:+.4f}",
                ha="center",
                fontsize=11,
            )
        ax.set_ylabel("Spearman ρ vs. Schwartz theory")
        ax.set_title("Geometry Alignment Before/After SAE Purification")
        ax.set_ylim(min(raw_rho, pur_rho, 0) - 0.05, max(raw_rho, pur_rho, 0) + 0.1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.config.output_dir, self.config.model_name_safe, "rho_comparison.png"
            ),
            dpi=200,
        )
        plt.close()

    # ── Step 6: Disjointness test ─────────────────────────────────────────────
    def disjointness_test(self) -> dict:
        """
        For each higher-order group, collect the union of top-K features across
        its member values (after removing common features).  Then compute
        pairwise Jaccard similarity between group feature sets.

        The hypothesis (from Schwartz theory) is that *opposing* groups
        (Conservation ↔ Openness-to-Change; Self-Enhancement ↔ Self-Transcendence)
        should have *low* Jaccard similarity (disjoint feature sets).
        """
        assert self.feature_matrix is not None

        out_dir = self.config.subdir("disjointness")
        k = self.config.top_features_per_value

        # Per-value top-K features (excluding common features to focus on value-specific signal)
        topk_per_value: Dict[str, set] = {}
        for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            feat = self.feature_matrix[i].clone()
            if self.common_features is not None:
                feat[self.common_features] = 0.0
            topk_per_value[val] = set(feat.topk(k).indices.tolist())

        # Group feature sets = union of member top-K sets
        group_features: Dict[str, set] = {
            group: set().union(*(topk_per_value[v] for v in members))
            for group, members in HIGHER_ORDER_GROUPS.items()
        }

        # Pairwise Jaccard
        groups = list(HIGHER_ORDER_GROUPS.keys())
        n = len(groups)
        jaccard_matrix = np.zeros((n, n))
        pairwise: Dict[str, dict] = {}

        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                s1, s2 = group_features[g1], group_features[g2]
                if i == j:
                    j_sim = 1.0
                else:
                    inter = len(s1 & s2)
                    union = len(s1 | s2)
                    j_sim = inter / union if union > 0 else 0.0
                jaccard_matrix[i, j] = j_sim
                if i < j:
                    is_opp = (g1, g2) in OPPOSING_PAIRS or (g2, g1) in OPPOSING_PAIRS
                    pairwise[f"{g1} vs {g2}"] = {
                        "jaccard": round(j_sim, 4),
                        "intersection_features": len(s1 & s2),
                        "union_features": len(s1 | s2),
                        "g1_features": len(s1),
                        "g2_features": len(s2),
                        "is_opposing_pair": is_opp,
                    }

        results = {
            "top_k_per_value": k,
            "common_features_excluded": self.common_features is not None,
            "group_feature_set_sizes": {g: len(fs) for g, fs in group_features.items()},
            "pairwise_jaccard": pairwise,
        }
        with open(os.path.join(out_dir, "disjointness_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Jaccard heatmap
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            jaccard_matrix,
            xticklabels=groups,
            yticklabels=groups,
            cmap="YlOrRd_r",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(
            f"Feature-Set Jaccard Similarity between Higher-Order Groups\n"
            f"(top-{k} features per value, common features excluded)"
        )
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "jaccard_heatmap.png"), dpi=200)
        plt.close()

        # Overlap bar chart for opposing pairs
        self._plot_opposing_overlap(pairwise, out_dir, groups, group_features)

        print("\n[Disjointness] Opposing pair results:")
        for pair in OPPOSING_PAIRS:
            key = f"{pair[0]} vs {pair[1]}"
            if key in pairwise:
                r = pairwise[key]
                print(
                    f"  {key}: Jaccard={r['jaccard']:.4f}  "
                    f"(overlap={r['intersection_features']} / union={r['union_features']})"
                )

        return results

    def _plot_opposing_overlap(
        self,
        pairwise: Dict[str, dict],
        out_dir: str,
        groups: List[str],
        group_features: Dict[str, set],
    ) -> None:
        """Stacked bar showing exclusive vs. shared features for opposing pairs."""
        fig, axes = plt.subplots(1, len(OPPOSING_PAIRS), figsize=(5 * len(OPPOSING_PAIRS), 5))
        if len(OPPOSING_PAIRS) == 1:
            axes = [axes]

        for ax, (g1, g2) in zip(axes, OPPOSING_PAIRS):
            s1, s2 = group_features[g1], group_features[g2]
            only_g1 = len(s1 - s2)
            shared = len(s1 & s2)
            only_g2 = len(s2 - s1)

            bars = ax.bar([g1, g2], [only_g1 + shared, only_g2 + shared])
            ax.bar([g1], [only_g1], color=GROUP_COLORS.get(g1, "blue"), label=f"Exclusive {g1}")
            ax.bar([g1], [shared], bottom=[only_g1], color="lightgrey", label="Shared")
            ax.bar([g2], [shared], color="lightgrey")
            ax.bar([g2], [only_g2], bottom=[shared], color=GROUP_COLORS.get(g2, "orange"), label=f"Exclusive {g2}")
            ax.set_title(f"{g1}\nvs\n{g2}")
            ax.set_ylabel("Number of features")
            ax.legend(fontsize=7)
            key = f"{g1} vs {g2}"
            if key in pairwise:
                ax.set_xlabel(f"Jaccard = {pairwise[key]['jaccard']:.4f}", fontsize=9)

        plt.suptitle("Feature Overlap Between Opposing Schwartz Groups", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "opposing_overlap.png"), dpi=200)
        plt.close()

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run_full_analysis(
        self, vectors: Optional[Dict[str, torch.Tensor]] = None
    ) -> dict:
        """
        Run all SAE analysis steps in order.

        Args:
            vectors: Optional pre-computed MLP vectors (from the extract step).
                     If None, loads from the cached ``mlp_vectors/`` directory.
        """
        print("\n── Step 1/5: Loading SAE ──────────────────────────────────────")
        self.load_sae()

        print("\n── Step 2/5: Loading MLP vectors ──────────────────────────────")
        self.load_vectors(vectors)

        print("\n── Step 3/5: Projecting through SAE encoder ───────────────────")
        self.project_through_sae()

        print("\n── Step 4/5: Finding common features ──────────────────────────")
        self.find_common_features()

        print("\n── Step 5a: Purifying vectors ──────────────────────────────────")
        self.purify_vectors()

        print("\n── Step 5b: Geometry comparison ────────────────────────────────")
        geo_results = self.geometry_comparison()

        print("\n── Step 5c: Disjointness test ──────────────────────────────────")
        disj_results = self.disjointness_test()

        # ── Summary ───────────────────────────────────────────────────────────
        opposing_summary = {
            k: v
            for k, v in disj_results["pairwise_jaccard"].items()
            if v.get("is_opposing_pair")
        }

        summary = {
            "model": self.config.model_name,
            "sae_checkpoint": self.config.sae_checkpoint,
            "mlp_layer": self.config.mlp_layer,
            "common_features_removed": len(self.common_features),  # type: ignore[arg-type]
            "geometry": geo_results,
            "opposing_pair_jaccard": opposing_summary,
        }

        out_path = os.path.join(
            self.config.output_dir, self.config.model_name_safe, "analysis_summary.json"
        )
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n══════════════════════ Analysis Complete ══════════════════════")
        print(json.dumps(summary, indent=2))

        return summary
