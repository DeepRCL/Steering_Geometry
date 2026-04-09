"""
Main steering pipeline: layer selection, per-value vector training, and evaluation.

Layer selection uses mean normalized L2 separation of activations.
Evaluation uses geometry analysis of steering vectors against the theoretical
Schwartz value relationship matrix (Spearman/Pearson correlation, Procrustes
alignment, silhouette scores, and dimensionality-reduction visualizations).
"""

import dataclasses
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Ensure steering_opt is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import steering_opt

from .config import (
    SteeringConfig,
    SCHWARTZ_CIRCUMPLEX_ORDER,
    HIGHER_ORDER_GROUPS,
    GROUP_COLORS,
    value_to_group,
)
from . import data_utils


class SteeringPipeline:
    """
    End-to-end pipeline for value-steering vector optimization.

    Steps:
        1. Load model and tokenizer
        2. Load dataset, perform stratified split
        3. (Optional) Layer sweep to find best layer
        4. Train one steering vector per Schwartz value
        5. Evaluate each vector on the held-out validation set
        6. Save vectors and metrics
    """

    def __init__(self, config: SteeringConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_rows: List[dict] = []
        self.val_rows: List[dict] = []
        self.values: List[str] = []

    # ─── Model Loading ───────────────────────────────────────────────────

    def load_model(self):
        """Load the HuggingFace model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._log(f"Loading model: {self.config.model_name}")
        self._log(f"  dtype: {self.config.torch_dtype}, device: {self.config.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.get_dtype(),
            trust_remote_code=True
        ).to(self.config.device)

        torch.set_default_device(self.config.device)

        n_layers = self.model.config.num_hidden_layers
        d_model = self.model.config.hidden_size
        self._log(f"  Loaded: {n_layers} layers, d_model={d_model}")
        self._log("")

    # ─── Data Preparation ────────────────────────────────────────────────

    def prepare_data(self):
        """Load dataset and perform stratified train/val split."""
        self._log(f"Loading dataset: {self.config.dataset_path}")
        all_rows = data_utils.load_dataset(self.config.dataset_path)
        self.values = data_utils.get_unique_values(all_rows)

        self._log(f"  Total rows: {len(all_rows)}, Unique values: {len(self.values)}")
        self._log(f"  Train ratio: {self.config.train_ratio}")

        self.train_rows, self.val_rows = data_utils.stratified_split(
            all_rows,
            train_ratio=self.config.train_ratio,
            seed=self.config.random_seed,
        )

        if self.config.verbose:
            data_utils.print_split_summary(
                self.train_rows, self.val_rows, self.values
            )

    # ─── Layer Selection (Mean Normalized L2 Separation) ────────────────

    def _get_sweep_candidates(self) -> List[int]:
        """Determine which layers to sweep."""
        if self.config.layer_sweep_candidates is not None:
            return self.config.layer_sweep_candidates

        n_layers = self.model.config.num_hidden_layers
        n_cand = min(self.config.layer_sweep_n_candidates, n_layers)

        start = max(1, int(n_layers * 0.15))
        end = int(n_layers * 0.85)
        step = max(1, (end - start) // (n_cand - 1)) if n_cand > 1 else 1
        candidates = list(range(start, end + 1, step))[:n_cand]

        return candidates

    @staticmethod
    def _compute_mean_activation_separation(
        activations_pos: List[torch.Tensor],
        activations_neg: List[torch.Tensor],
    ) -> float:
        """
        Compute scale-normalized L2 separation for a single value at a single layer.

        Each sample activation is L2-normalized to unit norm, then we compute
        || mean(normalized_pos) - mean(normalized_neg) ||_2.
        This reduces bias toward later layers with larger residual-stream magnitudes.
        """
        n = min(len(activations_pos), len(activations_neg))
        if n == 0:
            return 0.0

        pos_stack = torch.stack(activations_pos[:n])
        neg_stack = torch.stack(activations_neg[:n])

        pos_stack = pos_stack / pos_stack.norm(dim=1, keepdim=True).clamp_min(1e-12)
        neg_stack = neg_stack / neg_stack.norm(dim=1, keepdim=True).clamp_min(1e-12)

        pos_mean = pos_stack.mean(dim=0)
        neg_mean = neg_stack.mean(dim=0)
        return float(torch.norm(pos_mean - neg_mean, p=2).item())

    @torch.no_grad()
    def _extract_last_token_activation(
        self, text: str, layer: int
    ) -> torch.Tensor:
        """Run a forward pass and capture the last-token hidden state at `layer`."""
        activs_list: list = []
        hook = (layer, steering_opt.make_activs_hook_hf(activs_list))

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
            self.config.device
        )

        with steering_opt.hf_hooks_contextmanager(self.model, [hook]):
            self.model(input_ids)

        last_token_activ = activs_list[0][0, -1, :].detach().cpu()
        return last_token_activ

    def select_layer(self) -> int:
        """
        Layer sweep using mean normalized L2 separation of activations.

        For each candidate layer and each Schwartz value, extracts last-token
        activations for positive and negative completions, L2-normalizes each
        sample, and measures || mean(pos) - mean(neg) ||_2.
        Picks the layer with the highest mean separation across values.

        Returns:
            Best layer index.
        """
        candidates = self._get_sweep_candidates()
        self._log(f"Layer sweep (normalized L2 separation) over candidates: {candidates}")

        sweep_values = [v for v in self.values if v in SCHWARTZ_CIRCUMPLEX_ORDER]
        if not sweep_values:
            sweep_values = self.values

        # activations[value][layer] = {"pos": [...], "neg": [...]}
        activations: Dict[str, Dict[int, Dict[str, List[torch.Tensor]]]] = {}

        self._log("  Extracting activations...")
        pbar_values = tqdm(sweep_values, desc="Extracting activations", leave=True)
        for value in pbar_values:
            pbar_values.set_description(f"Activations: {value}")
            train_value_rows = data_utils.get_rows_for_value(self.train_rows, value)

            if not train_value_rows:
                continue

            rng = random.Random(self.config.random_seed)
            sample_rows = rng.sample(
                train_value_rows,
                min(self.config.layer_sweep_n_samples, len(train_value_rows)),
            )

            activations[value] = {}
            for layer in candidates:
                pos_acts = []
                neg_acts = []
                for row in sample_rows:
                    prompt = data_utils.format_prompt(
                        row["question"],
                        self.tokenizer,
                        self.config.use_chat_template,
                        self.config.prompt_template,
                    )
                    pos_text = prompt + " " + row["positive_answer"]
                    neg_text = prompt + " " + row["negative_answer"]

                    pos_acts.append(self._extract_last_token_activation(pos_text, layer))
                    neg_acts.append(self._extract_last_token_activation(neg_text, layer))

                activations[value][layer] = {"pos": pos_acts, "neg": neg_acts}

        mean_scores: Dict[int, float] = {}
        per_value_scores: Dict[int, Dict[str, float]] = {}

        for layer in candidates:
            layer_value_scores = {}
            for value in sweep_values:
                if value not in activations:
                    continue
                acts = activations[value].get(layer)
                if acts is None:
                    continue
                layer_value_scores[value] = self._compute_mean_activation_separation(
                    acts["pos"], acts["neg"]
                )

            per_value_scores[layer] = layer_value_scores
            scores_list = list(layer_value_scores.values())
            mean_scores[layer] = float(np.mean(scores_list)) if scores_list else 0.0

        if not mean_scores or all(v == 0.0 for v in mean_scores.values()):
            best_layer = candidates[len(candidates) // 2]
            self._log(f"\n  Warning: all layers scored 0. Falling back to layer {best_layer}\n")
        else:
            best_layer = max(candidates, key=lambda l: mean_scores.get(l, 0.0))
            self._log(
                f"\n  Best layer: {best_layer} "
                f"(mean normalized L2 sep = {mean_scores[best_layer]:.4f})\n"
            )

        for layer in candidates:
            self._log(f"    Layer {layer}: mean normalized L2 sep = {mean_scores.get(layer, 0):.4f}")

        sweep_path = os.path.join(self.config.output_dir, "layer_sweep.json")
        scores_dict = {
            str(layer): {
                "mean_normalized_l2_separation": mean_scores.get(layer, 0),
                "per_value_normalized_l2_separation": {
                    k: round(v, 6) for k, v in per_value_scores.get(layer, {}).items()
                },
            }
            for layer in candidates
        }
        with open(sweep_path, "w") as f:
            json.dump(
                {"candidates": candidates, "scores": scores_dict, "best_layer": best_layer},
                f, indent=2,
            )

        return best_layer

    # ─── Training ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_layers(layer: Union[int, List[int]]) -> List[int]:
        if isinstance(layer, int):
            return [layer]
        return list(layer)

    def train_vectors(self, layer: Union[int, List[int]]) -> Dict[str, torch.Tensor]:
        """
        Train one steering vector per Schwartz value.

        Args:
            layer: The layer at which to optimize steering vectors.

        Returns:
            Dict mapping value name -> optimized steering vector.
        """
        self._log(f"Training steering vectors at layer {layer}")
        self._log(f"  lr={self.config.lr}, max_iters={self.config.max_iters}, "
                   f"max_norm={self.config.max_norm}, alpha={self.config.alpha}")
        self._log("")

        vectors: Dict[str, torch.Tensor] = {}
        train_info: Dict[str, dict] = {}

        # pbar_train = tqdm(self.values, desc="Training Vectors", leave=True)
        for value in self.values:
            # pbar_train.set_description(f"Training: {value}")
            train_value_rows = data_utils.get_rows_for_value(
                self.train_rows, value
            )

            if not train_value_rows:
                self._log(f"    Skipping (no training data)")
                continue

            datapoints = data_utils.create_datapoints(
                train_value_rows,
                tokenizer=self.tokenizer,
                use_chat_template=self.config.use_chat_template,
                prompt_template=self.config.prompt_template,
                n_samples=self.config.n_training_samples,
                seed=self.config.random_seed,
            )

            self._log(f"    Using {len(datapoints)} training datapoints")

            t0 = time.time()
            
            vector, info = steering_opt.optimize_vector(
                self.model,
                datapoints,
                layer,
                tokenizer=self.tokenizer,
                lr=self.config.lr,
                max_iters=self.config.max_iters,
                return_info=True,
                show_iter_progress=False,
            )            

            elapsed = time.time() - t0
            vectors[value] = vector.detach().clone()
            train_info[value] = {
                "iters": info.get("iters", -1),
                "loss": info.get("loss", -1),
                "norm": info.get("norm", -1),
                "time_sec": round(elapsed, 2),
            }
            self._log(
                f"    ✓ Done in {elapsed:.1f}s | "
                f"iters={info.get('iters')} | "
                f"loss={info.get('loss', 0):.4f} | "
                f"norm={info.get('norm', 0):.2f}"
            )

        # Save training info
        info_path = os.path.join(self.config.output_dir, "training_info.json")
        with open(info_path, "w") as f:
            json.dump(train_info, f, indent=2)

        self._log(f"\n  Trained {len(vectors)}/{len(self.values)} vectors\n")
        return vectors

    # ─── Geometry Analysis ─────────────────────────────────────────────

    @staticmethod
    def _circular_step_distance(i: int, j: int, n: int) -> int:
        """Shortest step distance between positions *i* and *j* on a circle of size *n*."""
        return min(abs(i - j), n - abs(i - j))

    @staticmethod
    def _plot_embedding_2d(out_path: str, title: str, coords: np.ndarray):
        """Scatter plot of a 2-D embedding, coloured by Schwartz higher-order group."""
        plt.figure(figsize=(10, 8))
        for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            group = value_to_group(val)
            color = GROUP_COLORS.get(group, "black")
            plt.scatter(coords[i, 0], coords[i, 1], c=color, s=100)
            plt.annotate(
                val.split(":")[-1].strip(),
                (coords[i, 0], coords[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        from matplotlib.lines import Line2D
        legend_els = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=10, label=g)
            for g, c in GROUP_COLORS.items()
        ]
        plt.legend(handles=legend_els, loc="best")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def analyze_geometry(
        self, vectors: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Full geometry analysis of steering vectors against the theoretical
        Schwartz circumplex.

        Computes Spearman/Pearson correlations, silhouette scores, within-
        vs across-group cosine similarities, Procrustes alignment to the
        theoretical circle, and generates heatmaps plus dimensionality-
        reduction plots (UMAP, PCA, t-SNE, MDS with circumplex overlay).

        Returns:
            Dict of geometry metrics (also saved as geometry_metrics.json).
        """
        self._log("Running geometry analysis...")
        out_dir = os.path.join(self.config.output_dir, "geometry")
        os.makedirs(out_dir, exist_ok=True)

        unit_vectors: Dict[str, torch.Tensor] = {}
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            vec = vectors[val].detach().cpu().float()
            norm = vec.norm()
            unit_vectors[val] = vec / norm if norm > 0 else vec

        num_values = len(SCHWARTZ_CIRCUMPLEX_ORDER)

        # ── 1. Empirical similarity matrix ────────────────────────────
        empirical_sim = np.zeros((num_values, num_values))
        for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
                empirical_sim[i, j] = F.cosine_similarity(
                    unit_vectors[v1], unit_vectors[v2], dim=0
                ).item()

        # ── 2. Theoretical similarity matrix ──────────────────────────
        with open(self.config.relations_path, "r") as f:
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

        self._log(
            f"Spearman correlation between theoretical and empirical "
            f"similarities: rho={rho:.4f}, p={p_val:.4g}"
        )

        # ── 4. Heatmaps ──────────────────────────────────────────────
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            empirical_sim,
            xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
            yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
            cmap="coolwarm", vmin=-1, vmax=1,
        )
        plt.title("Empirical Cosine Similarities")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            theoretical_sim,
            xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
            yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
            cmap="coolwarm", vmin=-1, vmax=1,
        )
        plt.title("Theoretical Relationships")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "theoretical_similarity_heatmap.png"), dpi=300)
        plt.close()

        # ── 5. Dimensionality-reduction projections ───────────────────
        X = np.stack([unit_vectors[v].numpy() for v in SCHWARTZ_CIRCUMPLEX_ORDER])

        reducer = umap.UMAP(n_components=2, metric="cosine",
                            n_jobs=1, random_state=self.config.random_seed)
        X_umap = reducer.fit_transform(X)
        self._plot_embedding_2d(
            os.path.join(out_dir, "umap_2d.png"),
            "UMAP 2D Projection of Steering Vectors", X_umap,
        )

        X_pca = PCA(n_components=2, random_state=self.config.random_seed).fit_transform(X)
        self._plot_embedding_2d(
            os.path.join(out_dir, "pca_2d.png"),
            "PCA 2D Projection of Steering Vectors", X_pca,
        )

        perplexity = min(5, max(2, num_values - 1))
        X_tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=self.config.random_seed,
        ).fit_transform(X)
        self._plot_embedding_2d(
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
            random_state=self.config.random_seed,
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
        for i in range(num_values):
            for j in range(i + 1, num_values):
                same = (
                    value_to_group(SCHWARTZ_CIRCUMPLEX_ORDER[i])
                    == value_to_group(SCHWARTZ_CIRCUMPLEX_ORDER[j])
                )
                same_group_mask.append(same)
                different_group_mask.append(not same)

                step = self._circular_step_distance(i, j, num_values)
                circular_step_flat.append(step)
                if step == 1:
                    neighbor_empirical.append(empirical_sim[i, j])
                if step == num_values // 2:
                    opposite_empirical.append(empirical_sim[i, j])

        same_group_mask = np.array(same_group_mask, dtype=bool)
        different_group_mask = np.array(different_group_mask, dtype=bool)
        circular_step_flat = np.array(circular_step_flat, dtype=float)

        within_group_mean = float(emp_flat[same_group_mask].mean())
        across_group_mean = float(emp_flat[different_group_mask].mean())
        within_minus_across = within_group_mean - across_group_mean

        neighbor_mean = float(np.mean(neighbor_empirical))
        opposite_mean = float(np.mean(opposite_empirical))
        neighbor_minus_opposite = neighbor_mean - opposite_mean
        circular_distance_spearman, circular_distance_p = spearmanr(
            emp_flat, -circular_step_flat
        )

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
            "procrustes_disparity": float(procrustes_disparity),
            "procrustes_rmse_after_alignment": procrustes_rmse,
            "mds_stress": float(mds.stress_),
        }
        with open(os.path.join(out_dir, "geometry_metrics.json"), "w") as f:
            json.dump(geometry_metrics, f, indent=2)

        # ── 8. MDS circumplex overlay plot ────────────────────────────
        plt.figure(figsize=(12, 12))
        circle_patch = plt.Circle((0, 0), 1, color="lightgray",
                                  fill=False, linestyle="--")
        plt.gca().add_patch(circle_patch)

        for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            tx, ty = X_circle[i]
            plt.plot(tx, ty, "x", color="gray", markersize=8)

            ex, ey = X_mds_aligned[i]
            group = value_to_group(val)
            color = GROUP_COLORS.get(group, "black")

            plt.plot(ex, ey, "o", color=color, markersize=8)
            plt.plot([tx, ex], [ty, ey], color="gray", alpha=0.3, linestyle=":")

            label = val.split(":")[-1].strip()
            plt.annotate(label, (ex, ey), xytext=(5, 5),
                         textcoords="offset points", fontsize=9, color=color)

        plt.title("2D MDS Aligned to Theoretical Circumplex")
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="datalim")
        scale = np.max(np.abs(X_mds_aligned))
        lim = max(1.2, scale * 1.2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mds_circumplex.png"), dpi=300)
        plt.close()

        # ── 9. Theory vs empirical scatter ────────────────────────────
        plt.figure(figsize=(8, 5))
        jitter = np.random.default_rng(self.config.random_seed).normal(
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

        self._log("Geometry analysis complete!")
        return geometry_metrics

    # ─── Save / Load ─────────────────────────────────────────────────────

    def save_vectors(
        self, vectors: Dict[str, torch.Tensor], layer: int
    ):
        """
        Save each value's steering vector as a .pt file with metadata JSON.

        File naming: {sanitized_value_name}.pt  and  {sanitized_value_name}.json
        """
        vectors_dir = os.path.join(self.config.output_dir, "vectors")
        os.makedirs(vectors_dir, exist_ok=True)

        manifest = {}
        for value, vector in vectors.items():
            safe_name = self._sanitize_filename(value)

            # Save vector
            vec_path = os.path.join(vectors_dir, f"{safe_name}.pt")
            torch.save(vector.detach().cpu(), vec_path)

            # Save metadata
            meta = {
                "value": value,
                "layer": layer,
                "norm": vector.norm().item(),
                "d_model": vector.shape[0],
                "model_name": self.config.model_name,
                "alpha": self.config.alpha,
            }
            meta_path = os.path.join(vectors_dir, f"{safe_name}.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            manifest[value] = {
                "vector_file": f"{safe_name}.pt",
                "metadata_file": f"{safe_name}.json",
                "layer": layer,
                "norm": round(vector.norm().item(), 4),
            }

        # Save manifest
        manifest_path = os.path.join(vectors_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self._log(f"Saved {len(vectors)} vectors to {vectors_dir}/")

    @staticmethod
    def load_vectors(vectors_dir: str) -> Dict[str, Tuple[torch.Tensor, dict]]:
        """
        Load all saved vectors and their metadata from a directory.

        Returns:
            Dict mapping value name -> (vector_tensor, metadata_dict)
        """
        manifest_path = os.path.join(vectors_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        result = {}
        for value, info in manifest.items():
            vec = torch.load(
                os.path.join(vectors_dir, info["vector_file"]),
                map_location="cpu",
                weights_only=True,
            )
            meta_path = os.path.join(vectors_dir, info["metadata_file"])
            with open(meta_path) as f:
                meta = json.load(f)
            result[value] = (vec, meta)

        return result

    # ─── Full Pipeline ───────────────────────────────────────────────────

    def run(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, dict], int]:
        """
        Run the full pipeline end-to-end.

        Returns:
            (vectors, metrics, best_layer)
        """
        self._log("=" * 60)
        self._log("  Value-Steering Optimization Pipeline")
        self._log("=" * 60)
        self._log("")

        # 1. Load model
        self.load_model()

        # Nest output under a model-specific subdirectory
        safe_model = (
            self.config.model_name.replace("/", "__").replace(" ", "_")
        )
        self.config.output_dir = os.path.join(self.config.output_dir, safe_model)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # 2. Prepare data
        self.prepare_data()

        # 3. Layer selection
        if self.config.layer_sweep_enabled:
            best_layer = self.select_layer()
        else:
            candidates = self._get_sweep_candidates()
            # If --layer was given, candidates is [that_layer]; otherwise pick middle
            best_layer = candidates[0] if len(candidates) == 1 else candidates[len(candidates) // 2]
            self._log(f"Layer sweep disabled. Using layer {best_layer}\n")

        # 4. Train vectors
        vectors = self.train_vectors(best_layer)

        # 5. Geometry analysis
        metrics = self.analyze_geometry(vectors)

        # 6. Save vectors
        if self.config.save_vectors:
            self.save_vectors(vectors, best_layer)

        # Save full config
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2, default=str)

        self._log("=" * 60)
        self._log("  Pipeline complete!")
        self._log(f"  Results saved to: {self.config.output_dir}/")
        self._log("=" * 60)

        return vectors, metrics, best_layer

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.config.verbose:
            print(msg)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Convert a value name to a safe filename."""
        return (
            name.lower()
            .replace(": ", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

