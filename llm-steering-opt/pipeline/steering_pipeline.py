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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from tqdm import tqdm

# Ensure steering_opt and repo-root shared utils are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import steering_opt
from shared import schwartz_eval

from .config import (
    SteeringConfig,
    SCHWARTZ_CIRCUMPLEX_ORDER,
    HIGHER_ORDER_GROUPS,
    GROUP_COLORS,
    value_to_group,
)
from . import data_utils

# ─── Plotting Constants ──────────────────────────────────────────────────────
PLOT_LABEL_FONTSIZE = 13
PLOT_TITLE_FONTSIZE = 18
PLOT_AXIS_FONTSIZE = 12
PLOT_LEGEND_FONTSIZE = 13
PLOT_ANNOTATION_FONTSIZE = 9
PLOT_SCATTER_SIZE = 150
PLOT_MARKER_SIZE = 12
PLOT_MARKER_RADIUS = 0.055
EMBEDDING_FIGURE_SIZE = (14, 11)
HEATMAP_FIGURE_SIZE = (14, 12)
MDS_FIGURE_SIZE = (15, 15)
SCATTER_FIGURE_SIZE = (8, 5)
DIFFERENCE_HEATMAP_SIZE = (12, 10)

BOUNDARY_GROUPS = {
    "Hedonism": ("Openness to Change", "Self-Enhancement"),
    "Face": ("Self-Enhancement", "Conservation"),
    "Humility": ("Conservation", "Self-Transcendence"),
}


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

    # ─── Plotting Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _short_value_label(value: str) -> str:
        """Extract short label from value name (after ':' if present)."""
        return value.split(":")[-1].strip() if ":" in value else value

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
            dtype=self.config.get_dtype(),
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

    def select_layer(self) -> Tuple[int, Dict[str, Any]]:
        """
        Layer sweep using mean normalized L2 separation of activations.

        For each candidate layer and each Schwartz value, extracts last-token
        activations for positive and negative completions, L2-normalizes each
        sample, and measures || mean(pos) - mean(neg) ||_2.
        Picks the layer with the highest mean separation across values.

        Returns:
            Best layer index and the JSON-ready layer sweep payload.
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

        scores_dict = {
            str(layer): {
                "mean_normalized_l2_separation": mean_scores.get(layer, 0),
                "per_value_normalized_l2_separation": {
                    k: round(v, 6) for k, v in per_value_scores.get(layer, {}).items()
                },
            }
            for layer in candidates
        }
        sweep_payload = {
            "candidates": candidates,
            "scores": scores_dict,
            "best_layer": best_layer,
        }

        return best_layer, sweep_payload

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
                   f"max_norm={self.config.max_norm}, alpha={self.config.alpha}, train_ratio={self.config.train_ratio}")
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

    # ─── Steering Evaluation (Log-Likelihood) ─────────────────────────

    @torch.no_grad()
    def _compute_logprob(
        self,
        prompt: str,
        completion: str,
        hook_infos: Optional[list] = None,
    ) -> float:
        """
        Mean per-token logprob of ``completion`` given ``prompt`` (ODESteer eval).

        Single forward on ``prompt + completion``; hooks apply to the full pass.
        """
        device = next(self.model.parameters()).device
        full_inputs, prompt_len, answer_ids = schwartz_eval.prepare_qa_completion_inputs(
            self.tokenizer, prompt, completion, device
        )
        if hook_infos:
            with steering_opt.hf_hooks_contextmanager(self.model, hook_infos):
                outputs = self.model(
                    input_ids=full_inputs.input_ids,
                    attention_mask=full_inputs.get("attention_mask"),
                )
        else:
            outputs = self.model(
                input_ids=full_inputs.input_ids,
                attention_mask=full_inputs.get("attention_mask"),
            )
        return schwartz_eval.mean_completion_logprob_from_logits(
            outputs.logits[0], prompt_len, answer_ids
        )

    @torch.no_grad()
    def _score_ab_next_token(
        self,
        row: dict,
        hook_infos: Optional[list] = None,
    ) -> dict:
        pos_is_a = schwartz_eval.eval_pos_is_a(row, self.config.random_seed)
        tokens, a_token_id, b_token_id = schwartz_eval.format_ab_eval_tokens(
            row["question"],
            row["positive_answer"],
            row["negative_answer"],
            pos_is_a,
            self.tokenizer,
            self.config.model_name,
        )
        input_ids = torch.tensor([tokens], device=self.config.device)

        if hook_infos:
            with steering_opt.hf_hooks_contextmanager(self.model, hook_infos):
                logits = self.model(input_ids).logits
        else:
            logits = self.model(input_ids).logits

        return schwartz_eval.score_ab_from_logits(
            logits[0, -1, :], a_token_id, b_token_id, pos_is_a
        )

    def evaluate_steering(
        self,
        vectors: Dict[str, torch.Tensor],
        layer: Union[int, List[int]],
    ) -> Dict[str, Any]:
        """
        Evaluate steering vectors on the held-out validation set.

        Metric is selected via ``config.eval_metric``:

        - ``full_logprob``: mean per-token logprob of positive vs negative answer.
        - ``ab_next_token``: CAA-style P(A) vs P(B) on an MCQ prompt ending in ``" ("``.
        """
        eval_metric = self.config.eval_metric
        metric_label = schwartz_eval.eval_metric_label(eval_metric)
        self._log("\n" + "─" * 60)
        self._log(f"  Steering Evaluation ({metric_label})")
        self._log("─" * 60)

        layers = self._normalize_layers(layer)
        alpha = self.config.alpha
        n_eval = self.config.n_eval_samples
        use_ab = eval_metric == schwartz_eval.EVAL_METRIC_AB_NEXT_TOKEN

        records: List[dict] = []
        eval_values = [v for v in self.values if v in vectors]

        if use_ab:
            schwartz_eval.assign_pos_is_a_caa(
                self.val_rows, SCHWARTZ_CIRCUMPLEX_ORDER, self.config.random_seed
            )

        for value in eval_values:
            vec = vectors[value].detach().to(self.config.device)
            scaled_vec = alpha * vec
            hook_fn = steering_opt.make_steering_hook_hf(scaled_vec)
            hook_infos = [(l, hook_fn) for l in layers]

            val_rows = data_utils.get_rows_for_value(self.val_rows, value)
            if not val_rows:
                self._log(f"  {value}: no validation rows – skipping")
                continue

            if n_eval is not None and n_eval < len(val_rows):
                rng = random.Random(self.config.random_seed)
                val_rows = rng.sample(val_rows, n_eval)

            self._log(f"  Evaluating {value} ({len(val_rows)} samples) ...")

            pbar = tqdm(val_rows, desc="Eval steering", position=0, leave=True)
            for row in pbar:
                pbar.set_description(f"Eval: {value}")
                rec = {"value": value}

                if use_ab:
                    ab_base = self._score_ab_next_token(row)
                    ab_steer = self._score_ab_next_token(row, hook_infos)
                    rec.update({
                        "ab_prob_positive_base": ab_base["prob_positive"],
                        "ab_prob_negative_base": ab_base["prob_negative"],
                        "ab_margin_base": ab_base["positive_margin"],
                        "ab_correct_base": ab_base["is_correct"],
                        "ab_prob_positive_steer": ab_steer["prob_positive"],
                        "ab_prob_negative_steer": ab_steer["prob_negative"],
                        "ab_margin_steer": ab_steer["positive_margin"],
                        "ab_correct_steer": ab_steer["is_correct"],
                    })
                else:
                    prompt = schwartz_eval.format_qa_eval_prompt(
                        row["question"],
                        tokenizer=self.tokenizer,
                        model_name=self.config.model_name,
                    )
                    pos = row["positive_answer"]
                    neg = row["negative_answer"]
                    rec.update({
                        "lp_pos_base": self._compute_logprob(prompt, pos),
                        "lp_neg_base": self._compute_logprob(prompt, neg),
                        "lp_pos_steer": self._compute_logprob(prompt, pos, hook_infos),
                        "lp_neg_steer": self._compute_logprob(prompt, neg, hook_infos),
                    })

                records.append(rec)

        if not records:
            self._log("  WARNING: no evaluation records collected!")
            return {}

        eval_payload = schwartz_eval.build_eval_payload(
            eval_metric,
            records,
            self.values,
            extra_fields={
                "alpha": alpha,
                "layer": layer if isinstance(layer, int) else layers,
            },
        )

        per_value = eval_payload["per_value"]
        overall = eval_payload["overall"]
        delta_key = (
            "mean_delta_positive_margin"
            if use_ab
            else "mean_delta_logprob"
        )

        self._log("")
        self._log(
            f"  {'Value':<35} {'Base Acc':>9} {'Steer Acc':>10} "
            f"{'Δ Acc':>7} {'Δ':>9}"
        )
        self._log("  " + "-" * 75)
        for value in self.values:
            if value not in per_value:
                continue
            m = per_value[value]
            self._log(
                f"  {value:<35} {m['accuracy_baseline']:>9.1%} "
                f"{m['accuracy_steered']:>10.1%} "
                f"{m['delta_accuracy']:>+7.1%} "
                f"{m.get(delta_key, 0):>+9.4f}"
            )
        self._log("  " + "-" * 75)
        o = overall
        self._log(
            f"  {'OVERALL':<35} {o['accuracy_baseline']:>9.1%} "
            f"{o['accuracy_steered']:>10.1%} "
            f"{o['delta_accuracy']:>+7.1%} "
            f"{o.get(delta_key, 0):>+9.4f}\n"
        )

        eval_path = os.path.join(
            self.config.output_dir, "steering_eval_metrics.json"
        )
        with open(eval_path, "w") as f:
            json.dump(eval_payload, f, indent=2)
        self._log(f"  Saved evaluation metrics → {eval_path}")

        self._plot_eval_accuracy(per_value, overall, eval_metric)

        return eval_payload

    def _plot_eval_accuracy(
        self,
        per_value: Dict[str, dict],
        overall: dict,
        eval_metric: Optional[str] = None,
    ):
        """Grouped bar chart comparing baseline vs steered accuracy."""
        eval_metric = eval_metric or self.config.eval_metric
        labels = [v for v in self.values if v in per_value]
        if not labels:
            return

        base_accs = [per_value[v]["accuracy_baseline"] for v in labels]
        steer_accs = [per_value[v]["accuracy_steered"] for v in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 6))
        bars1 = ax.bar(x - width / 2, base_accs, width, label="Baseline",
                       color="#90CAF9", edgecolor="#1565C0")
        bars2 = ax.bar(x + width / 2, steer_accs, width, label="Steered",
                       color="#A5D6A7", edgecolor="#2E7D32")

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5,
                   label="Chance (50%)")
        ax.set_ylabel("Accuracy (positive preferred)")
        ax.set_title(
            f"Steering Evaluation — {schwartz_eval.eval_metric_label(eval_metric)}\n"
            f"(α={self.config.alpha}, overall: "
            f"{overall['accuracy_baseline']:.1%} → {overall['accuracy_steered']:.1%})"
        )
        ax.set_xticks(x)
        short_labels = [v.split(":")[-1].strip() if ":" in v else v
                        for v in labels]
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(
            self.config.output_dir, "steering_eval_accuracy.png"
        )
        plt.savefig(out_path, dpi=300)
        plt.close()
        self._log(f"  Saved accuracy plot   → {out_path}")

    # ─── Geometry Analysis ─────────────────────────────────────────────

    @staticmethod
    def _circular_step_distance(i: int, j: int, n: int) -> int:
        """Shortest step distance between positions *i* and *j* on a circle of size *n*."""
        return min(abs(i - j), n - abs(i - j))

    @staticmethod
    def _lower_order_family(value: str) -> str:
        return value.split(":")[0].strip() if ":" in value else value

    @staticmethod
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

    @staticmethod
    def _groups_are_opposite(group_a: str, group_b: str) -> bool:
        opposite_pairs = {
            frozenset({"Openness to Change", "Conservation"}),
            frozenset({"Self-Enhancement", "Self-Transcendence"}),
        }
        return frozenset({group_a, group_b}) in opposite_pairs

    @classmethod
    def _hierarchical_theory_distance(cls, value_a: str, value_b: str) -> tuple[int, str]:
        if cls._lower_order_family(value_a) == cls._lower_order_family(value_b):
            return 1, "same_lower_order"

        groups_a = cls._higher_order_groups_for_value(value_a)
        groups_b = cls._higher_order_groups_for_value(value_b)

        if groups_a & groups_b:
            return 2, "same_higher_order"

        if any(cls._groups_are_opposite(group_a, group_b) for group_a in groups_a for group_b in groups_b):
            return 10, "opposite_higher_order"

        return 5, "no_relation"

    @staticmethod
    def _group_legend_handles():
        """Create legend handles for higher-order groups."""
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

    @classmethod
    def _add_group_legend(cls, ax) -> None:
        """Add group legend to axes."""
        legend = ax.legend(
            handles=cls._group_legend_handles(),
            loc="upper right",
            frameon=True,
            framealpha=0.94,
            facecolor="white",
            edgecolor="lightgray",
            fontsize=PLOT_LEGEND_FONTSIZE,
            borderpad=0.5,
            labelspacing=0.45,
            handletextpad=0.6,
        )
        ax.add_artist(legend)

    @staticmethod
    def _draw_value_marker(ax, x: float, y: float, value: str, radius: float = PLOT_MARKER_RADIUS) -> None:
        """Draw a value marker (circle or split wedge for boundary values)."""
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

    @classmethod
    def _plot_embedding_2d(cls, out_path: str, title: str, coords: np.ndarray):
        """Scatter plot of a 2-D embedding, coloured by Schwartz higher-order group."""
        plt.figure(figsize=EMBEDDING_FIGURE_SIZE)
        for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            group = value_to_group(val)
            color = GROUP_COLORS.get(group, "black")
            plt.scatter(coords[i, 0], coords[i, 1], c=color, s=PLOT_SCATTER_SIZE, edgecolors="white", linewidths=1.2)
            plt.annotate(
                cls._short_value_label(val),
                (coords[i, 0], coords[i, 1]),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=PLOT_LABEL_FONTSIZE,
                fontweight="semibold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
            )

        plt.legend(handles=cls._group_legend_handles(), loc="best", fontsize=PLOT_LEGEND_FONTSIZE)
        plt.title(title, fontsize=PLOT_TITLE_FONTSIZE)
        plt.xticks(fontsize=PLOT_AXIS_FONTSIZE)
        plt.yticks(fontsize=PLOT_AXIS_FONTSIZE)
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

        # 4a. Original empirical heatmap (fixed range [-1, 1])
        plt.figure(figsize=HEATMAP_FIGURE_SIZE)
        sns.heatmap(
            empirical_sim,
            xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
            yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
            cmap="coolwarm", vmin=-1, vmax=1,
        )
        plt.title("Empirical Cosine Similarities", fontsize=PLOT_TITLE_FONTSIZE)
        plt.xticks(fontsize=PLOT_AXIS_FONTSIZE, rotation=45, ha="right")
        plt.yticks(fontsize=PLOT_AXIS_FONTSIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap.png"), dpi=300)
        plt.close()

        # 4b. Contrast-enhanced: auto-scale to off-diagonal range
        off_diag_mask = ~np.eye(num_values, dtype=bool)
        off_diag_vals = empirical_sim[off_diag_mask]
        vmin_auto = off_diag_vals.min()
        vmax_auto = off_diag_vals.max()

        plt.figure(figsize=HEATMAP_FIGURE_SIZE)
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
            fontsize=PLOT_TITLE_FONTSIZE,
        )
        plt.xticks(fontsize=PLOT_AXIS_FONTSIZE, rotation=45, ha="right")
        plt.yticks(fontsize=PLOT_AXIS_FONTSIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap_enhanced.png"), dpi=300)
        plt.close()

        # 4c. Rank-transformed heatmap for maximum contrast
        rank_matrix = np.zeros_like(empirical_sim)
        rank_vals = rankdata(off_diag_vals)  # rank the off-diagonal values
        rank_matrix[off_diag_mask] = rank_vals / rank_vals.max()  # normalize to [0,1]
        np.fill_diagonal(rank_matrix, 1.0)  # diagonal = max similarity

        plt.figure(figsize=HEATMAP_FIGURE_SIZE)
        sns.heatmap(
            rank_matrix,
            xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
            yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
            cmap="coolwarm",
            vmin=0, vmax=1,
        )
        plt.title("Empirical Similarity (rank-transformed, 0=least similar, 1=most)", fontsize=PLOT_TITLE_FONTSIZE)
        plt.xticks(fontsize=PLOT_AXIS_FONTSIZE, rotation=45, ha="right")
        plt.yticks(fontsize=PLOT_AXIS_FONTSIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap_ranked.png"), dpi=300)
        plt.close()

        plt.figure(figsize=HEATMAP_FIGURE_SIZE)
        sns.heatmap(
            theoretical_sim,
            xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
            yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER,
            cmap="coolwarm", vmin=-1, vmax=1,
        )
        plt.title("Theoretical Relationships", fontsize=PLOT_TITLE_FONTSIZE)
        plt.xticks(fontsize=PLOT_AXIS_FONTSIZE, rotation=45, ha="right")
        plt.yticks(fontsize=PLOT_AXIS_FONTSIZE)
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

                step = self._circular_step_distance(i, j, num_values)
                circular_step_flat.append(step)
                if step == 1:
                    neighbor_empirical.append(empirical_sim[i, j])
                if step == num_values // 2:
                    opposite_empirical.append(empirical_sim[i, j])

                hierarchical_distance, relation_bucket = self._hierarchical_theory_distance(
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
        plt.figure(figsize=MDS_FIGURE_SIZE)
        ax = plt.gca()
        # Draw theoretical unit circle
        ax.add_patch(Circle((0, 0), 1, color="lightgray", fill=False, linestyle="--"))

        for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            # Theoretical position on circle
            tx, ty = X_circle[i]
            ax.plot(tx, ty, "x", color="gray", markersize=9)

            # Empirical position from MDS
            ex, ey = X_mds_aligned[i]
            group = value_to_group(val)
            color = GROUP_COLORS.get(group, "black")

            # Draw value marker at empirical position
            self._draw_value_marker(ax, ex, ey, val)
            # Draw line connecting theoretical to empirical
            ax.plot([tx, ex], [ty, ey], color="gray", alpha=0.3, linestyle=":")

            # Annotate with short label
            ax.annotate(
                self._short_value_label(val),
                (ex, ey),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=PLOT_LABEL_FONTSIZE,
                fontweight="semibold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.8),
            )

        self._add_group_legend(ax)
        plt.title("2D MDS Aligned to Theoretical Circumplex", fontsize=PLOT_TITLE_FONTSIZE)
        plt.axis("equal")
        # Set limits clearly showing unit circle
        scale = np.max(np.abs(X_mds_aligned))
        lim = max(1.2, scale * 1.2)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.grid(alpha=0.2)
        plt.xticks(fontsize=PLOT_AXIS_FONTSIZE)
        plt.yticks(fontsize=PLOT_AXIS_FONTSIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mds_circumplex.png"), dpi=300)
        plt.close()

        # ── 9. Theory vs empirical scatter ────────────────────────────
        plt.figure(figsize=SCATTER_FIGURE_SIZE)
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
        plt.figure(figsize=DIFFERENCE_HEATMAP_SIZE)
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

        Directory layout::

            {output_dir}/{model_name}/lr_{lr}-alpha_{alpha}-layer_{layer}-max_iter_{max_iter}-train_ratio_{ratio}-eval_{metric}/vectors/
                manifest.json
                {value_name}.pt
                {value_name}.json

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
                "lr": self.config.lr,
                "alpha": self.config.alpha,
                "max_iters": self.config.max_iters,
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

    def try_load_cached_vectors(
        self,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Attempt to load previously saved vectors for the current config.

        Checks the expected output directory (derived from model name,
        lr, alpha, and layer) for a ``vectors/manifest.json``.  If found,
        loads all vectors and returns them.  Otherwise returns ``None``.
        """
        vectors_dir = os.path.join(self.config.output_dir, "vectors")
        manifest_path = os.path.join(vectors_dir, "manifest.json")

        if not os.path.isfile(manifest_path):
            return None

        self._log(f"  Found cached vectors at {vectors_dir}/")

        try:
            loaded = self.load_vectors(vectors_dir)
        except Exception as e:
            self._log(f"  WARNING: failed to load cached vectors: {e}")
            return None

        vectors: Dict[str, torch.Tensor] = {}
        for value, (vec, _meta) in loaded.items():
            vectors[value] = vec

        self._log(f"  Loaded {len(vectors)} cached vectors — skipping training")
        return vectors

    # ─── Full Pipeline ───────────────────────────────────────────────────

    def run(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, dict], int]:
        """
        Run the full pipeline end-to-end.

        Returns:
            (vectors, metrics, best_layer)
        """

        if torch.cuda.is_available():
            self._log(f"Total GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self._log(f"\n--- GPU {i}: {props.name} ---")
                self._log(f"  Compute Capability: {props.major}.{props.minor}")
                self._log(f"  Total Memory: {round(props.total_memory / 1024**3, 2)} GB")
                self._log(f"  Multi-processors: {props.multi_processor_count}")
        else:
            self._log("No CUDA-capable GPUs detected.")

        self._log("=" * 60)
        self._log("  Value-Steering Optimization Pipeline")
        self._log("=" * 60)
        self._log("")

        output_base = self.config.output_dir

        # 1. Load model
        self.load_model()

        # 2. Prepare data
        self.prepare_data()

        # 3. Layer selection
        layer_sweep_payload: Optional[Dict[str, Any]] = None
        if self.config.layer_sweep_enabled:
            best_layer, layer_sweep_payload = self.select_layer()
        else:
            candidates = self._get_sweep_candidates()
            # If --layer was given, candidates is [that_layer]; otherwise pick middle
            best_layer = candidates[0] if len(candidates) == 1 else candidates[len(candidates) // 2]
            self._log(f"Layer sweep disabled. Using layer {best_layer}\n")

        self.config.output_dir = os.path.join(
            output_base, self._run_dir_name(best_layer)
        )
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        if layer_sweep_payload is not None:
            sweep_path = os.path.join(self.config.output_dir, "layer_sweep.json")
            with open(sweep_path, "w") as f:
                json.dump(layer_sweep_payload, f, indent=2)

        # 4. Try loading cached vectors; train if unavailable
        cached = self.try_load_cached_vectors()
        if cached is not None:
            vectors = cached
        else:
            vectors = self.train_vectors(best_layer)
            # Save immediately so next run can reuse them
            if self.config.save_vectors:
                self.save_vectors(vectors, best_layer)

        # 5. Steering evaluation (log-likelihood on validation set)
        eval_metrics = self.evaluate_steering(vectors, best_layer)

        # 6. Geometry analysis
        metrics = self.analyze_geometry(vectors)

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
            print(msg,flush=True)

    def _run_dir_name(self, layer: int) -> str:
        """
        Hierarchical run directory::

            {model_short_name}/lr_{lr}-alpha_{alpha}-layer_{layer}-max_iter_{max_iter}-train_ratio_{ratio}-eval_{metric}

        Example::

            Qwen3.5-9B-Base/lr_0p01-alpha_40p0-layer_22-max_iter_30-train_ratio_0p005-eval_full_logprob
        """
        model_short = self.config.model_name.split("/")[-1].replace(" ", "_")
        lr_slug = str(self.config.lr).replace(".", "p").replace("-", "neg")
        alpha_slug = str(self.config.alpha).replace(".", "p").replace("-", "neg")
        eval_slug = self.config.eval_metric.replace("_", "-")
        run_name = (
            f"lr_{lr_slug}-alpha_{alpha_slug}-layer_{layer}-"
            f"max_iter_{self.config.max_iters}-"
            f"train_ratio_{str(self.config.train_ratio).replace('.', 'p')}-"
            f"eval_{eval_slug}"
        )
        return os.path.join(model_short, run_name)

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
