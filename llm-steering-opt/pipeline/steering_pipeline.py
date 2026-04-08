"""
Main steering pipeline: layer selection, per-value vector training, and evaluation.

Layer selection uses mean normalized L2 separation of activations.
Evaluation uses Spearman correlation between empirical steering-vector cosine
similarities and the theoretical Schwartz value relationship matrix.
"""

import dataclasses
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# Ensure steering_opt is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import steering_opt

from .config import SteeringConfig, SCHWARTZ_CIRCUMPLEX_ORDER
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

    def train_vectors(self, layer: int) -> Dict[str, torch.Tensor]:
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

        pbar_train = tqdm(self.values, desc="Training Vectors", leave=True)
        for value in pbar_train:
            pbar_train.set_description(f"Training: {value}")
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
            try:
                vector, info = steering_opt.optimize_vector(
                    self.model,
                    datapoints,
                    layer,
                    tokenizer=self.tokenizer,
                    use_transformer_lens=False,
                    lr=self.config.lr,
                    max_iters=self.config.max_iters,
                    max_norm=self.config.max_norm,
                    starting_norm=self.config.starting_norm,
                    coldness=self.config.coldness,
                    target_loss=self.config.target_loss,
                    return_info=True,
                )
            except Exception as e:
                self._log(f"    ✗ Failed: {e}")
                continue

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

    # ─── Evaluation (Spearman Correlation) ──────────────────────────────

    @torch.no_grad()
    def _compute_accuracy(
        self,
        vector: torch.Tensor,
        layer: int,
        datapoints: List[steering_opt.TrainingDatapoint],
    ) -> float:
        """
        Compute accuracy: fraction of datapoints where the steered model
        assigns higher probability to dst_completion than src_completion.
        """
        if not datapoints:
            return 0.0

        alpha = self.config.alpha
        vector = vector.detach()
        correct = 0

        for dp in datapoints:
            if not dp.dst_completions or not dp.src_completions:
                continue

            hook = (layer, steering_opt.make_steering_hook_hf(alpha * vector))
            with steering_opt.hf_hooks_contextmanager(self.model, [hook]):
                dst_lp = steering_opt.get_completion_logprob_hf(
                    self.model, dp.prompt, dp.dst_completions[0], self.tokenizer
                )
                src_lp = steering_opt.get_completion_logprob_hf(
                    self.model, dp.prompt, dp.src_completions[0], self.tokenizer
                )

            if dst_lp > src_lp:
                correct += 1

        return correct / len(datapoints)

    def _compute_spearman(
        self, vectors: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute Spearman (and Pearson) correlation between the empirical
        pairwise cosine similarities of unit-normed steering vectors and
        the theoretical Schwartz relationship matrix.

        Returns dict with spearman_rho, spearman_p_value, pearson_r, etc.
        """
        ordered_values = [v for v in SCHWARTZ_CIRCUMPLEX_ORDER if v in vectors]
        num_values = len(ordered_values)

        unit_vectors: Dict[str, torch.Tensor] = {}
        for val in ordered_values:
            vec = vectors[val].detach().cpu().float()
            norm = vec.norm()
            unit_vectors[val] = vec / norm if norm > 0 else vec

        empirical_sim = np.zeros((num_values, num_values))
        for i, v1 in enumerate(ordered_values):
            for j, v2 in enumerate(ordered_values):
                empirical_sim[i, j] = F.cosine_similarity(
                    unit_vectors[v1], unit_vectors[v2], dim=0
                ).item()

        with open(self.config.relations_path, "r") as f:
            rel_data = json.load(f)
        rel_matrix = rel_data["basic_value_relationship_matrix"]

        theoretical_sim = np.zeros((num_values, num_values))
        for i, v1 in enumerate(ordered_values):
            for j, v2 in enumerate(ordered_values):
                if v1 in rel_matrix and v2 in rel_matrix[v1]:
                    theoretical_sim[i, j] = rel_matrix[v1][v2]

        triu_indices = np.triu_indices(num_values, k=1)
        emp_flat = empirical_sim[triu_indices]
        theo_flat = theoretical_sim[triu_indices]

        rho, p_val = spearmanr(emp_flat, theo_flat)
        pearson_r, pearson_p = pearsonr(emp_flat, theo_flat)

        return {
            "spearman_rho": float(rho),
            "spearman_p_value": float(p_val),
            "pearson_r": float(pearson_r),
            "pearson_p_value": float(pearson_p),
            "num_value_pairs": int(len(emp_flat)),
            "num_values_used": num_values,
            "empirical_similarity_matrix": empirical_sim.tolist(),
            "theoretical_similarity_matrix": theoretical_sim.tolist(),
            "ordered_values": ordered_values,
        }

    def evaluate(
        self, vectors: Dict[str, torch.Tensor], layer: int
    ) -> Dict[str, dict]:
        """
        Evaluate steering vectors using:
          1. Spearman correlation between empirical cosine similarities of
             steering vectors and Schwartz theoretical relationship matrix.
          2. Per-value accuracy on the held-out validation set.

        Returns:
            Dict with per-value accuracy, overall accuracy, and Spearman metrics.
        """
        self._log("Evaluating steering vectors")
        metrics: Dict[str, dict] = {}

        # --- Spearman correlation (main metric) ---
        self._log("  Computing Spearman correlation with Schwartz theory...")
        spearman_metrics = self._compute_spearman(vectors)
        metrics["__spearman__"] = {
            "spearman_rho": spearman_metrics["spearman_rho"],
            "spearman_p_value": spearman_metrics["spearman_p_value"],
            "pearson_r": spearman_metrics["pearson_r"],
            "pearson_p_value": spearman_metrics["pearson_p_value"],
            "num_value_pairs": spearman_metrics["num_value_pairs"],
            "num_values_used": spearman_metrics["num_values_used"],
        }
        self._log(
            f"    Spearman rho = {spearman_metrics['spearman_rho']:.4f} "
            f"(p = {spearman_metrics['spearman_p_value']:.4g})"
        )
        self._log(
            f"    Pearson r    = {spearman_metrics['pearson_r']:.4f} "
            f"(p = {spearman_metrics['pearson_p_value']:.4g})"
        )

        # Save detailed Spearman report separately
        spearman_path = os.path.join(self.config.output_dir, "spearman_report.json")
        with open(spearman_path, "w") as f:
            json.dump(spearman_metrics, f, indent=2)

        # --- Per-value accuracy ---
        self._log("\n  Computing per-value steered accuracy...")
        all_accs = []

        pbar_eval = tqdm(self.values, desc="Evaluating accuracy", leave=True)
        for value in pbar_eval:
            if value not in vectors:
                continue

            pbar_eval.set_description(f"Eval: {value}")
            vector = vectors[value]
            val_value_rows = data_utils.get_rows_for_value(self.val_rows, value)

            if not val_value_rows:
                self._log(f"  {value}: no validation data, skipping")
                continue

            val_datapoints = data_utils.create_datapoints(
                val_value_rows,
                tokenizer=self.tokenizer,
                use_chat_template=self.config.use_chat_template,
                prompt_template=self.config.prompt_template,
            )

            accuracy = self._compute_accuracy(vector, layer, val_datapoints)

            metrics[value] = {
                "accuracy": round(accuracy, 4),
                "n_val_examples": len(val_datapoints),
            }
            all_accs.append(accuracy)

            self._log(
                f"    {value}: Accuracy = {accuracy:.1%} | n = {len(val_datapoints)}"
            )

        if all_accs:
            metrics["__overall__"] = {
                "mean_accuracy": round(float(np.mean(all_accs)), 4),
                "n_values": len(all_accs),
                "spearman_rho": spearman_metrics["spearman_rho"],
                "spearman_p_value": spearman_metrics["spearman_p_value"],
            }
            self._log(
                f"\n  Overall: Accuracy = {np.mean(all_accs):.1%} "
                f"(across {len(all_accs)} values)"
            )
            self._log(
                f"  Spearman rho = {spearman_metrics['spearman_rho']:.4f} "
                f"(p = {spearman_metrics['spearman_p_value']:.4g})\n"
            )

        metrics_path = os.path.join(self.config.output_dir, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

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

        # 5. Evaluate
        metrics = self.evaluate(vectors, best_layer)

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

