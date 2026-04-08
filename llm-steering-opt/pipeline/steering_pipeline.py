"""
Main steering pipeline: layer selection, per-value vector training, and evaluation.
"""

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

# Ensure steering_opt is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import steering_opt

from .config import SteeringConfig
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

    # ─── Layer Selection ─────────────────────────────────────────────────

    def _get_sweep_candidates(self) -> List[int]:
        """Determine which layers to sweep."""
        if self.config.layer_sweep_candidates is not None:
            return self.config.layer_sweep_candidates

        n_layers = self.model.config.num_hidden_layers
        n_cand = min(self.config.layer_sweep_n_candidates, n_layers)

        # Sample layers from the 15%-85% depth range (skip very early/late)
        start = max(1, int(n_layers * 0.15))
        end = int(n_layers * 0.85)
        step = max(1, (end - start) // (n_cand - 1)) if n_cand > 1 else 1
        candidates = list(range(start, end + 1, step))[:n_cand]

        return candidates

    def select_layer(self) -> int:
        """
        Layer sweep: optimize a quick vector at each candidate layer using
        a small subset, then pick the layer with the best validation metric.

        Returns:
            Best layer index.
        """
        candidates = self._get_sweep_candidates()
        self._log(f"Layer sweep over candidates: {candidates}")

        results: Dict[int, float] = {}

        pbar_layers = tqdm(candidates, desc="Layer Sweep", leave=True)
        for layer in pbar_layers:
            pbar_layers.set_description(f"Sweeping layer {layer}")
            layer_scores = []

            # Use a subset of values for speed (max 5 values)
            sweep_values = self.values[:5] if len(self.values) > 5 else self.values

            for value in sweep_values:
                train_value_rows = data_utils.get_rows_for_value(
                    self.train_rows, value
                )
                val_value_rows = data_utils.get_rows_for_value(
                    self.val_rows, value
                )

                if not train_value_rows or not val_value_rows:
                    continue

                # Create a small training set
                datapoints = data_utils.create_datapoints(
                    train_value_rows,
                    tokenizer=self.tokenizer,
                    use_chat_template=self.config.use_chat_template,
                    prompt_template=self.config.prompt_template,
                    n_samples=self.config.layer_sweep_n_samples,
                    seed=self.config.random_seed,
                )

                if not datapoints:
                    continue

                # Quick optimization (fewer iters for speed)
                try:
                    vector, info = steering_opt.optimize_vector(
                        self.model,
                        datapoints,
                        layer,
                        tokenizer=self.tokenizer,
                        use_transformer_lens=False,
                        lr=self.config.lr,
                        max_iters=min(15, self.config.max_iters),
                        max_norm=self.config.max_norm,
                        starting_norm=self.config.starting_norm,
                        coldness=self.config.coldness,
                        return_info=True,
                    )
                except Exception as e:
                    self._log(f"    Warning: failed on value '{value}': {e}")
                    continue

                # Evaluate on a few val examples
                val_datapoints = data_utils.create_datapoints(
                    val_value_rows,
                    tokenizer=self.tokenizer,
                    use_chat_template=self.config.use_chat_template,
                    prompt_template=self.config.prompt_template,
                    n_samples=3,  # Just a few for speed
                    seed=self.config.random_seed,
                )

                score = self._compute_delta_logprob(vector, layer, val_datapoints)
                layer_scores.append(score)

            avg_score = np.mean(layer_scores) if layer_scores else float("-inf")
            results[layer] = avg_score
            self._log(f"    Layer {layer}: avg Δ log P = {avg_score:.4f}")

        if not results:
            # All layers failed — fall back to middle candidate
            best_layer = candidates[len(candidates) // 2]
            self._log(f"\n  ⚠ All layers failed. Falling back to layer {best_layer}\n")
        else:
            best_layer = max(results, key=results.get)
            self._log(f"\n  ✓ Best layer: {best_layer} (Δ log P = {results[best_layer]:.4f})\n")

        # Save sweep results
        sweep_path = os.path.join(self.config.output_dir, "layer_sweep.json")
        with open(sweep_path, "w") as f:
            json.dump(
                {"candidates": candidates, "scores": {str(k): v for k, v in results.items()}, "best_layer": best_layer},
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

    # ─── Evaluation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_delta_logprob(
        self,
        vector: torch.Tensor,
        layer: int,
        datapoints: List[steering_opt.TrainingDatapoint],
    ) -> float:
        """
        Compute mean Δ log P across datapoints.

        For each datapoint, computes:
            Δ = [log P(dst | steered) - log P(dst | unsteered)]
              + [log P(src | unsteered) - log P(src | steered)]

        Positive Δ means steering is working (promoting dst, suppressing src).
        """
        if not datapoints:
            return 0.0

        alpha = self.config.alpha
        vector = vector.detach()
        deltas = []

        for dp in datapoints:
            delta = 0.0

            # Evaluate dst_completions (should be promoted)
            for comp in dp.dst_completions:
                # Unsteered
                unsteered_lp = steering_opt.get_completion_logprob_hf(
                    self.model, dp.prompt, comp, self.tokenizer
                )
                # Steered
                hook = (layer, steering_opt.make_steering_hook_hf(alpha * vector))
                with steering_opt.hf_hooks_contextmanager(self.model, [hook]):
                    steered_lp = steering_opt.get_completion_logprob_hf(
                        self.model, dp.prompt, comp, self.tokenizer
                    )
                delta += (steered_lp - unsteered_lp).item()

            # Evaluate src_completions (should be suppressed)
            for comp in dp.src_completions:
                unsteered_lp = steering_opt.get_completion_logprob_hf(
                    self.model, dp.prompt, comp, self.tokenizer
                )
                hook = (layer, steering_opt.make_steering_hook_hf(alpha * vector))
                with steering_opt.hf_hooks_contextmanager(self.model, [hook]):
                    steered_lp = steering_opt.get_completion_logprob_hf(
                        self.model, dp.prompt, comp, self.tokenizer
                    )
                # For src, we want steered_lp < unsteered_lp,
                # so positive delta = unsteered - steered
                delta += (unsteered_lp - steered_lp).item()

            deltas.append(delta)

        return float(np.mean(deltas))

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

    def evaluate(
        self, vectors: Dict[str, torch.Tensor], layer: int
    ) -> Dict[str, dict]:
        """
        Evaluate all steering vectors on the validation set.

        Returns:
            Dict mapping value name -> {delta_logprob, accuracy, n_examples}.
        """
        self._log("Evaluating steering vectors on validation set")
        metrics: Dict[str, dict] = {}

        all_deltas = []
        all_accs = []

        pbar_eval = tqdm(self.values, desc="Evaluating", leave=True)
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

            self._log(f"  Evaluating: {value} ({len(val_datapoints)} val examples)...")

            delta_lp = self._compute_delta_logprob(vector, layer, val_datapoints)
            accuracy = self._compute_accuracy(vector, layer, val_datapoints)

            metrics[value] = {
                "delta_logprob": round(delta_lp, 4),
                "accuracy": round(accuracy, 4),
                "n_val_examples": len(val_datapoints),
            }
            all_deltas.append(delta_lp)
            all_accs.append(accuracy)

            self._log(
                f"    Δ log P = {delta_lp:+.4f} | "
                f"Accuracy = {accuracy:.1%} | "
                f"n = {len(val_datapoints)}"
            )

        # Overall metrics
        if all_deltas:
            metrics["__overall__"] = {
                "mean_delta_logprob": round(float(np.mean(all_deltas)), 4),
                "mean_accuracy": round(float(np.mean(all_accs)), 4),
                "n_values": len(all_deltas),
            }
            self._log(
                f"\n  Overall: Δ log P = {np.mean(all_deltas):+.4f} | "
                f"Accuracy = {np.mean(all_accs):.1%} "
                f"(across {len(all_deltas)} values)\n"
            )

        # Save metrics
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

