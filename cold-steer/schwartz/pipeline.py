"""End-to-end COLD-Steer Schwartz value-steering pipeline.

Supports ``cold_fd`` (LossFDSteerer) and ``cold_kernel`` (KernelLossSteerer).
Select via ``SchwartzColdConfig.method`` or ``--method`` on the CLI.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_THIS_DIR = os.path.dirname(__file__)
_COLD_STEER_ROOT = os.path.dirname(_THIS_DIR)
if _COLD_STEER_ROOT not in sys.path:
    sys.path.insert(0, _COLD_STEER_ROOT)

from src.llm import SteerableLLM  # noqa: E402

from . import data_utils
from . import evaluate as eval_mod
from . import geometry as geometry_mod
from . import layer_selection
from . import method_adapters
from .config import SCHWARTZ_CIRCUMPLEX_ORDER, SchwartzColdConfig
from .schwartz_dataset import SchwartzValueDataset


class SchwartzColdPipeline:
    """End-to-end COLD-Steer pipeline for the Schwartz benchmark."""

    def __init__(self, config: SchwartzColdConfig) -> None:
        self.config = config
        self.steerable_llm: Optional[SteerableLLM] = None
        self.tokenizer = None
        self.train_rows: List[dict] = []
        self.val_rows: List[dict] = []
        self.values: List[str] = []
        self._output_base = config.output_dir

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg, flush=True)

    # ── Model ────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        self._log(f"Loading model via SteerableLLM: {self.config.model_name}")
        self._log(f"  dtype={self.config.torch_dtype}")

        original_kwargs = {}
        # cold-steer's SteerableLLM forces device_map='balanced' and uses
        # default float32. We rely on it for module placement, but cast the
        # underlying model to the requested dtype after load.
        self.steerable_llm = SteerableLLM(
            model_name=self.config.model_name,
            steering_layer_indices=[1],  # placeholder, fixed in select_layer step
        )
        dtype = self.config.get_dtype()
        if dtype != torch.float32:
            try:
                self.steerable_llm.model = self.steerable_llm.model.to(dtype)
                # Refresh cached params dict (SteerableLLM caches them in __init__)
                self.steerable_llm.params = {
                    k: v.detach() for k, v in self.steerable_llm.model.named_parameters()
                }
            except Exception as e:
                self._log(f"  WARNING: could not cast model to {dtype}: {e}")
        self.tokenizer = self.steerable_llm.tokenizer

        n_layers = self.steerable_llm.model.config.num_hidden_layers
        d_model = self.steerable_llm.model.config.hidden_size
        self._log(f"  Loaded: {n_layers} layers, d_model={d_model}")

    # ── Data ─────────────────────────────────────────────────────────────

    def prepare_data(self) -> None:
        self._log(f"Loading dataset: {self.config.dataset_path}")
        all_rows = data_utils.load_dataset(self.config.dataset_path)
        self.values = data_utils.get_unique_values(all_rows)
        self._log(f"  Total rows: {len(all_rows)}, unique values: {len(self.values)}")
        self._log(f"  n_training_samples (per value): {self.config.n_training_samples}")
        self.train_rows, self.val_rows = data_utils.split_by_n_train(
            all_rows,
            n_train=self.config.n_training_samples,
            seed=self.config.random_seed,
        )
        if self.config.verbose:
            data_utils.print_split_summary(self.train_rows, self.val_rows, self.values)

    # ── Layer Selection ──────────────────────────────────────────────────

    def select_layer(self) -> Tuple[int, Optional[Dict[str, Any]]]:
        assert self.steerable_llm is not None
        if not self.config.layer_sweep_enabled:
            cand = self.config.layer_sweep_candidates
            if cand is not None and len(cand) > 0:
                if len(cand) == 1:
                    self._log(f"Layer sweep disabled. Using fixed layer {cand[0]}")
                    return int(cand[0]), None
                best = cand[len(cand) // 2]
                self._log(f"Layer sweep disabled. Using middle of {cand} → {best}")
                return int(best), None
            n_layers = self.steerable_llm.model.config.num_hidden_layers
            best = max(1, int(n_layers * 0.6))
            self._log(f"Layer sweep disabled and no candidates. Using ~60% depth → {best}")
            return best, None

        return layer_selection.select_layer(
            steerable_llm=self.steerable_llm,
            train_rows=self.train_rows,
            values=self.values,
            candidates=self.config.layer_sweep_candidates,
            n_samples_per_value=self.config.layer_sweep_n_samples,
            seed=self.config.random_seed,
            use_chat_template=self.config.use_chat_template,
            prompt_template=self.config.prompt_template,
            verbose=self.config.verbose,
        )

    # ── Training (one steerer per value) ─────────────────────────────────

    def train_vectors(
        self, layer_idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, method_adapters.Steerer]]:
        """Train one steerer per Schwartz value at the given layer."""
        assert self.steerable_llm is not None
        self._log(f"\nTraining {self.config.method} steering at layer {layer_idx}")
        self._log(
            f"  eta={self.config.eta} training_mode={self.config.training_mode} "
            f"steer_masking={self.config.steer_masking} "
            f"n_training_samples={self.config.n_training_samples}"
        )
        if self.config.method == "cold_fd":
            self._log(f"  epsilon={self.config.epsilon}")
        else:
            self._log(f"  kernel={self.config.kernel}")

        # Re-bind the LLM's steering layer to our chosen layer
        method_adapters.set_steering_layers(self.steerable_llm, [layer_idx])

        vectors: Dict[str, torch.Tensor] = {}
        train_info: Dict[str, Any] = {}
        steerers_by_value: Dict[str, method_adapters.Steerer] = {}

        for value in self.values:
            value_rows = data_utils.get_rows_for_value(self.train_rows, value)
            if not value_rows:
                self._log(f"  {value}: no training rows, skipping")
                continue

            dataset = SchwartzValueDataset(
                rows=value_rows,
                tokenizer=self.tokenizer,
                device=self.config.device,
                use_chat_template=self.config.use_chat_template,
                prompt_template=self.config.prompt_template,
            )
            if len(dataset) == 0:
                self._log(f"  {value}: empty dataset after sampling, skipping")
                continue

            self._log(f"  Training {value} ({len(dataset)} samples)...")
            steerer = method_adapters.make_steerer(
                method=self.config.method,
                steerable_llm=self.steerable_llm,
                epsilon=self.config.epsilon,
                eta=self.config.eta,
                training=self.config.training_mode,
                steer_masking=self.config.steer_masking,
                gen_masking=self.config.gen_masking,
                training_batch_size=self.config.training_batch_size,
                kernel=self.config.kernel,
                log_dir=self.config.output_dir,
            )
            t0 = time.time()
            steerer.train(dataset)
            elapsed = time.time() - t0

            # Representative direction
            n_samples = len(dataset)
            vec = steerer.extract_representative_vector(dataset, layer_idx=layer_idx)
            vectors[value] = vec.detach().clone()
            method_adapters.offload_steerer_state(steerer)
            steerers_by_value[value] = steerer
            del dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            norm = float(vec.norm().item())
            train_info[value] = {
                "n_samples": n_samples,
                "rep_vec_norm": round(norm, 6),
                "time_sec": round(elapsed, 2),
            }
            self._log(f"    ✓ Done in {elapsed:.1f}s | norm={norm:.4f}")

        with open(os.path.join(self.config.output_dir, "training_info.json"), "w") as f:
            json.dump(train_info, f, indent=2)
        self._log(f"\n  Trained {len(vectors)}/{len(self.values)} {self.config.method} steerers\n")
        return vectors, train_info, steerers_by_value

    # ── Persistence ──────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        return (
            name.lower()
            .replace(": ", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

    def save_vectors(
        self,
        vectors: Dict[str, torch.Tensor],
        layer: int,
        steerers_by_value: Optional[Dict[str, method_adapters.Steerer]] = None,
    ) -> None:
        vec_dir = os.path.join(self.config.output_dir, "vectors")
        os.makedirs(vec_dir, exist_ok=True)
        manifest = {}
        for value, vector in vectors.items():
            safe = self._sanitize_filename(value)
            vec_path = os.path.join(vec_dir, f"{safe}.pt")
            torch.save(vector.detach().cpu(), vec_path)
            steerer_file = f"{safe}_steerer.pt"
            if steerers_by_value and value in steerers_by_value:
                method_adapters.save_steerer_checkpoint(
                    steerers_by_value[value],
                    os.path.join(vec_dir, steerer_file),
                )
            meta = {
                "value": value,
                "layer": layer,
                "norm": float(vector.norm().item()),
                "d_model": int(vector.shape[0]),
                "model_name": self.config.model_name,
                "method": self.config.method,
                "eta": self.config.eta,
                "training_mode": self.config.training_mode,
                "n_training_samples": self.config.n_training_samples,
                "eval_metric": self.config.eval_metric,
            }
            if self.config.method == "cold_fd":
                meta["epsilon"] = self.config.epsilon
            else:
                meta["kernel"] = self.config.kernel
            with open(os.path.join(vec_dir, f"{safe}.json"), "w") as f:
                json.dump(meta, f, indent=2)
            entry = {
                "vector_file": f"{safe}.pt",
                "metadata_file": f"{safe}.json",
                "layer": layer,
                "norm": round(float(vector.norm().item()), 4),
            }
            if steerers_by_value and value in steerers_by_value:
                entry["steerer_file"] = steerer_file
            manifest[value] = entry
        with open(os.path.join(vec_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        self._log(f"Saved {len(vectors)} vectors to {vec_dir}/")

    def _cache_meta_matches(self, meta: dict, layer: int) -> bool:
        """True if saved metadata matches the current run configuration."""
        checks = [
            ("model_name", self.config.model_name),
            ("method", self.config.method),
            ("layer", layer),
            ("eta", self.config.eta),
            ("training_mode", self.config.training_mode),
            ("n_training_samples", self.config.n_training_samples),
            ("eval_metric", self.config.eval_metric),
        ]
        for key, expected in checks:
            if key not in meta:
                if key == "eval_metric":
                    continue
                return False
            if meta.get(key) != expected:
                return False
        if self.config.method == "cold_fd":
            return meta.get("epsilon") == self.config.epsilon
        return meta.get("kernel") == self.config.kernel

    def try_load_cached_training(
        self, layer_idx: int
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, method_adapters.Steerer]]]:
        """Load vectors (+ steerers when checkpoints exist) and skip training."""
        assert self.steerable_llm is not None
        vec_dir = os.path.join(self.config.output_dir, "vectors")
        manifest_path = os.path.join(vec_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            return None

        with open(manifest_path) as f:
            manifest = json.load(f)

        vectors: Dict[str, torch.Tensor] = {}
        steerers_by_value: Dict[str, method_adapters.Steerer] = {}
        vector_only_values: List[str] = []

        for value, info in manifest.items():
            meta_path = os.path.join(vec_dir, info["metadata_file"])
            if not os.path.isfile(meta_path):
                self._log(f"  Cache miss: missing metadata for {value}")
                return None
            with open(meta_path) as f:
                meta = json.load(f)
            if not self._cache_meta_matches(meta, layer_idx):
                self._log(f"  Cache miss: config mismatch for {value}")
                return None

            vec_path = os.path.join(vec_dir, info["vector_file"])
            if not os.path.isfile(vec_path):
                self._log(f"  Cache miss: missing vector for {value}")
                return None
            vectors[value] = torch.load(vec_path, map_location="cpu", weights_only=True)

            steerer_path = os.path.join(
                vec_dir, info.get("steerer_file", "")
            )
            if info.get("steerer_file") and os.path.isfile(steerer_path):
                steerer = method_adapters.make_steerer(
                    method=self.config.method,
                    steerable_llm=self.steerable_llm,
                    epsilon=self.config.epsilon,
                    eta=self.config.eta,
                    training=self.config.training_mode,
                    steer_masking=self.config.steer_masking,
                    gen_masking=self.config.gen_masking,
                    training_batch_size=self.config.training_batch_size,
                    kernel=self.config.kernel,
                    log_dir=self.config.output_dir,
                )
                method_adapters.load_steerer_checkpoint(steerer, steerer_path)
                method_adapters.offload_steerer_state(steerer)
                steerers_by_value[value] = steerer
            else:
                vector_only_values.append(value)
                steerers_by_value[value] = method_adapters.VectorSteerProxy(
                    vectors[value], self.config.eta, layer_idx
                )

        if not vectors:
            return None

        train_info: Dict[str, Any] = {}
        ti_path = os.path.join(self.config.output_dir, "training_info.json")
        if os.path.isfile(ti_path):
            with open(ti_path) as f:
                train_info = json.load(f)

        self._log(f"  Loaded {len(vectors)} cached vectors from {vec_dir}/")
        if vector_only_values:
            self._log(
                f"  WARNING: no steerer checkpoints for {len(vector_only_values)} "
                f"values — eval uses additive η·vector fallback (not identical to "
                f"cold-steer FD/kernel hooks). Re-run training once to save "
                f"{{value}}_steerer.pt checkpoints."
            )
        else:
            self._log("  Loaded steerer checkpoints — skipping training")
        return vectors, train_info, steerers_by_value

    # ── Run dir naming (mirrors llm-steering-opt) ────────────────────────

    def _run_dir_name(self, layer: int) -> str:
        model_short = self.config.model_name.split("/")[-1].replace(" ", "_")
        eta = str(self.config.eta).replace(".", "p").replace("-", "neg")
        parts = [
            self.config.method,
            self.config.training_mode,
            f"eta_{eta}",
        ]
        if self.config.method == "cold_fd":
            eps = str(self.config.epsilon).replace(".", "p").replace("-", "neg")
            parts.append(f"eps_{eps}")
        else:
            parts.append(f"kernel_{self.config.kernel}")
        parts.append(f"layer_{layer}")
        parts.append(f"n_train_{self.config.n_training_samples}")
        eval_slug = self.config.eval_metric.replace("_", "-")
        parts.append(f"eval_{eval_slug}")
        return os.path.join(model_short, "-".join(parts))

    # ── Full pipeline ────────────────────────────────────────────────────

    def run(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], int]:
        self._log("=" * 60)
        self._log(f"  {self.config.method} × Schwartz Value Steering Pipeline")
        self._log("=" * 60)

        if torch.cuda.is_available():
            self._log(f"Total GPUs: {torch.cuda.device_count()}")

        self.load_model()
        self.prepare_data()

        best_layer, sweep_payload = self.select_layer()
        self.config.output_dir = os.path.join(
            self._output_base, self._run_dir_name(best_layer)
        )
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        if sweep_payload is not None:
            with open(os.path.join(self.config.output_dir, "layer_sweep.json"), "w") as f:
                json.dump(sweep_payload, f, indent=2)

        method_adapters.set_steering_layers(self.steerable_llm, [best_layer])

        cached = (
            None
            if self.config.force_retrain
            else self.try_load_cached_training(best_layer)
        )
        if cached is not None:
            vectors, train_info, steerers_by_value = cached
        else:
            vectors, train_info, steerers_by_value = self.train_vectors(best_layer)
            if self.config.save_vectors:
                self.save_vectors(vectors, best_layer, steerers_by_value)

        eval_metrics = eval_mod.evaluate_steerer(
            steerable_llm=self.steerable_llm,
            steerers_by_value=steerers_by_value,
            val_rows=self.val_rows,
            values=self.values,
            layer_idx=best_layer,
            method=self.config.method,
            eta=self.config.eta,
            eval_metric=self.config.eval_metric,
            model_name=self.config.model_name,
            n_eval_samples=self.config.n_eval_samples,
            seed=self.config.random_seed,
            use_chat_template=self.config.use_chat_template,
            prompt_template=self.config.prompt_template,
            output_dir=self.config.output_dir,
            verbose=self.config.verbose,
        )

        # Geometry needs all 20 canonical Schwartz values. Skip cleanly if any are missing.
        missing = [v for v in SCHWARTZ_CIRCUMPLEX_ORDER if v not in vectors]
        geometry_metrics: Dict[str, float] = {}
        if missing:
            self._log(
                f"WARNING: missing vectors for {len(missing)} canonical values; "
                f"skipping geometry analysis. Missing: {missing}"
            )
        else:
            geometry_metrics = geometry_mod.analyze_geometry(
                vectors=vectors,
                relations_path=self.config.relations_path,
                output_dir=self.config.output_dir,
                random_seed=self.config.random_seed,
                verbose=self.config.verbose,
            )

        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2, default=str)

        self._log("=" * 60)
        self._log(f"  {self.config.method} pipeline complete!")
        self._log(f"  Results saved to: {self.config.output_dir}/")
        self._log("=" * 60)

        return vectors, {"training_info": train_info,
                         "eval_metrics": eval_metrics,
                         "geometry_metrics": geometry_metrics}, best_layer


# Backward-compatible alias
SchwartzColdFDPipeline = SchwartzColdPipeline
