"""
ODESteer implementation of SteeringMethod.

Two modes are supported:

``exact``
    Fit one ODESteer/StepODESteer model per Schwartz value from the non-eval
    portion of the configured dataset, then apply the fitted nonlinear
    ``steer(hidden, T=alpha)`` transform at evaluation time.

``vectors``
    Load ODESteer's saved mean displacement vectors from ``run_dir/vectors`` and
    apply them additively.  This is cheaper but is only an approximation to the
    nonlinear ODE hook.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .circumplex_utils import CIRCUMPLEX_ORDER
from .config import TransferExperimentConfig
from .steering_method import SteeringMethod


def _ode_safe_name(value: str) -> str:
    return value.lower().replace(": ", "_").replace(":", "_").replace(" ", "_").replace("-", "_")


def _format_qa_prompt(question: str, answer: str, tokenizer, model_name: Optional[str]) -> str:
    """Match odesteer/scripts/schwartz/config.py::format_qa_prompt."""
    name_lower = (model_name or "").lower()
    is_instruct = "base" not in name_lower if name_lower else bool(getattr(tokenizer, "chat_template", None))
    if is_instruct and getattr(tokenizer, "chat_template", None):
        try:
            base = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return f"{base} {answer}"
        except Exception:
            pass
    return f"Q: {question}\nA: {answer}"


def _ode_steering_hook(
    module: Any,
    inputs: Any,
    output: Any,
    steer_model: Any,
    alpha: float,
    position: str,
) -> Any:
    is_tuple = isinstance(output, tuple)
    hidden_states = output[0] if is_tuple else output
    hidden = hidden_states.clone()
    orig_dtype = hidden.dtype

    if position == "all":
        flat = hidden.float().reshape(-1, hidden.shape[-1])
        steered = steer_model.steer(flat, T=alpha).to(dtype=orig_dtype)
        hidden = steered.reshape_as(hidden)
    elif position == "last":
        h = hidden[:, -1, :].float()
        hidden[:, -1, :] = steer_model.steer(h, T=alpha).to(dtype=orig_dtype)
    else:
        raise ValueError(f"Unknown ODESteer hook position: {position}")

    return (hidden,) + output[1:] if is_tuple else hidden


def _additive_hook(
    module: Any,
    inputs: Any,
    output: Any,
    vector: torch.Tensor,
    alpha: float,
    position: str,
) -> Any:
    is_tuple = isinstance(output, tuple)
    hidden_states = output[0] if is_tuple else output
    hidden = hidden_states.clone()
    if position == "all":
        hidden = hidden + alpha * vector
    elif position == "last":
        hidden[:, -1, :] = hidden[:, -1, :] + alpha * vector
    else:
        raise ValueError(f"Unknown ODESteer vector hook position: {position}")
    return (hidden,) + output[1:] if is_tuple else hidden


class ODESteerMethod(SteeringMethod):
    """Cross-value-transfer adapter for ODESteer."""

    def __init__(
        self,
        config: TransferExperimentConfig,
        mode: str = "exact",
        method_name: str = "odesteer",
        model_name: Optional[str] = None,
        position: str = "last",
    ) -> None:
        if mode not in {"exact", "vectors"}:
            raise ValueError(f"ODESteerMethod mode must be 'exact' or 'vectors', got {mode!r}.")
        if position not in {"last", "all"}:
            raise ValueError(f"ODESteerMethod position must be 'last' or 'all', got {position!r}.")

        self._config = config
        self._mode = mode
        self._method_name = method_name
        self._model_name_override = model_name
        self._position = position
        self._run_dir = Path(config.odesteer_run_dir).resolve() if config.odesteer_run_dir else None
        self._layer = config.odesteer_layer if config.odesteer_layer is not None else config.caa_layer
        if self._layer is None:
            self._layer = self._layer_from_run_dir()
        if self._layer is None:
            raise ValueError("ODESteerMethod requires --odesteer_layer or --caa_layer.")

        self._steer_models: Dict[str, Any] = {}
        self._vectors: Dict[str, torch.Tensor] = {}
        self._prepared = False

    @property
    def name(self) -> str:
        return self._method_name

    @property
    def layer(self) -> int:
        return int(self._layer)

    @property
    def model_name(self) -> str:
        if self._model_name_override is not None:
            return self._model_name_override
        if self._run_dir is not None:
            config_path = self._run_dir / "config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f).get("model", "unknown")
        return "unknown"

    def _layer_from_run_dir(self) -> Optional[int]:
        if self._run_dir is None:
            return None
        config_path = self._run_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("layer_idx") is not None:
                return int(data["layer_idx"])
        manifest_path = self._run_dir / "vectors" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            values = manifest.get("values", {})
            for entry in values.values():
                if entry.get("layer") is not None:
                    return int(entry["layer"])
        return None

    def _ensure_odesteer_imports(self):
        root = Path(__file__).resolve().parents[2]
        odesteer_src = root / "odesteer" / "src"
        if str(odesteer_src) not in sys.path:
            sys.path.insert(0, str(odesteer_src))
        from odesteer.steer import get_steer_model  # type: ignore

        return get_steer_model

    def _steer_kwargs(self) -> dict:
        if self._config.odesteer_type in {"ODESteer", "RFFODESteer"}:
            return {
                "solver": self._config.odesteer_solver,
                "steps": self._config.odesteer_steps,
                "n_components": self._config.odesteer_n_components,
                "degree": self._config.odesteer_degree,
                "gamma": self._config.odesteer_gamma,
                "coef0": self._config.odesteer_coef0,
                "lin_clf_type": self._config.odesteer_lin_clf_type,
            }
        return {
            "n_components": self._config.odesteer_n_components,
            "degree": self._config.odesteer_degree,
            "gamma": self._config.odesteer_gamma,
            "coef0": self._config.odesteer_coef0,
            "lin_clf_type": self._config.odesteer_lin_clf_type,
        }

    def _load_training_rows(self) -> Dict[str, List[dict]]:
        grouped: Dict[str, List[dict]] = {v: [] for v in CIRCUMPLEX_ORDER}
        allowed_eval_splits = set(self._config.eval_splits or [])
        use_all_eval_splits = not allowed_eval_splits

        with open(self._config.eval_dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            has_split = "split" in fieldnames
            has_caa_suitable = "caa_suitable" in fieldnames
            rows_by_value: Dict[str, List[dict]] = {v: [] for v in CIRCUMPLEX_ORDER}

            for row in reader:
                value = (row.get("value") or "").strip()
                if value not in rows_by_value:
                    continue
                if has_caa_suitable and (row.get("caa_suitable") or "").strip() != "True":
                    continue
                if has_split:
                    split = (row.get("split") or "").strip()
                    if split == "training":
                        grouped[value].append(row)
                    elif use_all_eval_splits:
                        # If eval is explicitly all rows, keep no training rows
                        # to avoid train/eval overlap.
                        continue
                    elif split not in allowed_eval_splits:
                        grouped[value].append(row)
                else:
                    rows_by_value[value].append(row)

        if any(grouped[v] for v in CIRCUMPLEX_ORDER):
            return grouped

        # No split column: mirror cross-value-transfer eval fallback and train
        # on the complement of the deterministic held-out prefix.
        with open(self._config.eval_dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows_by_value = {v: [] for v in CIRCUMPLEX_ORDER}
            for row in reader:
                value = (row.get("value") or "").strip()
                if value in rows_by_value:
                    rows_by_value[value].append(row)

        rng = random.Random(self._config.seed)
        for value in CIRCUMPLEX_ORDER:
            rows = rows_by_value[value]
            rng.shuffle(rows)
            n_eval = int(len(rows) * self._config.eval_split_fraction)
            grouped[value] = rows[n_eval:]
        return grouped

    @torch.no_grad()
    def _extract_activations(self, model_info: Any, rows: List[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        pos_acts = []
        neg_acts = []
        for row in rows:
            for key, out in (("positive_answer", pos_acts), ("negative_answer", neg_acts)):
                prompt = _format_qa_prompt(
                    row.get("question") or "",
                    row.get(key) or "",
                    model_info.tokenizer,
                    self.model_name,
                )
                input_ids = torch.tensor(
                    [model_info.tokenizer.encode(prompt, add_special_tokens=True)],
                    device=model_info.device,
                )
                outputs = model_info.model(input_ids, output_hidden_states=True)
                act = outputs.hidden_states[1:][self.layer][0, -1, :].detach().cpu().float()
                out.append(act)
        return torch.stack(pos_acts), torch.stack(neg_acts)

    def prepare(self, model_info: Any) -> None:
        if self._prepared or self._mode != "exact":
            return

        get_steer_model = self._ensure_odesteer_imports()
        train_rows = self._load_training_rows()
        kwargs = self._steer_kwargs()

        print(
            f"Fitting {self._config.odesteer_type} models for cross-value transfer "
            f"(layer {self.layer})..."
        )
        alpha = float(getattr(self, "alpha", self._config.alpha))
        for value in CIRCUMPLEX_ORDER:
            rows = train_rows[value]
            if len(rows) < 2:
                raise ValueError(f"ODESteerMethod: not enough training rows for '{value}' ({len(rows)}).")
            pos_X, neg_X = self._extract_activations(model_info, rows)
            steer = get_steer_model(self._config.odesteer_type, **kwargs)
            steer.fit(pos_X, neg_X)
            self._steer_models[value] = steer
            disp = steer.steer(pos_X, T=alpha) - pos_X
            self._vectors[value] = disp.mean(dim=0).detach().cpu().float()
            print(f"  {value}: fit on {len(rows)} rows")

        self._prepared = True

    def _load_saved_vectors(self) -> Dict[str, torch.Tensor]:
        if self._run_dir is None:
            raise ValueError("odesteer_vectors requires --odesteer_run_dir.")
        manifest_path = self._run_dir / "vectors" / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"ODESteer vector manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        values = manifest.get("values", {})

        vectors = {}
        for value in CIRCUMPLEX_ORDER:
            entry = values.get(value)
            if not entry:
                raise FileNotFoundError(f"ODESteer manifest has no vector entry for '{value}'.")
            path = self._run_dir / "vectors" / entry["file"]
            if not path.exists():
                raise FileNotFoundError(f"ODESteer vector file not found: {path}")
            vectors[value] = torch.load(path, map_location="cpu", weights_only=True).float()
        return vectors

    def load_vectors(self) -> Dict[str, Any]:
        if self._mode == "exact":
            if not self._prepared:
                raise RuntimeError("ODESteerMethod.prepare(model_info) must be called before load_vectors().")
            return {value: value for value in CIRCUMPLEX_ORDER}
        return self._load_saved_vectors()

    def apply_hook(self, model_info: Any, vector: Any, alpha: float) -> Any:
        from CAA.Geometry.model_loader import get_decoder_layers

        if self._mode == "exact":
            value = str(vector)
            steer_model = self._steer_models[value]
            hook_fn = partial(
                _ode_steering_hook,
                steer_model=steer_model,
                alpha=alpha,
                position=self._position,
            )
        else:
            target_vec = vector.to(device=model_info.device, dtype=model_info.model.dtype)
            hook_fn = partial(
                _additive_hook,
                vector=target_vec,
                alpha=alpha,
                position=self._position,
            )
        return get_decoder_layers(model_info)[self.layer].register_forward_hook(hook_fn)

    def remove_hook(self, handle: Any) -> None:
        handle.remove()

    def cache_metadata(self) -> dict:
        return {
            "mode": self._mode,
            "run_dir": str(self._run_dir) if self._run_dir else "",
            "layer": self.layer,
            "position": self._position,
            "odesteer_type": self._config.odesteer_type,
            "solver": self._config.odesteer_solver,
            "steps": self._config.odesteer_steps,
            "n_components": self._config.odesteer_n_components,
            "degree": self._config.odesteer_degree,
            "gamma": self._config.odesteer_gamma,
            "coef0": self._config.odesteer_coef0,
            "lin_clf_type": self._config.odesteer_lin_clf_type,
        }
