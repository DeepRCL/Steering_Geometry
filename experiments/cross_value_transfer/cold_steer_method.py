"""
COLD-Steer implementation of SteeringMethod.

This adapter evaluates saved vectors from ``cold-steer/schwartz`` runs in the
cross-value transfer experiment.  The Schwartz COLD pipeline writes one
representative vector per value under ``vectors/manifest.json``.

For ``cold_fd`` runs the saved vector is ``(z(theta') - z(theta)) / epsilon``;
the native COLD hook applies ``z -= eta * vector``.  For ``cold_kernel`` runs
the saved vector is already ``eta * v_steer``; the native hook subtracts it.
``recommended_alpha`` encodes those defaults, while the CLI can still force a
global ``--alpha``.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .circumplex_utils import CIRCUMPLEX_ORDER
from .steering_method import SteeringMethod


def _cold_steer_hook(
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
        raise ValueError(f"Unknown COLD-Steer vector hook position: {position}")

    return (hidden,) + output[1:] if is_tuple else hidden


class ColdSteerMethod(SteeringMethod):
    """Cross-value-transfer adapter for saved COLD-Steer representative vectors."""

    def __init__(
        self,
        run_dir: str | Path,
        layer: Optional[int] = None,
        model_name: Optional[str] = None,
        method_name: str = "cold_steer",
        position: str = "all",
    ) -> None:
        self._run_dir = Path(run_dir).resolve()
        self._layer_override = layer
        self._model_name_override = model_name
        self._method_name = method_name
        self._position = position

        if self._position not in {"all", "last"}:
            raise ValueError(
                "ColdSteerMethod: position must be 'all' or 'last', "
                f"got {self._position!r}."
            )
        if not self._run_dir.exists():
            raise FileNotFoundError(
                f"ColdSteerMethod: run_dir does not exist: {self._run_dir}"
            )

        self._vectors_dir = self._run_dir / "vectors"
        self._manifest_path = self._vectors_dir / "manifest.json"
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                "ColdSteerMethod: expected a cold-steer run directory "
                f"containing vectors/manifest.json, got:\n  {self._run_dir}"
            )

    @property
    def name(self) -> str:
        return self._method_name

    @property
    def layer(self) -> int:
        if self._layer_override is not None:
            return self._layer_override

        manifest_layers = {
            int(info["layer"])
            for info in self._load_manifest().values()
            if isinstance(info, dict) and info.get("layer") is not None
        }
        if len(manifest_layers) == 1:
            return next(iter(manifest_layers))
        if len(manifest_layers) > 1:
            raise ValueError(
                "ColdSteerMethod: manifest contains vectors from multiple "
                f"layers: {sorted(manifest_layers)}. Pass --cold_steer_layer."
            )

        candidates = self._load_config().get("layer_sweep_candidates")
        if isinstance(candidates, list) and len(candidates) == 1:
            return int(candidates[0])

        raise ValueError(
            "ColdSteerMethod: could not infer layer from manifest/config. "
            "Pass --cold_steer_layer <N>."
        )

    @property
    def model_name(self) -> str:
        if self._model_name_override is not None:
            return self._model_name_override
        return self._load_config().get("model_name", "unknown")

    @property
    def recommended_alpha(self) -> float:
        config = self._load_config()
        method = str(config.get("method", "")).lower()
        eta = float(config.get("eta", 1.0))
        if method == "cold_fd":
            return -eta
        if method == "cold_kernel":
            return -1.0
        return -eta

    def _load_manifest(self) -> dict:
        with open(self._manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_config(self) -> dict:
        config_path = self._run_dir / "config.json"
        if not config_path.exists():
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def cache_metadata(self) -> dict:
        config = self._load_config()
        return {
            "run_dir": str(self._run_dir),
            "layer": self.layer,
            "position": self._position,
            "recommended_alpha": self.recommended_alpha,
            "source_config": {
                k: config.get(k)
                for k in (
                    "model_name",
                    "method",
                    "eta",
                    "epsilon",
                    "kernel",
                    "training_mode",
                    "n_training_samples",
                    "eval_metric",
                    "random_seed",
                )
                if k in config
            },
            "manifest": self._load_manifest(),
        }

    def load_vectors(self) -> Dict[str, torch.Tensor]:
        manifest = self._load_manifest()
        vectors: Dict[str, torch.Tensor] = {}
        missing = []

        for value in CIRCUMPLEX_ORDER:
            info = manifest.get(value)
            if not isinstance(info, dict):
                missing.append(value)
                continue
            vector_file = info.get("vector_file")
            if not vector_file:
                raise FileNotFoundError(
                    "ColdSteerMethod: manifest entry for "
                    f"{value!r} is missing 'vector_file'."
                )
            vec_path = self._vectors_dir / vector_file
            if not vec_path.exists():
                raise FileNotFoundError(
                    f"ColdSteerMethod: vector file missing for {value!r}.\n"
                    f"Expected path: {vec_path}"
                )
            vectors[value] = torch.load(
                vec_path,
                map_location="cpu",
                weights_only=True,
            ).float()

        if missing:
            raise FileNotFoundError(
                "ColdSteerMethod: vectors/manifest.json is missing entries "
                f"for values: {missing}"
            )

        return vectors

    def apply_hook(
        self,
        model_info: Any,
        vector: torch.Tensor,
        alpha: float,
    ) -> Any:
        from CAA.Geometry.model_loader import get_decoder_layers

        target_vec = vector.to(
            device=model_info.device,
            dtype=model_info.model.dtype,
        )
        hook_fn = partial(
            _cold_steer_hook,
            vector=target_vec,
            alpha=alpha,
            position=self._position,
        )
        return get_decoder_layers(model_info)[self.layer].register_forward_hook(hook_fn)

    def remove_hook(self, handle: Any) -> None:
        handle.remove()
