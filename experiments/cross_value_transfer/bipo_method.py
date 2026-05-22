"""
BiPO / optimized-vector implementation of SteeringMethod.

The local BiPO Schwartz runs are produced through the CAA/Geometry
``OptimizedSteeringMethod`` path.  At inference time those vectors are applied
with an additive forward pre-hook, optionally at all positions or just the last
position.  The saved vectors are not unit-normalised in the original evaluator,
so this adapter preserves raw magnitudes by default.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .circumplex_utils import CIRCUMPLEX_ORDER
from .steering_method import SteeringMethod


def _safe_name(s: str) -> str:
    """Convert a Schwartz value name to the CAA/Geometry filesystem form."""
    return s.replace(": ", "__").replace(" ", "_").replace("/", "-")


def _additive_pre_hook(
    module: Any,
    inputs: Any,
    vector: torch.Tensor,
    alpha: float,
    position: str,
) -> Any:
    hidden_states = inputs[0]
    delta = alpha * vector.to(device=hidden_states.device, dtype=hidden_states.dtype)

    if position == "all":
        steered = hidden_states + delta
    elif position == "last":
        steered = hidden_states.clone()
        steered[:, -1, :] = steered[:, -1, :] + delta
    else:
        raise ValueError(f"Unknown BiPO steer position: {position}")

    return (steered,) + inputs[1:]


class BiPOMethod(SteeringMethod):
    """Adapter for BiPO/optimized steering-vector runs."""

    def __init__(
        self,
        run_dir: str | Path,
        layer: Optional[int] = None,
        model_name: Optional[str] = None,
        method_name: str = "bipo",
        steer_position: Optional[str] = None,
        normalize_vectors: bool = False,
        vector_source: str = "vectors",
    ) -> None:
        self._run_dir = Path(run_dir).resolve()
        self._layer_override = layer
        self._model_name_override = model_name
        self._method_name = method_name
        self.normalize_vectors = normalize_vectors
        self._vector_source = vector_source

        if not self._run_dir.exists():
            raise FileNotFoundError(f"BiPOMethod: run_dir does not exist: {self._run_dir}")
        if self._vector_source not in {"vectors", "geometry_vectors"}:
            raise ValueError(
                "BiPOMethod: vector_source must be either 'vectors' or "
                f"'geometry_vectors', got {self._vector_source!r}."
            )
        if not (self._run_dir / self._vector_source).exists():
            raise FileNotFoundError(
                f"BiPOMethod: no {self._vector_source}/ subdirectory found in:\n"
                f"  {self._run_dir}"
            )

        config = self._load_config()
        self.steer_position = (
            steer_position
            if steer_position is not None
            else config.get("opt_steer_position", "all")
        )
        if self.steer_position not in {"last", "all"}:
            raise ValueError(
                "BiPOMethod: steer_position must be 'last' or 'all', "
                f"got {self.steer_position!r}."
            )

    @property
    def name(self) -> str:
        return self._method_name

    @property
    def layer(self) -> int:
        if self._layer_override is not None:
            return self._layer_override

        config = self._load_config()
        layer_override = config.get("layer_override")
        if layer_override is not None:
            return int(layer_override)

        manifest_path = self._run_dir / "geometry_vectors" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                return int(json.load(f)["layer_idx"])

        first_value = CIRCUMPLEX_ORDER[0]
        first_vec_dir = self._run_dir / "vectors" / _safe_name(first_value)
        available = (
            sorted(first_vec_dir.glob("layer_*.pt")) if first_vec_dir.exists() else []
        )
        available_layers = [p.stem.replace("layer_", "") for p in available]
        raise ValueError(
            "BiPOMethod: no layer specified and no layer metadata found.\n"
            f"Available layers for '{first_value}': {available_layers or 'none found'}.\n"
            "Pass --bipo_layer <N> to specify the layer explicitly."
        )

    @property
    def model_name(self) -> str:
        if self._model_name_override is not None:
            return self._model_name_override
        return self._load_config().get("model_name", "unknown")

    def _load_config(self) -> dict:
        config_path = self._run_dir / "config.json"
        if not config_path.exists():
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def cache_metadata(self) -> dict:
        return {
            "run_dir": str(self._run_dir),
            "layer": self.layer,
            "vector_source": self._vector_source,
            "steer_position": self.steer_position,
            "normalize_vectors": self.normalize_vectors,
        }

    def _load_geometry_vectors(self) -> Dict[str, torch.Tensor]:
        geometry_dir = self._run_dir / "geometry_vectors"
        manifest_path = geometry_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"BiPOMethod: geometry vector manifest not found: {manifest_path}"
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        vector_files = manifest.get("vectors", {})

        vectors: Dict[str, torch.Tensor] = {}
        missing = []
        for value in CIRCUMPLEX_ORDER:
            filename = vector_files.get(value)
            if not filename:
                missing.append(value)
                continue
            vec_path = geometry_dir / filename
            if not vec_path.exists():
                raise FileNotFoundError(
                    f"BiPOMethod: geometry vector file missing for value '{value}'.\n"
                    f"Expected path: {vec_path}"
                )
            vec = torch.load(vec_path, map_location="cpu", weights_only=True)
            vec = vec.to(dtype=torch.float32)
            if self.normalize_vectors:
                vec = vec / vec.norm().clamp_min(1e-12)
            vectors[value] = vec

        if missing:
            raise FileNotFoundError(
                "BiPOMethod: geometry_vectors/manifest.json is missing entries "
                f"for values: {missing}"
            )
        return vectors

    def load_vectors(self) -> Dict[str, torch.Tensor]:
        if self._vector_source == "geometry_vectors":
            return self._load_geometry_vectors()

        layer_idx = self.layer
        vectors: Dict[str, torch.Tensor] = {}

        for value in CIRCUMPLEX_ORDER:
            vec_path = (
                self._run_dir / "vectors" / _safe_name(value) / f"layer_{layer_idx}.pt"
            )
            if not vec_path.exists():
                vec_dir = self._run_dir / "vectors" / _safe_name(value)
                available = (
                    sorted(vec_dir.glob("layer_*.pt")) if vec_dir.exists() else []
                )
                available_layers = [p.stem.replace("layer_", "") for p in available]
                raise FileNotFoundError(
                    f"BiPOMethod: vector file not found for value '{value}' "
                    f"at layer {layer_idx}.\n"
                    f"Expected path: {vec_path}\n"
                    f"Available layers for this value: {available_layers or 'none'}."
                )

            vec = torch.load(vec_path, map_location="cpu", weights_only=True)
            vec = vec.to(dtype=torch.float32)
            if self.normalize_vectors:
                vec = vec / vec.norm().clamp_min(1e-12)
            vectors[value] = vec

        return vectors

    def apply_hook(
        self,
        model_info: Any,
        vector: torch.Tensor,
        alpha: float,
    ) -> Any:
        import sys
        from pathlib import Path as _Path

        _root = str(_Path(__file__).resolve().parents[2])
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from CAA.Geometry.model_loader import get_decoder_layers

        hook_fn = partial(
            _additive_pre_hook,
            vector=vector.detach(),
            alpha=alpha,
            position=self.steer_position,
        )
        decoder_layers = get_decoder_layers(model_info)
        return decoder_layers[self.layer].register_forward_pre_hook(hook_fn)

    def remove_hook(self, handle: Any) -> None:
        handle.remove()
