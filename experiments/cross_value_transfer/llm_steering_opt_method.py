"""
llm-steering-opt implementation of SteeringMethod.

This adapter evaluates vectors produced by ``llm-steering-opt/pipeline`` in the
cross-value transfer experiment.  It loads the pipeline's ``vectors/manifest.json``
format and applies the same HuggingFace forward pre-hook semantics used by
``llm-steering-opt/steering_opt.py::make_steering_hook_hf``.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .circumplex_utils import CIRCUMPLEX_ORDER
from .steering_method import SteeringMethod


def _llm_steering_opt_hook(
    module: Any,
    inputs: tuple,
    vector: torch.Tensor,
    alpha: float,
) -> tuple:
    """Forward pre-hook matching llm-steering-opt additive steering."""
    if not inputs:
        return inputs

    hidden_states = inputs[0]
    steered = hidden_states + (alpha * vector)
    return (steered,) + tuple(inputs[1:])


class LLMSteeringOptMethod(SteeringMethod):
    """Steering adapter for saved llm-steering-opt vectors.

    Parameters
    ----------
    run_dir:
        Path to one llm-steering-opt run directory, i.e. the directory that
        contains ``vectors/manifest.json`` and usually ``config.json``.
    layer:
        Optional layer override. If omitted, the layer is inferred from the
        vector manifest.
    normalize_vectors:
        If True, L2-normalise loaded vectors before applying ``alpha``.  The
        default False mirrors llm-steering-opt's native evaluation, where the
        optimized vector norm is part of the learned steering direction.
    """

    def __init__(
        self,
        run_dir: str | Path,
        layer: Optional[int] = None,
        model_name: Optional[str] = None,
        method_name: str = "llm_steering_opt",
        normalize_vectors: bool = False,
    ) -> None:
        self._run_dir = Path(run_dir).resolve()
        self._layer_override = layer
        self._model_name_override = model_name
        self._method_name = method_name
        self._normalize_vectors = normalize_vectors

        if not self._run_dir.exists():
            raise FileNotFoundError(
                f"LLMSteeringOptMethod: run_dir does not exist: {self._run_dir}"
            )

        self._vectors_dir = self._run_dir / "vectors"
        self._manifest_path = self._vectors_dir / "manifest.json"
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                "LLMSteeringOptMethod: expected a llm-steering-opt run directory "
                f"containing vectors/manifest.json, got:\n  {self._run_dir}"
            )

    @property
    def name(self) -> str:
        return self._method_name

    @property
    def layer(self) -> int:
        if self._layer_override is not None:
            return self._layer_override

        manifest = self._load_manifest()
        manifest_layers = {
            int(info["layer"])
            for info in manifest.values()
            if isinstance(info, dict) and "layer" in info
        }
        if len(manifest_layers) == 1:
            return next(iter(manifest_layers))
        if len(manifest_layers) > 1:
            raise ValueError(
                "LLMSteeringOptMethod: manifest contains vectors from multiple "
                f"layers: {sorted(manifest_layers)}. Pass --llm_steering_opt_layer."
            )

        config = self._load_config()
        candidates = config.get("layer_sweep_candidates")
        if isinstance(candidates, list) and len(candidates) == 1:
            return int(candidates[0])

        raise ValueError(
            "LLMSteeringOptMethod: could not infer layer from manifest/config. "
            "Pass --llm_steering_opt_layer <N>."
        )

    @property
    def model_name(self) -> str:
        if self._model_name_override is not None:
            return self._model_name_override
        return self._load_config().get("model_name", "unknown")

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
        metadata = {
            "run_dir": str(self._run_dir),
            "layer": self.layer,
            "normalize_vectors": self._normalize_vectors,
        }
        config = self._load_config()
        if config:
            metadata["source_config"] = {
                k: config.get(k)
                for k in (
                    "model_name",
                    "lr",
                    "alpha",
                    "max_iters",
                    "max_norm",
                    "train_ratio",
                    "random_seed",
                    "eval_metric",
                    "n_training_samples",
                )
                if k in config
            }
        metadata["manifest"] = self._load_manifest()
        return metadata

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
                    "LLMSteeringOptMethod: manifest entry for "
                    f"{value!r} is missing 'vector_file'."
                )

            vec_path = self._vectors_dir / vector_file
            if not vec_path.exists():
                raise FileNotFoundError(
                    f"LLMSteeringOptMethod: vector file missing for {value!r}.\n"
                    f"Expected path: {vec_path}"
                )

            vec = torch.load(vec_path, map_location="cpu", weights_only=True)
            vec = vec.to(dtype=torch.float32)
            if self._normalize_vectors:
                norm = vec.norm()
                if norm > 0:
                    vec = vec / norm
            vectors[value] = vec

        if missing:
            raise FileNotFoundError(
                "LLMSteeringOptMethod: vectors/manifest.json is missing entries "
                f"for values: {missing}"
            )

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

        target_vec = vector.to(
            device=model_info.device,
            dtype=model_info.model.dtype,
        )
        hook_fn = partial(
            _llm_steering_opt_hook,
            vector=target_vec,
            alpha=alpha,
        )
        decoder_layers = get_decoder_layers(model_info)
        return decoder_layers[self.layer].register_forward_pre_hook(hook_fn)

    def remove_hook(self, handle: Any) -> None:
        handle.remove()
