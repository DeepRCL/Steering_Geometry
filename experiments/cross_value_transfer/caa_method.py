"""
CAA (Contrastive Activation Addition) implementation of SteeringMethod.

Loads pre-computed CAA difference vectors produced by the CAA/Geometry
pipeline and applies them as additive residual-stream hooks.

Run-directory layout expected (the directory containing ``vectors/``)::

    {run_dir}/
      vectors/
        {safe_name(value)}/
          layer_N.pt          ← one .pt file per extracted layer
      layer_selection/
        selected_layer.json   ← {"selected_layer": N}  (optional)
      config.json             ← PipelineConfig JSON (optional, for model_name)

Layer resolution priority:
  1. Explicit ``layer`` argument passed to the constructor.
  2. ``{run_dir}/layer_selection/selected_layer.json``.
  3. Raises ``ValueError`` listing the .pt files found for the first value.

Model-name resolution priority (used for informational display only):
  1. Explicit ``model_name`` argument passed to the constructor.
  2. ``model_name`` field inside ``{run_dir}/config.json``.
  3. Falls back to ``"unknown"``.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .steering_method import SteeringMethod
from .circumplex_utils import CIRCUMPLEX_ORDER


def _safe_name(s: str) -> str:
    """Convert a Schwartz value name to its filesystem-safe representation.

    Mirrors ``CAA/Geometry/config.py::safe_name`` exactly so that paths match
    what the extraction pipeline wrote to disk.
    """
    return s.replace(": ", "__").replace(" ", "_").replace("/", "-")


def _caa_steering_hook(
    module: Any,
    inputs: Any,
    output: Any,
    vector: torch.Tensor,
    alpha: float,
) -> Any:
    """Forward hook that adds ``alpha * vector`` to every sequence position.

    Mirrors the exact hook in ``CAA/Geometry/steering/caa.py::_steering_hook``.
    The vector must already be on the same device and dtype as the hidden states.
    """
    is_tuple = isinstance(output, tuple)
    hidden_states = output[0] if is_tuple else output
    hidden_states = hidden_states + (alpha * vector)
    return (hidden_states,) + output[1:] if is_tuple else hidden_states


class CAAMethod(SteeringMethod):
    """CAA steering method backed by pre-computed contrastive difference vectors.

    Parameters
    ----------
    run_dir:
        Path to the model-specific output directory produced by the
        CAA/Geometry pipeline.  This is the directory that directly
        contains ``vectors/`` (e.g.
        ``CAA/Geometry/outputs/qwen3_5_9b/Qwen__Qwen3.5-9B``).
    layer:
        Which layer's vector to load.  If ``None``, resolved automatically
        from ``layer_selection/selected_layer.json`` inside ``run_dir``.
    model_name:
        Human-readable model name for display / logging.  If ``None``,
        read from ``config.json`` inside ``run_dir``.
    method_name:
        Override the ``name`` property (default ``"caa"``).
    """

    def __init__(
        self,
        run_dir: str | Path,
        layer: Optional[int] = None,
        model_name: Optional[str] = None,
        method_name: str = "caa",
    ) -> None:
        self._run_dir = Path(run_dir).resolve()
        self._layer_override = layer
        self._model_name_override = model_name
        self._method_name = method_name

        if not self._run_dir.exists():
            raise FileNotFoundError(
                f"CAAMethod: run_dir does not exist: {self._run_dir}"
            )

        # Validate that this looks like the right directory level
        vectors_dir = self._run_dir / "vectors"
        if not vectors_dir.exists():
            # Try to give a helpful suggestion
            parent = self._run_dir.parent
            siblings = [d.name for d in parent.iterdir() if d.is_dir() and (d / "vectors").exists()]
            hint = (
                f"\n  Directories in {parent} that contain vectors/: {siblings}"
                if siblings else ""
            )
            raise FileNotFoundError(
                f"CAAMethod: no 'vectors/' subdirectory found in:\n  {self._run_dir}\n"
                f"Pass the model-specific directory that directly contains vectors/ "
                f"(e.g. CAA/Geometry/outputs/<run>/Qwen__Qwen3.5-9B-Base).{hint}"
            )

    # ── identity ─────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._method_name

    @property
    def layer(self) -> int:
        if self._layer_override is not None:
            return self._layer_override

        selected_layer_path = self._run_dir / "layer_selection" / "selected_layer.json"
        if selected_layer_path.exists():
            with open(selected_layer_path, "r") as f:
                return int(json.load(f)["selected_layer"])

        # Step 3: config.json → layer_override field (set by PipelineConfig.layer_override)
        config_path = self._run_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                data = json.load(f)
            layer_override = data.get("layer_override")
            if layer_override is not None:
                return int(layer_override)

        # Give a helpful error showing what .pt files exist for the first value
        first_value = CIRCUMPLEX_ORDER[0]
        first_vec_dir = self._run_dir / "vectors" / _safe_name(first_value)
        available = sorted(first_vec_dir.glob("layer_*.pt")) if first_vec_dir.exists() else []
        available_layers = [p.stem.replace("layer_", "") for p in available]
        raise ValueError(
            f"CAAMethod: no layer specified and "
            f"'{selected_layer_path}' does not exist.\n"
            f"Available layers for '{first_value}': {available_layers or 'none found'}.\n"
            f"Pass --caa_layer <N> to specify the layer explicitly."
        )

    @property
    def model_name(self) -> str:
        if self._model_name_override is not None:
            return self._model_name_override

        config_path = self._run_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                data = json.load(f)
            return data.get("model_name", "unknown")
        return "unknown"

    # ── vectors ──────────────────────────────────────────────────────────────

    def load_vectors(self) -> Dict[str, torch.Tensor]:
        """Load and unit-normalise the per-value CAA vectors for ``self.layer``.

        Returns
        -------
        dict mapping each value name → normalised 1-D CPU float32 tensor.

        Raises
        ------
        FileNotFoundError
            If any value's vector file is missing, with a message that lists
            what layers are available for that value.
        """
        layer_idx = self.layer
        vectors: Dict[str, torch.Tensor] = {}

        for value in CIRCUMPLEX_ORDER:
            vec_path = (
                self._run_dir / "vectors" / _safe_name(value) / f"layer_{layer_idx}.pt"
            )
            if not vec_path.exists():
                vec_dir = self._run_dir / "vectors" / _safe_name(value)
                available = sorted(vec_dir.glob("layer_*.pt")) if vec_dir.exists() else []
                available_layers = [p.stem.replace("layer_", "") for p in available]
                raise FileNotFoundError(
                    f"CAAMethod: vector file not found for value '{value}' "
                    f"at layer {layer_idx}.\n"
                    f"Expected path: {vec_path}\n"
                    f"Available layers for this value: {available_layers or 'none'}.\n"
                    f"Use --caa_layer with one of the available layer numbers."
                )

            vec = torch.load(vec_path, map_location="cpu", weights_only=True)
            vec = vec.to(dtype=torch.float32)

            norm = vec.norm()
            if norm > 0:
                vec = vec / norm

            vectors[value] = vec

        return vectors

    # ── hooks ─────────────────────────────────────────────────────────────────

    def apply_hook(
        self,
        model_info: Any,
        vector: torch.Tensor,
        alpha: float,
    ) -> Any:
        """Register a residual-stream addition hook at ``self.layer``.

        The vector is moved to the model's device and dtype before the hook
        is registered.  Returns the hook handle for cleanup.
        """
        # Import here to avoid circular/slow imports at module level
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
        hook_fn = partial(_caa_steering_hook, vector=target_vec, alpha=alpha)
        decoder_layers = get_decoder_layers(model_info)
        handle = decoder_layers[self.layer].register_forward_hook(hook_fn)
        return handle

    def remove_hook(self, handle: Any) -> None:
        """Remove the hook registered by ``apply_hook``."""
        handle.remove()
