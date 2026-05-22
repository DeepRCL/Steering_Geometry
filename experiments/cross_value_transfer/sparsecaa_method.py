"""
SparseCAA implementation of SteeringMethod.

SparseCAA steers in SAE feature space: hook the selected layer's MLP output,
encode it with the fine-tuned SAE, add ``alpha * sparse_persona_vec``, decode
back to dense MLP space, and return the modified MLP output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .circumplex_utils import CIRCUMPLEX_ORDER
from .steering_method import SteeringMethod


def _safe_name(s: str) -> str:
    return s.replace(": ", "__").replace(" ", "_").replace("/", "-")


def _sparse_steer_hook(
    module: Any,
    inputs: Any,
    output: Any,
    sae: Any,
    persona_vec: torch.Tensor,
    alpha: float,
    d_in: int,
) -> Any:
    act = output[0] if isinstance(output, tuple) else output
    original_shape = act.shape
    dtype = act.dtype
    flat = act.reshape(-1, d_in).to(torch.float32)

    z = sae.encode(flat)
    pv = persona_vec.to(device=z.device, dtype=z.dtype)
    z_steered = z + alpha * pv
    recon = sae.decode(z_steered).reshape(original_shape).to(dtype)

    return (recon,) + output[1:] if isinstance(output, tuple) else recon


class SparseCAAMethod(SteeringMethod):
    """Adapter for SAE/SparseCAA output directories."""

    def __init__(
        self,
        run_dir: str | Path,
        layer: Optional[int] = None,
        model_name: Optional[str] = None,
        method_name: str = "sparsecaa",
        sae_path: Optional[str | Path] = None,
        d_in: Optional[int] = None,
        d_sae: Optional[int] = None,
        vector_source: str = "sparse_vectors",
        normalize_vectors: bool = False,
    ) -> None:
        self._run_dir = Path(run_dir).resolve()
        self._layer_override = layer
        self._model_name_override = model_name
        self._method_name = method_name
        self._sae_path_override = Path(sae_path).resolve() if sae_path else None
        self._d_in_override = d_in
        self._d_sae_override = d_sae
        self._vector_source = vector_source
        self.normalize_vectors = normalize_vectors
        self._sae = None

        if not self._run_dir.exists():
            raise FileNotFoundError(
                f"SparseCAAMethod: run_dir does not exist: {self._run_dir}"
            )
        if self._vector_source not in {"sparse_vectors", "geometry_centered"}:
            raise ValueError(
                "SparseCAAMethod: vector_source must be 'sparse_vectors' or "
                f"'geometry_centered', got {self._vector_source!r}."
            )

    @property
    def name(self) -> str:
        return self._method_name

    @property
    def layer(self) -> int:
        if self._layer_override is not None:
            return self._layer_override
        cfg = self._load_config()
        if cfg.get("mlp_layer") is not None:
            return int(cfg["mlp_layer"])
        raise ValueError("SparseCAAMethod: no layer specified and no mlp_layer in config.")

    @property
    def model_name(self) -> str:
        if self._model_name_override is not None:
            return self._model_name_override
        return self._load_config().get("model_name", "unknown")

    @property
    def d_in(self) -> int:
        if self._d_in_override is not None:
            return self._d_in_override
        return int(self._load_config().get("d_in", 4096))

    @property
    def d_sae(self) -> int:
        if self._d_sae_override is not None:
            return self._d_sae_override
        return int(self._load_config().get("d_sae", 16384))

    @property
    def sae_path(self) -> Path:
        if self._sae_path_override is not None:
            return self._sae_path_override
        cfg = self._load_config()
        candidate = self._run_dir / "sae_finetuned.pt"
        if candidate.exists():
            return candidate
        sae_checkpoint = cfg.get("sae_checkpoint")
        if sae_checkpoint:
            p = Path(sae_checkpoint)
            return p if p.is_absolute() else Path.cwd() / p
        return self._run_dir / "sae_finetuned.pt"

    def _load_config(self) -> dict:
        config_path = self._run_dir / "pipeline_config.json"
        if not config_path.exists():
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def cache_metadata(self) -> dict:
        return {
            "run_dir": str(self._run_dir),
            "layer": self.layer,
            "sae_path": str(self.sae_path),
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "vector_source": self._vector_source,
            "normalize_vectors": self.normalize_vectors,
        }

    def _vector_path(self, value: str) -> Path:
        if self._vector_source == "sparse_vectors":
            return self._run_dir / "sparse_vectors" / f"{_safe_name(value)}.pt"
        return self._run_dir / "geometry_centered" / f"{_safe_name(value)}.pt"

    def load_vectors(self) -> Dict[str, torch.Tensor]:
        vectors: Dict[str, torch.Tensor] = {}
        for value in CIRCUMPLEX_ORDER:
            vec_path = self._vector_path(value)
            if not vec_path.exists():
                raise FileNotFoundError(
                    f"SparseCAAMethod: vector file not found for value '{value}'.\n"
                    f"Expected path: {vec_path}"
                )
            vec = torch.load(vec_path, map_location="cpu", weights_only=True)
            vec = vec.to(dtype=torch.float32)
            if self.normalize_vectors:
                vec = vec / vec.norm().clamp_min(1e-12)
            vectors[value] = vec
        return vectors

    def _load_sae(self, device: torch.device) -> Any:
        if self._sae is not None:
            return self._sae

        import sys

        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        from SAE.sae_model import load_sae

        if not self.sae_path.exists():
            raise FileNotFoundError(f"SparseCAAMethod: SAE checkpoint missing: {self.sae_path}")

        self._sae = load_sae(
            str(self.sae_path),
            d_in=self.d_in,
            d_sae=self.d_sae,
            device=str(device),
        )
        return self._sae

    def apply_hook(
        self,
        model_info: Any,
        vector: torch.Tensor,
        alpha: float,
    ) -> Any:
        import sys

        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        from CAA.Geometry.model_loader import get_decoder_layers

        sae = self._load_sae(model_info.device).eval()
        persona_vec = vector.detach().to(model_info.device)
        decoder_layers = get_decoder_layers(model_info)
        mlp_module = decoder_layers[self.layer].mlp

        def hook(module: Any, inputs: Any, output: Any) -> Any:
            return _sparse_steer_hook(
                module,
                inputs,
                output,
                sae=sae,
                persona_vec=persona_vec,
                alpha=alpha,
                d_in=self.d_in,
            )

        return mlp_module.register_forward_hook(hook)

    def remove_hook(self, handle: Any) -> None:
        handle.remove()
