"""
QwenScopeCAA implementation of SteeringMethod.

QwenScopeCAA steers through the Qwen-Scope TopK SAE over the residual stream:
hook the selected transformer layer output, encode with the TopK SAE, add the
per-value persona vector in SAE feature space, decode back to the residual
stream, and preserve unmodelled residual information via delta correction.
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


def _pre_topk_steer_hook(
    output: Any,
    sae: Any,
    persona_vec: torch.Tensor,
    alpha: float,
    d_in: int,
    use_delta_correction: bool,
) -> Any:
    hidden = output[0] if isinstance(output, tuple) else output
    original_shape = hidden.shape
    dtype = hidden.dtype
    flat = hidden.reshape(-1, d_in).to(torch.float32)

    pre = sae.pre_encode(flat)

    if use_delta_correction:
        topk_vals_u, topk_idx_u = pre.topk(sae.k, dim=-1)
        z_unsteered = torch.zeros_like(pre)
        z_unsteered.scatter_(-1, topk_idx_u, topk_vals_u)
        act_recon = sae.decode(z_unsteered)
        delta = flat - act_recon

    pv = persona_vec.to(device=pre.device, dtype=pre.dtype)
    pre_steered = pre + alpha * pv
    topk_vals, topk_idx = pre_steered.topk(sae.k, dim=-1)
    z_steered = torch.zeros_like(pre_steered)
    z_steered.scatter_(-1, topk_idx, topk_vals)
    recon = sae.decode(z_steered)

    if use_delta_correction:
        recon = recon + delta

    recon = recon.reshape(original_shape).to(dtype)
    return (recon,) + output[1:] if isinstance(output, tuple) else recon


def _post_topk_steer_hook(
    output: Any,
    sae: Any,
    persona_vec: torch.Tensor,
    alpha: float,
    d_in: int,
    use_delta_correction: bool,
) -> Any:
    hidden = output[0] if isinstance(output, tuple) else output
    original_shape = hidden.shape
    dtype = hidden.dtype
    flat = hidden.reshape(-1, d_in).to(torch.float32)

    z = sae.encode(flat)

    if use_delta_correction:
        act_recon = sae.decode(z)
        delta = flat - act_recon

    pv = persona_vec.to(device=z.device, dtype=z.dtype)
    z_steered_raw = z + alpha * pv
    topk_vals, topk_idx = z_steered_raw.topk(sae.k, dim=-1)
    z_steered = torch.zeros_like(z_steered_raw)
    z_steered.scatter_(-1, topk_idx, topk_vals)
    recon = sae.decode(z_steered)

    if use_delta_correction:
        recon = recon + delta

    recon = recon.reshape(original_shape).to(dtype)
    return (recon,) + output[1:] if isinstance(output, tuple) else recon


class QwenScopeMethod(SteeringMethod):
    """Adapter for SAE/QwenScopeCAA output directories."""

    def __init__(
        self,
        run_dir: str | Path,
        layer: Optional[int] = None,
        model_name: Optional[str] = None,
        method_name: str = "qwenscope",
        sae_path: Optional[str | Path] = None,
        vector_source: str = "auto",
        normalize_vectors: bool = False,
        use_delta_correction: Optional[bool] = None,
        use_pre_topk_personas: Optional[bool] = None,
    ) -> None:
        self._run_dir = Path(run_dir).resolve()
        self._layer_override = layer
        self._model_name_override = model_name
        self._method_name = method_name
        self._sae_path_override = Path(sae_path).resolve() if sae_path else None
        self._vector_source = vector_source
        self.normalize_vectors = normalize_vectors
        self._use_delta_correction_override = use_delta_correction
        self._use_pre_topk_personas_override = use_pre_topk_personas
        self._sae = None

        if not self._run_dir.exists():
            raise FileNotFoundError(
                f"QwenScopeMethod: run_dir does not exist: {self._run_dir}"
            )
        if self._vector_source not in {"auto", "sparse_vectors_caa_base", "sparse_vectors"}:
            raise ValueError(
                "QwenScopeMethod: vector_source must be 'auto', "
                "'sparse_vectors_caa_base', or 'sparse_vectors', got "
                f"{self._vector_source!r}."
            )

    @property
    def name(self) -> str:
        return self._method_name

    @property
    def layer(self) -> int:
        if self._layer_override is not None:
            return self._layer_override
        cfg = self._load_config()
        if cfg.get("layer") is not None:
            return int(cfg["layer"])
        raise ValueError("QwenScopeMethod: no layer specified and no layer in config.")

    @property
    def model_name(self) -> str:
        if self._model_name_override is not None:
            return self._model_name_override
        return self._load_config().get("model_name", "unknown")

    @property
    def k(self) -> int:
        return int(self._load_config().get("k", 50))

    @property
    def d_in(self) -> int:
        return int(self._load_config().get("d_in", 4096))

    @property
    def d_sae(self) -> int:
        return int(self._load_config().get("d_sae", 65536))

    @property
    def use_delta_correction(self) -> bool:
        if self._use_delta_correction_override is not None:
            return self._use_delta_correction_override
        return bool(self._load_config().get("use_delta_correction", True))

    @property
    def use_pre_topk_personas(self) -> bool:
        if self._use_pre_topk_personas_override is not None:
            return self._use_pre_topk_personas_override
        return bool(self._load_config().get("use_pre_topk_personas", True))

    @property
    def vector_source(self) -> str:
        if self._vector_source != "auto":
            return self._vector_source
        for dirname in ("sparse_vectors_caa_base", "sparse_vectors"):
            if (self._run_dir / dirname).exists():
                return dirname
        raise FileNotFoundError(
            "QwenScopeMethod: could not auto-detect vectors. Expected "
            f"sparse_vectors_caa_base/ or sparse_vectors/ under {self._run_dir}"
        )

    @property
    def sae_path(self) -> Path:
        if self._sae_path_override is not None:
            return self._sae_path_override
        candidate = self._run_dir / f"sae_finetuned_layer{self.layer}.pt"
        if candidate.exists():
            return candidate
        cfg = self._load_config()
        output_dir = Path(cfg.get("output_dir", self._run_dir.parent))
        cache_candidate = output_dir / "sae_checkpoints" / f"layer{self.layer}.sae.pt"
        return cache_candidate

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
            "vector_source": self.vector_source,
            "normalize_vectors": self.normalize_vectors,
            "k": self.k,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "use_delta_correction": self.use_delta_correction,
            "use_pre_topk_personas": self.use_pre_topk_personas,
        }

    def _vector_path(self, value: str) -> Path:
        return self._run_dir / self.vector_source / f"{_safe_name(value)}.pt"

    def load_vectors(self) -> Dict[str, torch.Tensor]:
        vectors: Dict[str, torch.Tensor] = {}
        for value in CIRCUMPLEX_ORDER:
            vec_path = self._vector_path(value)
            if not vec_path.exists():
                raise FileNotFoundError(
                    f"QwenScopeMethod: vector file not found for value '{value}'.\n"
                    f"Expected path: {vec_path}"
                )
            vec = torch.load(vec_path, map_location="cpu", weights_only=True)
            vec = vec.to(dtype=torch.float32)
            if vec.numel() != self.d_sae:
                raise ValueError(
                    f"QwenScopeMethod: vector for '{value}' has {vec.numel()} "
                    f"entries, expected d_sae={self.d_sae}. This adapter needs "
                    "SAE persona vectors, not dense geometry displacement vectors."
                )
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
        from SAE.QwenScopeCAA.topk_sae_model import load_qwenscope_sae

        if not self.sae_path.exists():
            raise FileNotFoundError(
                f"QwenScopeMethod: SAE checkpoint missing: {self.sae_path}"
            )

        self._sae = load_qwenscope_sae(
            str(self.sae_path),
            k=self.k,
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
        layer_module = decoder_layers[self.layer]

        def hook(module: Any, inputs: Any, output: Any) -> Any:
            if self.use_pre_topk_personas:
                return _pre_topk_steer_hook(
                    output,
                    sae=sae,
                    persona_vec=persona_vec,
                    alpha=alpha,
                    d_in=self.d_in,
                    use_delta_correction=self.use_delta_correction,
                )
            return _post_topk_steer_hook(
                output,
                sae=sae,
                persona_vec=persona_vec,
                alpha=alpha,
                d_in=self.d_in,
                use_delta_correction=self.use_delta_correction,
            )

        return layer_module.register_forward_hook(hook)

    def remove_hook(self, handle: Any) -> None:
        handle.remove()
