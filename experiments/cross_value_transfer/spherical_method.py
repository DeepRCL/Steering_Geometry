"""
SphericalSteer implementation of SteeringMethod.

This adapter loads the prototype vectors written by the CAA/Geometry
SphericalSteer pipeline and applies the same vMF/geodesic steering hook used
by ``CAA.Geometry.steering.spherical``.  Unlike CAA, SphericalSteer is not an
additive ``hidden_states + alpha * vector`` method.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .circumplex_utils import CIRCUMPLEX_ORDER
from .steering_method import SteeringMethod


def _safe_name(s: str) -> str:
    """Convert a Schwartz value name to the CAA/Geometry filesystem form."""
    return s.replace(": ", "__").replace(" ", "_").replace("/", "-")


def _safe_unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def _spherical_update(
    hidden_states: torch.Tensor,
    mu_t: torch.Tensor,
    kappa: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Rotate activations toward ``mu_t`` using the SphericalSteer update."""
    orig_dtype = hidden_states.dtype
    x = hidden_states.float()
    mu_t = _safe_unit(mu_t.to(device=x.device, dtype=torch.float32))
    mu_h = -mu_t

    orig_norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    x_hat = x / orig_norm

    cos_t = (x_hat * mu_t).sum(dim=-1).clamp(-1.0, 1.0)
    cos_h = (x_hat * mu_h).sum(dim=-1).clamp(-1.0, 1.0)
    logits = torch.stack([kappa * cos_t, kappa * cos_h], dim=-1)
    probs = F.softmax(logits, dim=-1)

    delta = probs[..., 1] - probs[..., 0]
    trigger_mask = delta > beta
    if not trigger_mask.any():
        return hidden_states

    denom = max(1e-6, 1.0 - beta)
    t = alpha * (delta - beta) / denom
    t = torch.clamp(t, 0.0, 1.0)

    theta = torch.acos(cos_t)
    valid_mask = trigger_mask & (theta >= 1e-4)
    if not valid_mask.any():
        return hidden_states

    sin_theta = torch.sin(theta).clamp_min(1e-12)
    tangent_dir = (x_hat - cos_t.unsqueeze(-1) * mu_t) / sin_theta.unsqueeze(-1)
    theta_new = (1.0 - t) * theta

    x_new_hat = (
        torch.cos(theta_new).unsqueeze(-1) * mu_t
        + torch.sin(theta_new).unsqueeze(-1) * tangent_dir
    )
    candidate = x_new_hat * orig_norm

    x_new = x.clone()
    x_new[valid_mask] = candidate[valid_mask]
    return x_new.to(orig_dtype)


def _spherical_steering_hook(
    module: Any,
    inputs: Any,
    output: Any,
    mu_t: torch.Tensor,
    kappa: float,
    alpha: float,
    beta: float,
    position: str,
) -> Any:
    is_tuple = isinstance(output, tuple)
    hidden_states = output[0] if is_tuple else output
    if position == "all":
        steered = _spherical_update(hidden_states, mu_t, kappa, alpha, beta)
    elif position == "last":
        steered = hidden_states.clone()
        steered[:, -1, :] = _spherical_update(
            hidden_states[:, -1, :],
            mu_t,
            kappa,
            alpha,
            beta,
        )
    else:
        raise ValueError(f"Unknown spherical steer position: {position}")
    return (steered,) + output[1:] if is_tuple else steered


class SphericalSteerMethod(SteeringMethod):
    """SphericalSteer adapter backed by precomputed prototype vectors."""

    def __init__(
        self,
        run_dir: str | Path,
        layer: Optional[int] = None,
        model_name: Optional[str] = None,
        method_name: str = "spherical",
        kappa: Optional[float] = None,
        beta: Optional[float] = None,
        steer_position: Optional[str] = None,
    ) -> None:
        self._run_dir = Path(run_dir).resolve()
        self._layer_override = layer
        self._model_name_override = model_name
        self._method_name = method_name

        if not self._run_dir.exists():
            raise FileNotFoundError(
                f"SphericalSteerMethod: run_dir does not exist: {self._run_dir}"
            )
        if not (self._run_dir / "vectors").exists():
            raise FileNotFoundError(
                "SphericalSteerMethod: no vectors/ subdirectory found in:\n"
                f"  {self._run_dir}"
            )

        config = self._load_config()
        self.kappa = float(
            kappa if kappa is not None else config.get("spherical_kappa", 20.0)
        )
        self.beta = float(
            beta if beta is not None else config.get("spherical_beta", -0.15)
        )
        self.steer_position = (
            steer_position
            if steer_position is not None
            else config.get("spherical_steer_position", "last")
        )
        if self.steer_position not in {"last", "all"}:
            raise ValueError(
                "SphericalSteerMethod: steer_position must be 'last' or 'all', "
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
            "SphericalSteerMethod: no layer specified and no layer metadata found.\n"
            f"Available layers for '{first_value}': {available_layers or 'none found'}.\n"
            "Pass --spherical_layer <N> to specify the layer explicitly."
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
            "kappa": self.kappa,
            "beta": self.beta,
            "steer_position": self.steer_position,
        }

    def load_vectors(self) -> Dict[str, torch.Tensor]:
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
                    f"SphericalSteerMethod: vector file not found for value '{value}' "
                    f"at layer {layer_idx}.\n"
                    f"Expected path: {vec_path}\n"
                    f"Available layers for this value: {available_layers or 'none'}."
                )

            vec = torch.load(vec_path, map_location="cpu", weights_only=True)
            vectors[value] = _safe_unit(vec.to(dtype=torch.float32))

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

        mu_t = _safe_unit(vector.detach()).to(
            device=model_info.device,
            dtype=model_info.model.dtype,
        )
        hook_fn = partial(
            _spherical_steering_hook,
            mu_t=mu_t,
            kappa=self.kappa,
            alpha=alpha,
            beta=self.beta,
            position=self.steer_position,
        )
        decoder_layers = get_decoder_layers(model_info)
        return decoder_layers[self.layer].register_forward_hook(hook_fn)

    def remove_hook(self, handle: Any) -> None:
        handle.remove()
