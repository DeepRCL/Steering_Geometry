"""
Geometry analysis for QwenScopeCAA.

By default this analyzes the actual dense residual-stream displacement induced
by steering:

    value_vector = mean(steered_activation - original_activation)

The final metric/plot generation delegates to SAE.SparseCAA.geometry.run_geometry;
only the vector definition differs from the old QwenScopeCAA behavior.

Outputs (written under config.run_dir):
  geometry_vectors/   — saved dense displacement vectors
  geometry_raw/       — raw geometry vectors
  geometry_centered/  — mean-centred vectors (visualisation only)

Each subdirectory contains:
  spearman_report.json
  geometry_metrics.json
  empirical_similarity_heatmap.png
  theoretical_similarity_heatmap.png
  mds_circumplex.png
  umap_2d.png
  tsne_2d.png
"""
from __future__ import annotations

import json
import os
from typing import Dict

import torch

# Import the fully-implemented geometry runner from SparseCAA.
# QwenScopePipelineConfig is duck-type compatible: it exposes
# relations_path, run_dir, seed, and subdir() — all that geometry needs.
from SAE.SparseCAA.geometry import run_geometry as _run_geometry

from .config import QwenScopePipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from .topk_sae_model import TopKSparseAutoencoder, get_or_download_sae


def _activation_path(config: QwenScopePipelineConfig, value: str) -> str:
    return os.path.join(config.steering_activations_dir, f"{safe_name(value)}.pt")


def _geometry_vector_path(config: QwenScopePipelineConfig, value: str) -> str:
    return os.path.join(config.geometry_vectors_dir, f"{safe_name(value)}.pt")


def _resolve_geometry_alpha(config: QwenScopePipelineConfig) -> float:
    if config.geometry_alpha is not None:
        return float(config.geometry_alpha)

    eval_path = os.path.join(config.evaluation_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            results = json.load(f)

        best_alpha = None
        best_gain = None
        for alpha in config.alpha_values:
            label = str(alpha)
            gains = []
            for value_result in results.values():
                steered = value_result.get("steered", {})
                if label in steered:
                    gains.append(steered[label].get("accuracy_gain_vs_baseline", 0.0))
            if not gains:
                continue
            mean_gain = sum(gains) / len(gains)
            if best_gain is None or mean_gain > best_gain:
                best_gain = mean_gain
                best_alpha = alpha

        if best_alpha is not None:
            print(f"[Geometry] Using best evaluated alpha={best_alpha} for displacement vectors.")
            return float(best_alpha)

    fallback = float(max(config.alpha_values))
    print(f"[Geometry] No evaluated best alpha found; using fallback alpha={fallback}.")
    return fallback


def _select_activations(payload: dict, source: str) -> torch.Tensor:
    if source == "neg":
        return payload["neg"].float()
    if source == "pos":
        return payload["pos"].float()
    if source == "all":
        return torch.cat([payload["pos"].float(), payload["neg"].float()], dim=0)
    raise ValueError("geometry_source must be one of: neg, pos, all")


def _topk_from_pre(sae: TopKSparseAutoencoder, pre: torch.Tensor) -> torch.Tensor:
    topk_vals, topk_idx = pre.topk(sae.k, dim=-1)
    z = torch.zeros_like(pre)
    z.scatter_(-1, topk_idx, topk_vals)
    return z


def _batch_displacements(
    sae: TopKSparseAutoencoder,
    activations: torch.Tensor,
    persona_vec: torch.Tensor,
    alpha: float,
    config: QwenScopePipelineConfig,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    displacements = []
    persona_vec = persona_vec.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, activations.shape[0], batch_size):
            batch = activations[start : start + batch_size].to(device=device, dtype=torch.float32)

            if config.use_pre_topk_personas:
                pre = sae.pre_encode(batch)
                if config.use_delta_correction:
                    delta = batch - sae.decode(_topk_from_pre(sae, pre))
                pre_steered = pre + alpha * persona_vec
                recon = sae.decode(_topk_from_pre(sae, pre_steered))
            else:
                z = sae.encode(batch)
                if config.use_delta_correction:
                    delta = batch - sae.decode(z)
                recon = sae.decode(z + alpha * persona_vec)

            if config.use_delta_correction:
                recon = recon + delta

            displacements.append((recon - batch).detach().cpu())

    return torch.cat(displacements, dim=0)


def _load_cached_geometry_vectors(
    config: QwenScopePipelineConfig,
    alpha: float,
) -> Dict[str, torch.Tensor] | None:
    manifest_path = os.path.join(config.geometry_vectors_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    expected = {
        "geometry_vector": config.geometry_vector,
        "geometry_alpha": alpha,
        "geometry_source": config.geometry_source,
        "layer": config.layer,
        "use_pre_topk_personas": config.use_pre_topk_personas,
        "use_delta_correction": config.use_delta_correction,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None

    if not all(os.path.exists(_geometry_vector_path(config, v)) for v in SCHWARTZ_CIRCUMPLEX_ORDER):
        return None

    print("[Geometry] Loading cached dense displacement vectors.")
    return {
        value: torch.load(_geometry_vector_path(config, value), map_location="cpu")
        for value in SCHWARTZ_CIRCUMPLEX_ORDER
    }


def _save_geometry_vectors(
    config: QwenScopePipelineConfig,
    vectors: Dict[str, torch.Tensor],
    alpha: float,
) -> None:
    os.makedirs(config.geometry_vectors_dir, exist_ok=True)
    manifest = {
        "geometry_vector": config.geometry_vector,
        "geometry_alpha": alpha,
        "geometry_source": config.geometry_source,
        "layer": config.layer,
        "use_pre_topk_personas": config.use_pre_topk_personas,
        "use_delta_correction": config.use_delta_correction,
        "vectors": {},
    }
    for value, vec in vectors.items():
        filename = f"{safe_name(value)}.pt"
        torch.save(vec.detach().cpu(), os.path.join(config.geometry_vectors_dir, filename))
        manifest["vectors"][value] = filename

    with open(os.path.join(config.geometry_vectors_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _compute_displacement_vectors(
    config: QwenScopePipelineConfig,
    persona_vectors: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    alpha = _resolve_geometry_alpha(config)
    cached = _load_cached_geometry_vectors(config, alpha)
    if cached is not None:
        return cached

    missing = [
        _activation_path(config, value)
        for value in SCHWARTZ_CIRCUMPLEX_ORDER
        if not os.path.exists(_activation_path(config, value))
    ]
    if missing:
        raise FileNotFoundError(
            "QwenScopeCAA displacement geometry requires saved training activations. "
            "Run the extract module once with the updated code first. Missing example: "
            f"{missing[0]}"
        )

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested = torch.device(config.device)
        device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")

    print(
        f"[Geometry] Computing dense displacement vectors "
        f"(source={config.geometry_source}, alpha={alpha}, device={device})."
    )
    sae = get_or_download_sae(config, device=str(device), use_finetuned=True).eval()

    displacement_vectors: Dict[str, torch.Tensor] = {}
    for value in SCHWARTZ_CIRCUMPLEX_ORDER:
        payload = torch.load(_activation_path(config, value), map_location="cpu")
        activations = _select_activations(payload, config.geometry_source)
        displacements = _batch_displacements(
            sae=sae,
            activations=activations,
            persona_vec=persona_vectors[value],
            alpha=alpha,
            config=config,
            device=device,
        )
        displacement_vectors[value] = displacements.mean(dim=0)

    _save_geometry_vectors(config, displacement_vectors, alpha)
    return displacement_vectors


def run_geometry(
    config: QwenScopePipelineConfig,
    vectors: Dict[str, torch.Tensor],
) -> dict:
    """
    Run geometry analysis for QwenScopeCAA.

    By default, geometry uses the actual dense residual-stream displacement
    induced by steering: mean(steered_activation - original_activation).  Set
    config.geometry_vector="persona" to analyze the previous SAE persona vector
    directly.

    Args:
        config  : QwenScopePipelineConfig
        vectors : {value → (d_sae,) float32 tensor}

    Returns a dict with Spearman ρ for raw and mean-centred variants.
    """
    if config.geometry_vector == "displacement":
        geometry_vectors = _compute_displacement_vectors(config, vectors)
    elif config.geometry_vector == "persona":
        geometry_vectors = vectors
    else:
        raise ValueError("geometry_vector must be one of: displacement, persona")

    return _run_geometry(config, geometry_vectors)  # type: ignore[arg-type]
