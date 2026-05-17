"""
Geometry analysis for QwenScopeCAA sparse persona vectors.

Delegates entirely to SAE.SparseCAA.geometry.run_geometry — the math is
identical regardless of SAE architecture.  QwenScopePipelineConfig exposes the
same interface (relations_path, run_dir, seed, subdir) that geometry needs, so
the call is a plain duck-typed pass-through.

Outputs (written under config.run_dir):
  geometry_raw/       — raw sparse vectors
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

from typing import Dict

import torch

# Import the fully-implemented geometry runner from SparseCAA.
# QwenScopePipelineConfig is duck-type compatible: it exposes
# relations_path, run_dir, seed, and subdir() — all that geometry needs.
from SAE.SparseCAA.geometry import run_geometry as _run_geometry

from .config import QwenScopePipelineConfig


def run_geometry(
    config: QwenScopePipelineConfig,
    vectors: Dict[str, torch.Tensor],
) -> dict:
    """
    Run geometry analysis on raw and mean-centred Qwen-Scope sparse persona vectors.

    Args:
        config  : QwenScopePipelineConfig
        vectors : {value → (d_sae,) float32 tensor}

    Returns a dict with Spearman ρ for raw and mean-centred variants.
    """
    return _run_geometry(config, vectors)  # type: ignore[arg-type]
