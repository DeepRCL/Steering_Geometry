"""
Configuration for the QwenScopeCAA pipeline.

Uses the Qwen-Scope pre-trained TopK SAE
(Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50) which covers all 32 residual-stream
layers of Qwen3.5-9B-Base.

Key differences from SAE/SparseCAA/config.py:
  - Hook point : full residual stream after layer N  (not MLP output)
  - SAE arch   : TopK (k=50, exactly 50 active features)  vs. ReLU
  - d_sae      : 65 536  (16× expansion)  vs. 16 384 (4×)
  - SAE source : downloaded from HuggingFace Hub per-layer  vs. local .pt
  - Fine-tuning: MSE only  (TopK enforces sparsity; no L1 needed)
  - Dataset    : 100 samples/value from the -final CSV  vs. 50 from -v3

Pipeline modules (run in order or independently):
  finetune  — adapt the pre-trained SAE to value-specific residual activations
  extract   — compute per-value persona vectors in the SAE sparse latent space
  evaluate  — steer the model through the sparse SAE space, measure A/B accuracy
  geometry  — Spearman ρ and visualisations (raw + mean-centred)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Schwartz constants
# ──────────────────────────────────────────────────────────────────────────────
SCHWARTZ_CIRCUMPLEX_ORDER: List[str] = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: dependability",
    "Benevolence: caring",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
    "Universalism: objectivity",
]

HIGHER_ORDER_GROUPS = {
    "Openness to Change": [
        "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism",
    ],
    "Self-Enhancement": [
        "Achievement", "Power: dominance", "Power: resources", "Face",
    ],
    "Conservation": [
        "Security: personal", "Security: societal", "Tradition",
        "Conformity: rules", "Conformity: interpersonal", "Humility",
    ],
    "Self-Transcendence": [
        "Benevolence: dependability", "Benevolence: caring",
        "Universalism: concern", "Universalism: nature",
        "Universalism: tolerance", "Universalism: objectivity",
    ],
}

GROUP_COLORS = {
    "Openness to Change": "#D4A017",
    "Self-Enhancement":   "#F44336",
    "Conservation":       "#1E88E5",
    "Self-Transcendence": "#4CAF50",
}


def value_to_group(value: str) -> str:
    for group, members in HIGHER_ORDER_GROUPS.items():
        if value in members:
            return group
    return "Unknown"


def safe_name(s: str) -> str:
    return s.replace(": ", "__").replace(" ", "_").replace("/", "-")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class QwenScopePipelineConfig:
    # ── Data ─────────────────────────────────────────────────────────────────
    # Primary dataset — all rows from all 20 values
    base_dataset_path: str = "CAA/value_data/final_dataset_200.csv"
    # Supplementary dataset — filtered to caa_suitable=True
    # Use all rows from base, then add up to touche_samples_per_value per value.
    # Hedonism has only 90 records in this file; those 90 are used as-is.
    touche_dataset_path: str = "SAE/touche_gemma4-v2_remaining-validated-final.csv"
    # Max rows to take per value from the Touche supplement (90 for Hedonism)
    touche_samples_per_value: int = 200
    # If True, cap all values at the minimum per-value count for strict balance.
    equal_samples_per_value: bool = False

    eval_split: float = 0.1
    seed: int = 42

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen3.5-9B-Base"
    device: str = "auto"

    # ── Qwen-Scope SAE ───────────────────────────────────────────────────────
    # HuggingFace repo ID for the Qwen-Scope SAE collection
    sae_repo: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"
    # Transformer layer index to hook (residual stream post-layer).
    # Qwen-Scope covers layers 0–31; default 16 ≈ 50% depth.
    layer: int = 16
    # TopK budget — the Qwen-Scope SAE keeps exactly k=50 features active
    k: int = 50
    # Model hidden dimension (Qwen3.5-9B-Base)
    d_in: int = 4096
    # SAE feature dimension (16× expansion)
    d_sae: int = 65536

    # ── Fine-tuning ──────────────────────────────────────────────────────────
    # MSE-only fine-tuning (TopK enforces sparsity; no L1 regularisation needed)
    finetune_lr: float = 1e-5
    finetune_epochs: int = 3
    finetune_batch_size: int = 4096   # activation vectors per SAE training step

    # ── Evaluation ───────────────────────────────────────────────────────────
    alpha_values: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 4.0]
    )

    # ── Schwartz relations ───────────────────────────────────────────────────
    relations_path: str = "schwartz_relations.json"

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "SAE/QwenScopeCAA/outputs"

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def model_name_safe(self) -> str:
        return self.model_name.replace("/", "__").replace(" ", "_")

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, f"{self.model_name_safe}_layer{self.layer}")

    def subdir(self, name: str) -> str:
        path = os.path.join(self.run_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def sae_cache_dir(self) -> str:
        """Local directory where downloaded Qwen-Scope layer .pt files are stored."""
        return os.path.join(self.output_dir, "sae_checkpoints")

    @property
    def layer_sae_path(self) -> str:
        """Path to the downloaded (pre-trained) SAE checkpoint for config.layer."""
        return os.path.join(self.sae_cache_dir, f"layer{self.layer}.sae.pt")

    @property
    def finetuned_sae_path(self) -> str:
        """Path where the fine-tuned SAE checkpoint is saved."""
        return os.path.join(self.run_dir, f"sae_finetuned_layer{self.layer}.pt")

    @property
    def sparse_vectors_dir(self) -> str:
        return self.subdir("sparse_vectors")

    def save(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "pipeline_config.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QwenScopePipelineConfig":
        with open(path) as f:
            return cls(**json.load(f))
