"""
Configuration for the SparseCAA pipeline.

The pipeline operates in four sequential modules:
  finetune  — adapt the pre-trained SAE to value-specific MLP activations
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
# Schwartz constants — duplicated here for self-containment
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
# Pipeline Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SparseCAAPipelineConfig:
    # ── Data ─────────────────────────────────────────────────────────────────
    # Primary dataset (all 20 values, 181–276 per value)
    base_dataset_path: str = "CAA/value_data/final_dataset_200.csv"
    # Supplementary dataset (filtered to caa_suitable=True, up to 50 per value)
    touche_dataset_path: str = "SAE/touche_gemma4-v2_remaining-validated-v3.csv"
    # Max rows to take per value from the Touche supplement
    touche_samples_per_value: int = 50
    # If True, cap all values at the minimum per-value count for strict balance.
    # Set True if downstream steps encounter tensor shape mismatches.
    equal_samples_per_value: bool = False

    eval_split: float = 0.1
    seed: int = 42

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen3.5-9B-Base"
    device: str = "auto"

    # ── SAE ──────────────────────────────────────────────────────────────────
    # Path to the base SAE checkpoint (starting point for fine-tuning)
    sae_checkpoint: str = "SAE/sae_base_best.pt"
    # Layer whose MLP output to hook — must match the SAE's training layer
    mlp_layer: int = 16
    d_in: int = 4096    # Qwen 3.5 9B hidden dim = SAE input dim
    d_sae: int = 16384  # SAE feature dim (4× expansion)

    # ── Fine-tuning ──────────────────────────────────────────────────────────
    finetune_lr: float = 1e-5
    finetune_epochs: int = 3
    finetune_batch_size: int = 4096   # activation vectors per SAE training step
    l1_coefficient: float = 0.005    # sparsity penalty (same as pre-training)

    # ── Evaluation ───────────────────────────────────────────────────────────
    alpha_values: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 4.0]
    )

    # ── Schwartz relations ───────────────────────────────────────────────────
    relations_path: str = "schwartz_relations.json"

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "SAE/SparseCAA/outputs"

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def model_name_safe(self) -> str:
        return self.model_name.replace("/", "__").replace(" ", "_")

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, self.model_name_safe)

    def subdir(self, name: str) -> str:
        path = os.path.join(self.run_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def finetuned_sae_path(self) -> str:
        return os.path.join(self.run_dir, "sae_finetuned.pt")

    @property
    def sparse_vectors_dir(self) -> str:
        return self.subdir("sparse_vectors")

    def save(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "pipeline_config.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SparseCAAPipelineConfig":
        with open(path) as f:
            return cls(**json.load(f))
