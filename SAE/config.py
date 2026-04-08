"""
Configuration and Schwartz-theory constants for the SAE analysis pipeline.

Keeps this package self-contained so no imports from CAA/Geometry are needed
for the CPU-only analysis steps.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Schwartz (2012) Refined Theory – circumplex order (counter-clockwise)
# Mirrors CAA/Geometry/config.py to keep this package self-contained.
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

HIGHER_ORDER_GROUPS: Dict[str, List[str]] = {
    "Openness to Change": [
        "Self-direction: thought",
        "Self-direction: action",
        "Stimulation",
        "Hedonism",
    ],
    "Self-Enhancement": [
        "Achievement",
        "Power: dominance",
        "Power: resources",
        "Face",
    ],
    "Conservation": [
        "Security: personal",
        "Security: societal",
        "Tradition",
        "Conformity: rules",
        "Conformity: interpersonal",
        "Humility",
    ],
    "Self-Transcendence": [
        "Benevolence: dependability",
        "Benevolence: caring",
        "Universalism: concern",
        "Universalism: nature",
        "Universalism: tolerance",
        "Universalism: objectivity",
    ],
}

# Theoretical polar opposites in the Schwartz circumplex
OPPOSING_PAIRS: List[Tuple[str, str]] = [
    ("Openness to Change", "Conservation"),
    ("Self-Enhancement", "Self-Transcendence"),
]

GROUP_COLORS: Dict[str, str] = {
    "Openness to Change": "#2196F3",
    "Self-Enhancement":   "#F44336",
    "Conservation":       "#FF9800",
    "Self-Transcendence": "#4CAF50",
}


def value_to_group(value: str) -> str:
    for group, members in HIGHER_ORDER_GROUPS.items():
        if value in members:
            return group
    return "Unknown"


def safe_name(s: str) -> str:
    """Filesystem-safe string for a value name."""
    return s.replace(": ", "__").replace(" ", "_").replace("/", "-")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SAEConfig:
    # ── SAE ──────────────────────────────────────────────────────────────────
    sae_checkpoint: str = ""      # path to sae_base_best.pt
    mlp_layer: int = 16           # which transformer layer's MLP to hook into
    d_in: int = 4096              # SAE input / model hidden dimension
    d_sae: int = 16384            # SAE feature dimension (4x expansion)

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen3.5-9B"
    device: str = "auto"          # "auto", "cuda", "cpu", "mps"

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_path: str = ""        # CSV with question/value/positive_answer/negative_answer
    relations_path: str = ""      # schwartz_relations.json
    eval_split: float = 0.1
    seed: int = 42

    # ── Analysis ─────────────────────────────────────────────────────────────
    # How many "universal" (common-to-all-values) features to zero out during
    # purification.  Tuned by inspecting the min-activation distribution.
    common_feature_top_k: int = 128
    # Top-K active features per value used in the disjointness / Jaccard test.
    top_features_per_value: int = 64
    # Features with activation ≤ this are treated as "off" in sparsity stats.
    activation_threshold: float = 0.0

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "SAE/outputs"

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def model_name_safe(self) -> str:
        return self.model_name.replace("/", "__").replace(" ", "_")

    def subdir(self, name: str) -> str:
        path = os.path.join(self.output_dir, self.model_name_safe, name)
        os.makedirs(path, exist_ok=True)
        return path

    def save(self):
        path = os.path.join(self.output_dir, self.model_name_safe, "sae_config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SAEConfig":
        with open(path) as f:
            return cls(**json.load(f))
