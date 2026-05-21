"""
Global configuration, Schwartz theory constants, and PipelineConfig dataclass.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import os, json

# ─────────────────────────────────────────────────────────────────────────────
# Schwartz (2012) Refined Theory — circumplex order (counter-clockwise)
# ─────────────────────────────────────────────────────────────────────────────
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
    """Convert a value name to a filesystem-safe string."""
    return s.replace(": ", "__").replace(" ", "_").replace("/", "-")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    model_name: str = ""
    dataset_path: str = ""
    relations_path: str = ""
    output_dir: str = ""

    # Steering
    alpha_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    steering_method: str = "caa"
    spherical_kappa: float = 20.0
    spherical_beta: float = -0.15
    spherical_steer_position: str = "last"
    spherical_geometry_alpha: Optional[float] = None
    spherical_geometry_source: str = "neg"
    spherical_geometry_vector: str = "displacement"
    opt_lr: float = 0.3
    opt_max_iters: int = 10
    opt_starting_norm: float = 1.0
    opt_max_norm: Optional[float] = None
    opt_n_training_samples: Optional[int] = None
    opt_steer_position: str = "all"
    geometry_transform: str = "none"

    # Data split
    eval_split: float = 0.1
    seed: int = 42

    # Layer selection
    layer_start_frac: float = 0.4   # start extraction from this fraction of depth
    layer_end_frac: float = 1.0     # stop extraction/selection before this fraction of depth
    layer_override: Optional[int] = None  # skip auto-selection, use this layer
    layer_selection_method: str = "normalized_l2"

    # Inference
    device: str = "auto"
    batch_size: int = 8             # for steered evaluation batching

    # Storage
    save_activations: bool = True   # store per-sample HDF5 activations

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def model_name_safe(self) -> str:
        return self.model_name.replace("/", "__").replace(" ", "_")

    @property
    def model_output_dir(self) -> str:
        return os.path.join(self.output_dir, self.model_name_safe)

    def subdir(self, name: str) -> str:
        path = os.path.join(self.model_output_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    def save(self, path: Optional[str] = None):
        path = path or os.path.join(self.model_output_dir, "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
