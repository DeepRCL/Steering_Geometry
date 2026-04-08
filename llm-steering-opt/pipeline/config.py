"""
Configuration for the value-steering optimization pipeline.

All hyperparameters are centralized here as a dataclass with sensible defaults.
"""

import dataclasses
from typing import List, Optional
from pathlib import Path


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


@dataclasses.dataclass
class SteeringConfig:
    """
    Configuration for the value-steering pipeline.

    Hyperparameter groups:
        - Model: which model/tokenizer to load
        - Dataset: path, split ratio, prompt formatting
        - Layer selection: sweep candidates and sample counts
        - Optimization: learning rate, iterations, norm constraints
        - Evaluation: Spearman correlation with Schwartz theory
        - Output: where to save vectors and results
    """

    # ── Model ────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen3.5-9B"
    torch_dtype: str = "bfloat16"   # "float16", "bfloat16", "float32"
    device: str = "cuda"

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_path: str = "final_dataset_v3.csv"
    relations_path: str = "schwartz_relations.json"
    train_ratio: float = 0.2       # Per-value stratified split ratio
    random_seed: int = 50
    use_chat_template: bool = True  # Use tokenizer's chat template for prompts
    # Fallback prompt template if chat template is unavailable or disabled
    prompt_template: str = (
        "Consider the following question and provide a well-reasoned argument.\n\n"
        "Question: {question}\n\nArgument:"
    )

    # ── Layer Selection ──────────────────────────────────────────────────
    layer_sweep_enabled: bool = True
    # Candidate layers to sweep; None = auto-compute from model depth
    layer_sweep_candidates: Optional[List[int]] = None
    # Number of candidate layers to auto-select (used when candidates=None)
    layer_sweep_n_candidates: int = 12
    # Samples per value used during layer sweep activation extraction
    layer_sweep_n_samples: int = 10

    # ── Optimization ─────────────────────────────────────────────────────
    lr: float = 0.1
    max_iters: int = 20
    max_norm: Optional[float] = None   # None = unconstrained norm
    starting_norm: float = 1.0
    coldness: float = 0.7
    # Number of training samples per value for full training (None = all available)
    n_training_samples: Optional[int] = None
    target_loss: Optional[float] = None  # Early stopping loss threshold

    # ── Steering Strength ────────────────────────────────────────────────
    alpha: float = 10.0  # Multiplier on the vector at evaluation/inference time

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = "steering_results"
    save_vectors: bool = True
    verbose: bool = True

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_dtype(self):
        import torch
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.torch_dtype]
