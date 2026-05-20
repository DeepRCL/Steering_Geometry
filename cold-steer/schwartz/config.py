"""Configuration for the cold-steer Schwartz value-steering pipeline.

Constants here are kept identical to ``llm-steering-opt/pipeline/config.py``
so geometry metrics and plots are directly comparable across methods.
"""

import dataclasses
from pathlib import Path
from typing import List, Optional


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
class SchwartzColdConfig:
    """Hyperparameters for one Schwartz COLD-Steer run (cold_fd or cold_kernel)."""

    # Steering method
    method: str = "cold_fd"   # 'cold_fd' | 'cold_kernel'
    kernel: str = "constant"  # cold_kernel only; see configs/steerer/cold_kernel.yaml

    # Model / device
    model_name: str = "Qwen/Qwen3.5-9B-Base"
    torch_dtype: str = "bfloat16"
    device: str = "cuda:0"

    # Dataset + split (``n_training_samples`` rows per value → train; rest → val)
    dataset_path: str = "final_dataset_v3.csv"
    relations_path: str = "schwartz_relations.json"
    random_seed: int = 10
    use_chat_template: bool = True
    prompt_template: str = (
        "Consider the following question and provide an answer.\n\n"
        "Question: {question}\n\nAnswer:"
    )

    # Layer selection
    layer_sweep_enabled: bool = True
    layer_sweep_candidates: Optional[List[int]] = None
    layer_sweep_n_candidates: int = 12
    layer_sweep_n_samples: int = 10

    # Shared cold-steer hyperparameters
    epsilon: float = 1e-6          # cold_fd only: θ' = θ + ε·mean_grad
    eta: float = 1.0
    training_mode: str = "sft"     # 'sft' | 'dpo' | 'negative_sft' | 'ce'
    steer_masking: str = "all"
    gen_masking: str = "prompt"
    training_batch_size: int = 1
    n_training_samples: int = 50

    # Behavioral evaluation
    n_eval_samples: Optional[int] = 30

    # Output
    output_dir: str = "schwartz_results"
    save_vectors: bool = True
    verbose: bool = True

    def __post_init__(self):
        if self.method not in ("cold_fd", "cold_kernel"):
            raise ValueError(f"Unknown method {self.method!r}; use cold_fd or cold_kernel")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_dtype(self):
        import torch
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.torch_dtype]


# Backward-compatible alias
SchwartzColdFDConfig = SchwartzColdConfig
