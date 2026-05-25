"""
Configuration for the LlamaScopeCAA pipeline.

Uses the Llama-Scope pre-trained SAE collection for meta-llama/Llama-3.1-8B.
The default repo contains residual-stream SAEs for every layer with 8x
expansion, i.e. 32K features for Llama-3.1-8B's 4096 hidden size.

Key differences from SAE/SparseCAA/config.py:
  - Hook point : full residual stream after layer N  (not MLP output)
  - SAE arch   : TopK (k=50, exactly 50 active features)  vs. ReLU
  - d_sae      : 32 768  (8× expansion)  vs. 16 384 (4×)
  - SAE source : downloaded from HuggingFace Hub per-layer  vs. local .pt
  - Fine-tuning: MSE only  (TopK enforces sparsity; no L1 needed)
  - Dataset    : 100 samples/value from the -final CSV  vs. 50 from -v3

Pipeline modules (run in order or independently):
  finetune  — adapt the pre-trained SAE to value-specific residual activations
  extract   — compute per-value persona vectors in the SAE sparse latent space
  evaluate  — steer the model through the sparse SAE space, measure A/B accuracy
  geometry  — Spearman ρ and visualisations (raw + mean-centred)

Persona vector extraction enhancements (all gated by config flags):
  tau                   — frequency threshold τ ∈ [0,1]: feature c is kept in
                          the non-zero mean only if it fires in ≥ τ·N samples.
                          Eliminates noisy features seen in only 1–2 prompts.
  remove_common_features— zero features active in BOTH v_pos and v_neg before
                          subtraction; removes shared syntactic/positional noise.
  use_delta_correction  — Δ = act − decode(encode(act)) from the unsteered pass
                          is added back after the steered decode, correcting for
                          SAE reconstruction error in the steering hook.
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
class LlamaScopePipelineConfig:
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
    model_name: str = "meta-llama/Llama-3.1-8B"
    device: str = "cuda"

    # ── Llama-Scope SAE ───────────────────────────────────────────────────────
    # HuggingFace repo ID for the Llama-Scope SAE collection
    sae_repo: str = "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x"
    sae_expansion: int = 8
    sae_site: str = "R"
    # Transformer layer index to hook (residual stream post-layer).
    # Llama-Scope covers layers 0–31; default 16 ≈ 50% depth.
    layer: int = 16
    # TopK budget — the Llama-Scope SAE keeps exactly k=50 features active
    k: int = 50
    # Model hidden dimension (Llama-3.1-8B)
    d_in: int = 4096
    # SAE feature dimension (8× expansion)
    d_sae: int = 32768

    # ── Persona vector mode ───────────────────────────────────────────────────
    # If True (recommended), compute persona vectors from the DENSE pre-TopK
    # encoder activations (x @ W_enc.T + b_enc) rather than from the post-TopK
    # sparse z.  This avoids two failure modes:
    #   1. Post-TopK vectors are ~99.9% zeros (50/32768 active per sample), so
    #      their mean is dominated by "common" features shared by all values,
    #      collapsing pairwise cosine similarities and hurting geometry.
    #   2. TopK includes negative pre-activations; their presence in the persona
    #      difference is semantically incorrect (features represent presence).
    # Steering is also applied in the pre-activation space (before TopK), so
    # the persona direction biases which 50 features are selected.
    use_pre_topk_personas: bool = True

    # Frequency threshold τ ∈ [0, 1]: feature c is included in the per-value
    # non-zero mean only if it is non-zero in at least τ·N training samples.
    # Features below the threshold are zeroed in the persona vector, eliminating
    # noise from features that fired for only 1–2 prompts.
    # tau=0.0 → keep all features (standard mean over non-zero rows).
    # For dense pre-TopK personas all entries are non-zero, so freq≈1.0 and the
    # threshold is effectively a no-op unless tau is set very close to 1.0.
    tau: float = 0.7

    # If True, zero features that are non-zero in BOTH v_pos and v_neg before
    # computing the difference vector.  Features shared by both sides are likely
    # syntactic or positional artifacts unrelated to the value contrast.
    remove_common_features: bool = True

    # If True, compute the SAE reconstruction residual Δ = act − decode(encode(act))
    # from the unsteered activation and add it back to the steered reconstruction.
    # This corrects for the SAE's inherent reconstruction error, which would
    # otherwise be injected into the residual stream on every steered forward pass,
    # causing erratic behaviour especially in earlier layers.
    use_delta_correction: bool = True

    # ── Fine-tuning ──────────────────────────────────────────────────────────
    # MSE-only fine-tuning (TopK enforces sparsity; no L1 regularisation needed)
    finetune_lr: float = 1e-5
    finetune_epochs: int = 3
    finetune_batch_size: int = 4096   # activation vectors per SAE training step

    # ── Evaluation ───────────────────────────────────────────────────────────
    alpha_values: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 4.0]
    )

    # ── Geometry ─────────────────────────────────────────────────────────────
    # "displacement" analyzes the actual dense residual-stream change induced by
    # the SAE steering hook: steered_activation - original_activation.
    # "persona" preserves the previous behavior and analyzes the SAE persona
    # direction directly.
    geometry_vector: str = "displacement"
    geometry_alpha: Optional[float] = None
    geometry_source: str = "neg"  # neg, pos, or all training activations

    # ── Schwartz relations ───────────────────────────────────────────────────
    relations_path: str = "schwartz_relations.json"

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "SAE/LlamaScopeCAA/outputs"

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def model_name_safe(self) -> str:
        return self.model_name.replace("/", "__").replace(" ", "_")

    @property
    def run_dir(self) -> str:
        # Include k in the directory name so experiments with different TopK
        # budgets never share a cache or overwrite each other's results.
        return os.path.join(self.output_dir, f"{self.model_name_safe}_layer{self.layer}_k{self.k}")

    def subdir(self, name: str) -> str:
        path = os.path.join(self.run_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def sae_cache_dir(self) -> str:
        """Local directory where downloaded Llama-Scope layer .pt files are stored."""
        return os.path.join(self.output_dir, "sae_checkpoints")

    @property
    def layer_sae_path(self) -> str:
        """Path to the downloaded (pre-trained) SAE checkpoint for config.layer."""
        return os.path.join(
            self.sae_cache_dir,
            f"Llama3_1-8B-Base-L{self.layer}{self.sae_site}-{self.sae_expansion}x",
            "checkpoints",
            "final.safetensors",
        )

    @property
    def finetuned_sae_path(self) -> str:
        """Path where the fine-tuned SAE checkpoint is saved."""
        return os.path.join(self.run_dir, f"sae_finetuned_layer{self.layer}.pt")

    @property
    def sparse_vectors_dir(self) -> str:
        return self.subdir("sparse_vectors_caa_base")

    @property
    def steering_split_manifest_path(self) -> str:
        return os.path.join(self.subdir("splits"), "caa_base_split.json")

    @property
    def steering_activations_dir(self) -> str:
        return self.subdir("steering_activations")

    @property
    def geometry_vectors_dir(self) -> str:
        return self.subdir("geometry_vectors")

    @property
    def evaluation_dir(self) -> str:
        return self.subdir("evaluation_caa_base")

    def save(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "pipeline_config.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LlamaScopePipelineConfig":
        with open(path) as f:
            return cls(**json.load(f))
