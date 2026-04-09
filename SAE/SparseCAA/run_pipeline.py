"""
SparseCAA pipeline — CLI entry point.

Four modules, run in order or independently:

  finetune   Adapt the pre-trained SAE to value-specific MLP activations.
             Requires GPU + Qwen 3.5 9B.  Writes sae_finetuned.pt.

  extract    Compute sparse CAA persona vectors (one per Schwartz value) using
             the fine-tuned SAE.  Requires GPU + Qwen.  Writes sparse_vectors/.

  evaluate   Steer the model via the sparse SAE space and measure A/B logit
             accuracy.  Requires GPU + Qwen.  Writes evaluation/.

  geometry   Compute Spearman ρ and produce all visualisations (UMAP, t-SNE,
             MDS circumplex, heatmaps) for both raw and mean-centred vectors.
             CPU-only.  Writes geometry_raw/ and geometry_centered/.

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────
  # Run the full pipeline (from project root):
  python -m SAE.SparseCAA.run_pipeline \\
    --base_dataset_path   CAA/value_data/final_dataset_200.csv \\
    --touche_dataset_path SAE/touche_gemma4-v2_remaining-validated-v3.csv \\
    --relations_path      schwartz_relations.json \\
    --sae_checkpoint      SAE/sae_base_best.pt \\
    --modules all

  # GPU steps only (then geometry on another machine):
  python -m SAE.SparseCAA.run_pipeline ... --modules finetune,extract,evaluate

  # CPU geometry only (after GPU steps are done):
  python -m SAE.SparseCAA.run_pipeline ... --modules geometry
"""
import argparse
import json
import os

import torch

from .config import SparseCAAPipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from .finetune_sae import finetune_sae
from .extract_sparse_vectors import extract_sparse_vectors
from .evaluate import evaluate_sparse_steering
from .geometry import run_geometry


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_vectors(config: SparseCAAPipelineConfig):
    """Load all sparse persona vectors from disk."""
    vectors = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        path = os.path.join(config.sparse_vectors_dir, f"{safe_name(val)}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Sparse persona vector not found: {path}\n"
                "Run the 'extract' module first."
            )
        vectors[val] = torch.load(path, map_location="cpu")
    return vectors


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SparseCAA: value persona vectors in SAE sparse latent space",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--base_dataset_path",
        default="CAA/value_data/final_dataset_200.csv",
        help="Primary dataset CSV (default: CAA/value_data/final_dataset_200.csv)",
    )
    p.add_argument(
        "--touche_dataset_path",
        default="SAE/touche_gemma4-v2_remaining-validated-v3.csv",
        help="Touche supplement CSV (filtered to caa_suitable=True, ≤50 per value)",
    )
    p.add_argument(
        "--touche_samples_per_value",
        type=int,
        default=50,
        help="Max rows to take per value from the Touche supplement (default: 50)",
    )
    p.add_argument(
        "--equal_samples_per_value",
        action="store_true",
        help="Cap all values at the minimum per-value count for strict balance",
    )
    p.add_argument("--eval_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model_name", default="Qwen/Qwen3.5-9B")
    p.add_argument(
        "--device",
        default="auto",
        help="auto | cuda | cpu | mps (default: auto)",
    )

    # ── SAE ───────────────────────────────────────────────────────────────────
    p.add_argument(
        "--sae_checkpoint",
        default="SAE/sae_base_best.pt",
        help="Base SAE checkpoint to fine-tune from",
    )
    p.add_argument(
        "--mlp_layer",
        type=int,
        default=16,
        help="Transformer layer index to hook (must match SAE training; default: 16)",
    )
    p.add_argument("--d_in",  type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    p.add_argument("--finetune_lr",      type=float, default=1e-5)
    p.add_argument("--finetune_epochs",  type=int,   default=3)
    p.add_argument("--finetune_batch_size", type=int, default=4096)
    p.add_argument("--l1_coefficient",   type=float, default=0.005)

    # ── Evaluation ────────────────────────────────────────────────────────────
    p.add_argument(
        "--alpha",
        default="0.5,1.0,2.0,4.0",
        help="Comma-separated alpha values for steering evaluation",
    )

    # ── Schwartz relations ────────────────────────────────────────────────────
    p.add_argument("--relations_path", default="schwartz_relations.json")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--output_dir",
        default="SAE/SparseCAA/outputs",
        help="Root output directory (default: SAE/SparseCAA/outputs)",
    )

    # ── Modules ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--modules",
        default="all",
        help=(
            "Comma-separated list of modules to run:\n"
            "  finetune  — fine-tune SAE on value-specific activations\n"
            "  extract   — extract sparse CAA persona vectors\n"
            "  evaluate  — steer model and measure A/B accuracy\n"
            "  geometry  — Spearman ρ, UMAP, t-SNE, MDS, heatmaps\n"
            "  all       — run all four modules in order  (default)"
        ),
    )

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    config = SparseCAAPipelineConfig(
        base_dataset_path=args.base_dataset_path,
        touche_dataset_path=args.touche_dataset_path,
        touche_samples_per_value=args.touche_samples_per_value,
        equal_samples_per_value=args.equal_samples_per_value,
        eval_split=args.eval_split,
        seed=args.seed,
        model_name=args.model_name,
        device=args.device,
        sae_checkpoint=args.sae_checkpoint,
        mlp_layer=args.mlp_layer,
        d_in=args.d_in,
        d_sae=args.d_sae,
        finetune_lr=args.finetune_lr,
        finetune_epochs=args.finetune_epochs,
        finetune_batch_size=args.finetune_batch_size,
        l1_coefficient=args.l1_coefficient,
        alpha_values=[float(a) for a in args.alpha.split(",")],
        relations_path=args.relations_path,
        output_dir=args.output_dir,
    )

    os.makedirs(config.run_dir, exist_ok=True)
    config.save()
    print(f"Config saved → {config.run_dir}/pipeline_config.json")
    print(f"SAE MLP layer: {config.mlp_layer}  |  d_in: {config.d_in}  |  d_sae: {config.d_sae}")

    modules = (
        ["finetune", "extract", "evaluate", "geometry"]
        if args.modules.strip() == "all"
        else [m.strip() for m in args.modules.split(",")]
    )

    sae = None
    vectors = None

    # ── MODULE 1: Fine-tune SAE ───────────────────────────────────────────────
    if "finetune" in modules:
        print("\n" + "=" * 66)
        print("  MODULE: finetune — adapt SAE to value-specific MLP activations")
        print("=" * 66)
        sae = finetune_sae(config)

    # ── MODULE 2: Extract sparse persona vectors ──────────────────────────────
    if "extract" in modules:
        print("\n" + "=" * 66)
        print("  MODULE: extract — sparse CAA persona vectors")
        print("=" * 66)
        vectors = extract_sparse_vectors(config, sae=sae)

    # ── MODULE 3: Evaluate steering ───────────────────────────────────────────
    if "evaluate" in modules:
        print("\n" + "=" * 66)
        print("  MODULE: evaluate — sparse-SAE steering accuracy")
        print("=" * 66)
        if vectors is None:
            vectors = _load_vectors(config)
        evaluate_sparse_steering(config, vectors, sae=sae)

    # ── MODULE 4: Geometry analysis ───────────────────────────────────────────
    if "geometry" in modules:
        print("\n" + "=" * 66)
        print("  MODULE: geometry — Spearman ρ + visualisations")
        print("=" * 66)
        if vectors is None:
            vectors = _load_vectors(config)
        run_geometry(config, vectors)

    print("\nAll requested modules complete.")
    print(f"Results are in: {config.run_dir}")


if __name__ == "__main__":
    main()
