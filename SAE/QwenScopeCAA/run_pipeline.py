"""
QwenScopeCAA pipeline — CLI entry point.

Uses the Qwen-Scope pre-trained TopK SAE (Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50)
to extract sparse CAA persona vectors from the residual stream of Qwen3.5-9B-Base.

Four modules, run in order or independently:

  finetune   Adapt the pre-trained Qwen-Scope SAE to value-specific residual
             activations.  Requires GPU + Qwen3.5-9B-Base.
             Writes sae_finetuned_layer{n}.pt.

  extract    Compute sparse CAA persona vectors (one per Schwartz value) using
             the (optionally fine-tuned) SAE.
             Requires GPU + Qwen.  Writes sparse_vectors/.

  evaluate   Steer the model via the sparse SAE space and measure A/B logit
             accuracy.  Requires GPU + Qwen.  Writes evaluation/.

  geometry   Compute Spearman ρ and produce all visualisations (UMAP, t-SNE,
             MDS circumplex, heatmaps) for both raw and mean-centred vectors.
             CPU-only.  Writes geometry_raw/ and geometry_centered/.

──────────────────────────────────────────────────────────────────────────────
Controlling SAE fine-tuning
──────────────────────────────────────────────────────────────────────────────
  With fine-tuning (default when --modules includes "finetune"):
    python -m SAE.QwenScopeCAA.run_pipeline --layer 20 --modules all

  Without fine-tuning (use pre-trained Qwen-Scope SAE directly):
    python -m SAE.QwenScopeCAA.run_pipeline --layer 20 --skip_finetune
    # equivalently:
    python -m SAE.QwenScopeCAA.run_pipeline --layer 20 --modules extract,evaluate,geometry

──────────────────────────────────────────────────────────────────────────────
Full usage examples
──────────────────────────────────────────────────────────────────────────────
  # Full pipeline at layer 20 (fine-tune → extract → evaluate → geometry):
  python -m SAE.QwenScopeCAA.run_pipeline \\
    --layer 20 --modules all

  # Pre-trained SAE only, layer 16:
  python -m SAE.QwenScopeCAA.run_pipeline \\
    --layer 16 --skip_finetune

  # GPU steps only (run geometry later on CPU):
  python -m SAE.QwenScopeCAA.run_pipeline --layer 20 --modules finetune,extract,evaluate
  python -m SAE.QwenScopeCAA.run_pipeline --layer 20 --modules geometry

  # Custom output directory (e.g. when comparing multiple layers):
  python -m SAE.QwenScopeCAA.run_pipeline \\
    --layer 24 --skip_finetune \\
    --output_dir SAE/QwenScopeCAA/outputs_layer24
"""
import argparse
import os

import torch

from .config import QwenScopePipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from .extract_sparse_vectors import extract_sparse_vectors
from .evaluate import evaluate_sparse_steering
from .finetune_sae import finetune_sae
from .geometry import run_geometry


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_vectors(config: QwenScopePipelineConfig):
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
        description=(
            "QwenScopeCAA: value persona vectors in Qwen-Scope SAE sparse latent space"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--base_dataset_path",
        default="CAA/value_data/final_dataset_200.csv",
        help="Primary dataset CSV — all rows used (default: CAA/value_data/final_dataset_200.csv)",
    )
    p.add_argument(
        "--touche_dataset_path",
        default="SAE/touche_gemma4-v2_remaining-validated-final.csv",
        help=(
            "Touche supplement CSV (filtered to caa_suitable=True, ≤100 per value).\n"
            "Hedonism has only 90 records — all 90 are used without padding."
        ),
    )
    p.add_argument(
        "--touche_samples_per_value",
        type=int,
        default=200,
        help="Max rows to take per value from the Touche supplement (default: 200)",
    )
    p.add_argument(
        "--equal_samples_per_value",
        action="store_true",
        help="Cap all values at the minimum per-value count for strict balance",
    )
    p.add_argument(
        "--no_pre_topk_personas",
        action="store_true",
        help=(
            "Use legacy post-TopK sparse persona vectors instead of the default\n"
            "pre-TopK dense mode.  Not recommended: post-TopK vectors are 99.9%%\n"
            "zeros and include negative activations, hurting geometry."
        ),
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.7,
        help=(
            "Frequency threshold τ ∈ [0, 1]: feature c is included in the persona\n"
            "mean only if it is non-zero in ≥ τ fraction of training samples.\n"
            "0.0 = keep all features (standard mean).  Default: 0.7."
        ),
    )
    p.add_argument(
        "--no_remove_common_features",
        action="store_true",
        help=(
            "Disable common-feature removal.  By default, features that are\n"
            "non-zero in BOTH v_pos and v_neg are zeroed on both sides before\n"
            "the difference is taken, removing shared syntactic/positional noise."
        ),
    )
    p.add_argument(
        "--no_delta_correction",
        action="store_true",
        help=(
            "Disable SAE reconstruction-error correction in the steering hook.\n"
            "By default Δ = act − decode(encode(act)) from the unsteered pass is\n"
            "added back after the steered decode to cancel the SAE's inherent\n"
            "reconstruction error."
        ),
    )
    p.add_argument("--eval_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model_name", default="Qwen/Qwen3.5-9B-Base")
    p.add_argument(
        "--device",
        default="cuda",
        help="cuda | cpu | mps | auto (default: cuda)",
    )

    # ── Qwen-Scope SAE ────────────────────────────────────────────────────────
    p.add_argument(
        "--sae_repo",
        default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",
        help="HuggingFace Hub repo ID for the Qwen-Scope SAE collection",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=16,
        help=(
            "Transformer layer index to hook (residual stream post-layer).\n"
            "Qwen-Scope covers layers 0–31.  Default: 16 (≈50%% depth)."
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        default=50,
        help=(
            "TopK budget: how many SAE features are kept active per token (default: 50).\n"
            "The Qwen-Scope SAE was trained with k=50, but any value can be used at\n"
            "inference time. Lower k = sparser features; higher k = richer activations.\n"
            "Each k gets its own output directory so results never overwrite each other."
        ),
    )
    p.add_argument("--d_in",  type=int, default=4096,  help="Model hidden dim (default: 4096)")
    p.add_argument("--d_sae", type=int, default=65536, help="SAE feature dim (default: 65536)")

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    p.add_argument("--finetune_lr",         type=float, default=1e-5)
    p.add_argument("--finetune_epochs",     type=int,   default=3)
    p.add_argument("--finetune_batch_size", type=int,   default=4096)

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
        default="SAE/QwenScopeCAA/outputs",
        help="Root output directory (default: SAE/QwenScopeCAA/outputs)",
    )

    # ── Module control ────────────────────────────────────────────────────────
    p.add_argument(
        "--modules",
        default="all",
        help=(
            "Comma-separated list of modules to run:\n"
            "  finetune  — fine-tune Qwen-Scope SAE on value-specific activations\n"
            "  extract   — extract sparse CAA persona vectors\n"
            "  evaluate  — steer model and measure A/B accuracy\n"
            "  geometry  — Spearman ρ, UMAP, t-SNE, MDS, heatmaps\n"
            "  all       — run all four modules in order  (default)"
        ),
    )
    p.add_argument(
        "--skip_finetune",
        action="store_true",
        help=(
            "Skip the fine-tuning step and use the pre-trained Qwen-Scope SAE directly.\n"
            "Equivalent to --modules extract,evaluate,geometry"
        ),
    )

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    config = QwenScopePipelineConfig(
        base_dataset_path=args.base_dataset_path,
        touche_dataset_path=args.touche_dataset_path,
        touche_samples_per_value=args.touche_samples_per_value,
        equal_samples_per_value=args.equal_samples_per_value,
        use_pre_topk_personas=not args.no_pre_topk_personas,
        tau=args.tau,
        remove_common_features=not args.no_remove_common_features,
        use_delta_correction=not args.no_delta_correction,
        eval_split=args.eval_split,
        seed=args.seed,
        model_name=args.model_name,
        device=args.device,
        sae_repo=args.sae_repo,
        layer=args.layer,
        k=args.k,
        d_in=args.d_in,
        d_sae=args.d_sae,
        finetune_lr=args.finetune_lr,
        finetune_epochs=args.finetune_epochs,
        finetune_batch_size=args.finetune_batch_size,
        alpha_values=[float(a) for a in args.alpha.split(",")],
        relations_path=args.relations_path,
        output_dir=args.output_dir,
    )

    # Resolve module list
    if args.skip_finetune:
        modules = ["extract", "evaluate", "geometry"]
    elif args.modules.strip() == "all":
        modules = ["finetune", "extract", "evaluate", "geometry"]
    else:
        modules = [m.strip() for m in args.modules.split(",")]

    os.makedirs(config.run_dir, exist_ok=True)
    config.save()
    print(f"Config saved → {config.run_dir}/pipeline_config.json")
    print(f"SAE repo     : {config.sae_repo}")
    print(f"Layer        : {config.layer}  |  k={config.k}  |  d_in={config.d_in}  |  d_sae={config.d_sae}")
    print(f"Persona mode : {'pre-TopK dense (recommended)' if config.use_pre_topk_personas else 'post-TopK sparse (legacy)'}")
    print(f"tau          : {config.tau}  |  remove_common: {config.remove_common_features}  |  delta_correction: {config.use_delta_correction}")
    print(f"Fine-tuning  : {'disabled (--skip_finetune)' if args.skip_finetune else 'enabled'}")
    print(f"Modules      : {modules}")

    sae = None
    vectors = None

    # ── MODULE 1: Fine-tune SAE ───────────────────────────────────────────────
    if "finetune" in modules:
        print("\n" + "=" * 66)
        print("  MODULE: finetune — adapt Qwen-Scope SAE to value activations")
        print("=" * 66)
        sae = finetune_sae(config)

    # ── MODULE 2: Extract sparse persona vectors ──────────────────────────────
    if "extract" in modules:
        print("\n" + "=" * 66)
        print("  MODULE: extract — Qwen-Scope sparse CAA persona vectors")
        print("=" * 66)
        vectors = extract_sparse_vectors(config, sae=sae)

    # ── MODULE 3: Evaluate steering ───────────────────────────────────────────
    if "evaluate" in modules:
        print("\n" + "=" * 66)
        print("  MODULE: evaluate — Qwen-Scope SAE steering accuracy")
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
