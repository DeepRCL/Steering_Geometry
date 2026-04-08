"""
CLI entry point for the SAE value-vector analysis pipeline.

Two modules, run sequentially or independently:

  extract   — Hook into the model's MLP at ``--mlp_layer``, extract CAA
              difference vectors for all 20 Schwartz values, and cache them to
              disk.  Requires a GPU and the full model in memory.

  analyze   — Load cached MLP vectors, project through the SAE, identify and
              remove common (value-generic) features, re-run geometry analysis,
              and test feature disjointness between opposing value groups.
              CPU-only; no model needed.

──────────────────────────────────────────────────────────────────────────────
Quick start (run both steps)
──────────────────────────────────────────────────────────────────────────────
  python -m SAE.run_sae_pipeline \\
    --model_name   Qwen/Qwen3.5-9B \\
    --dataset_path CAA/value_data/final_dataset_200.csv \\
    --relations_path schwartz_relations.json \\
    --sae_checkpoint SAE/sae_base_best.pt \\
    --modules all

──────────────────────────────────────────────────────────────────────────────
Step-by-step (recommended for large models)
──────────────────────────────────────────────────────────────────────────────
  # GPU step
  python -m SAE.run_sae_pipeline \\
    --model_name Qwen/Qwen3.5-9B \\
    --dataset_path CAA/value_data/final_dataset_200.csv \\
    --relations_path schwartz_relations.json \\
    --sae_checkpoint SAE/sae_base_best.pt \\
    --modules extract

  # CPU-only analysis (can run anywhere)
  python -m SAE.run_sae_pipeline \\
    --model_name Qwen/Qwen3.5-9B \\
    --dataset_path CAA/value_data/final_dataset_200.csv \\
    --relations_path schwartz_relations.json \\
    --sae_checkpoint SAE/sae_base_best.pt \\
    --modules analyze
"""
import argparse
import os
import sys

from .config import SAEConfig
from .extract_mlp_vectors import extract_mlp_vectors
from .sae_analysis import SAEAnalyzer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAE analysis pipeline for Schwartz value steering vectors",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required paths
    p.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="CSV with columns: id, question, value, positive_answer, negative_answer",
    )
    p.add_argument(
        "--relations_path",
        type=str,
        required=True,
        help="Path to schwartz_relations.json",
    )
    p.add_argument(
        "--sae_checkpoint",
        type=str,
        required=True,
        help="Path to SAE checkpoint file (e.g. SAE/sae_base_best.pt)",
    )

    # Model
    p.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="HuggingFace model name or local path (default: Qwen/Qwen3.5-9B)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model inference: auto | cuda | cpu | mps (default: auto)",
    )

    # SAE / layer
    p.add_argument(
        "--mlp_layer",
        type=int,
        default=16,
        help="Transformer layer index whose MLP output to use (default: 16)",
    )
    p.add_argument(
        "--d_in",
        type=int,
        default=4096,
        help="Model hidden dimension = SAE input dimension (default: 4096)",
    )
    p.add_argument(
        "--d_sae",
        type=int,
        default=16384,
        help="SAE feature dimension (default: 16384)",
    )

    # Analysis parameters
    p.add_argument(
        "--common_feature_top_k",
        type=int,
        default=128,
        help=(
            "Number of 'universal' features to zero out during purification.\n"
            "These are features with the highest *minimum* activation across all\n"
            "value vectors (active for every value, not value-specific).\n"
            "Higher k = more aggressive purification.  (default: 128)"
        ),
    )
    p.add_argument(
        "--top_features_per_value",
        type=int,
        default=64,
        help=(
            "Top-K active features per value used in the disjointness / Jaccard\n"
            "test.  (default: 64)"
        ),
    )

    # Data split
    p.add_argument(
        "--eval_split",
        type=float,
        default=0.1,
        help="Fraction held out as eval set (default: 0.1 – same as CAA pipeline)",
    )
    p.add_argument("--seed", type=int, default=42)

    # I/O
    p.add_argument(
        "--output_dir",
        type=str,
        default="SAE/outputs",
        help="Root output directory (default: SAE/outputs)",
    )

    # Modules
    p.add_argument(
        "--modules",
        type=str,
        default="all",
        help=(
            "Comma-separated list of modules to run:\n"
            "  extract  – extract MLP CAA vectors (needs GPU)\n"
            "  analyze  – SAE projection, purification, geometry, disjointness\n"
            "  all      – run both in order  (default)"
        ),
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    config = SAEConfig(
        model_name=args.model_name,
        device=args.device,
        dataset_path=args.dataset_path,
        relations_path=args.relations_path,
        sae_checkpoint=args.sae_checkpoint,
        mlp_layer=args.mlp_layer,
        d_in=args.d_in,
        d_sae=args.d_sae,
        common_feature_top_k=args.common_feature_top_k,
        top_features_per_value=args.top_features_per_value,
        eval_split=args.eval_split,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    config.save()
    print(f"Config saved to {config.output_dir}/{config.model_name_safe}/sae_config.json")

    if args.modules.strip() == "all":
        modules = ["extract", "analyze"]
    else:
        modules = [m.strip() for m in args.modules.split(",")]

    vectors = None

    if "extract" in modules:
        print("\n══════════════════════════════════════════════════════════════")
        print("  MODULE: extract – MLP vector extraction")
        print("══════════════════════════════════════════════════════════════")
        vectors = extract_mlp_vectors(config)

    if "analyze" in modules:
        print("\n══════════════════════════════════════════════════════════════")
        print("  MODULE: analyze – SAE projection & geometry analysis")
        print("══════════════════════════════════════════════════════════════")
        analyzer = SAEAnalyzer(config)
        analyzer.run_full_analysis(vectors)


if __name__ == "__main__":
    main()
