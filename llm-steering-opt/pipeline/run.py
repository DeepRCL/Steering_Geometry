#!/usr/bin/env python3
"""
CLI entry point for the value-steering optimization pipeline.

Usage examples:

    # Run full pipeline with defaults
    python -m pipeline.run

    # Custom split ratio and learning rate
    python -m pipeline.run --train_ratio 0.7 --lr 0.05

    # Skip layer sweep, use specific layer
    python -m pipeline.run --no_layer_sweep --layer 15

    # Custom alpha and norm constraint
    python -m pipeline.run --alpha 0.5 --max_norm 30

    # Use fewer training samples per value (faster)
    python -m pipeline.run --n_training_samples 5
"""

import argparse
import sys
import os

# Add parent dir to path for steering_opt import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from .config import SteeringConfig
from .steering_pipeline import SteeringPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Value-Steering Optimization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model_name", type=str, default=SteeringConfig.model_name,
                        help="HuggingFace model name or path")
    parser.add_argument("--torch_dtype", type=str, default=SteeringConfig.torch_dtype,
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default=SteeringConfig.device)

    # Dataset
    parser.add_argument("--dataset_path", type=str, default=SteeringConfig.dataset_path,
                        help="Path to the CSV dataset")
    parser.add_argument("--train_ratio", type=float, default=SteeringConfig.train_ratio,
                        help="Per-value train split ratio (e.g. 0.6 = 60%% train, 40%% val)")
    parser.add_argument("--random_seed", type=int, default=SteeringConfig.random_seed)
    parser.add_argument("--no_chat_template", action="store_true",
                        help="Disable chat template; use raw prompt_template instead")

    # Layer selection
    parser.add_argument("--no_layer_sweep", action="store_true",
                        help="Disable layer sweep")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to use (disables sweep)")
    parser.add_argument("--layer_candidates", type=int, nargs="+", default=None,
                        help="Specific layer candidates for sweep")
    parser.add_argument("--layer_sweep_n_samples", type=int,
                        default=SteeringConfig.layer_sweep_n_samples,
                        help="Training samples per value during layer sweep")

    # Optimization
    parser.add_argument("--lr", type=float, default=SteeringConfig.lr,
                        help="Learning rate")
    parser.add_argument("--max_iters", type=int, default=SteeringConfig.max_iters,
                        help="Maximum optimization iterations")
    parser.add_argument("--max_norm", type=float, default=None,
                        help="Maximum vector norm constraint (None = unconstrained)")
    parser.add_argument("--starting_norm", type=float,
                        default=SteeringConfig.starting_norm)
    parser.add_argument("--coldness", type=float, default=SteeringConfig.coldness)
    parser.add_argument("--n_training_samples", type=int, default=None,
                        help="Training samples per value (None = all)")
    parser.add_argument("--target_loss", type=float, default=None,
                        help="Early stopping loss threshold")

    # Steering
    parser.add_argument("--alpha", type=float, default=SteeringConfig.alpha,
                        help="Steering strength multiplier at evaluation time")

    # Output
    parser.add_argument("--output_dir", type=str, default=SteeringConfig.output_dir)
    parser.add_argument("--no_save_vectors", action="store_true",
                        help="Do not save steering vectors")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config from args
    config = SteeringConfig(
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device=args.device,
        dataset_path=args.dataset_path,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        use_chat_template=not args.no_chat_template,
        layer_sweep_enabled=not args.no_layer_sweep and args.layer is None,
        layer_sweep_candidates=args.layer_candidates or (
            [args.layer] if args.layer is not None else None
        ),
        layer_sweep_n_samples=args.layer_sweep_n_samples,
        lr=args.lr,
        max_iters=args.max_iters,
        max_norm=args.max_norm,
        starting_norm=args.starting_norm,
        coldness=args.coldness,
        n_training_samples=args.n_training_samples,
        target_loss=args.target_loss,
        alpha=args.alpha,
        output_dir=args.output_dir,
        save_vectors=not args.no_save_vectors,
        verbose=not args.quiet,
    )

    # Run pipeline
    pipeline = SteeringPipeline(config)
    vectors, metrics, best_layer = pipeline.run()

    return vectors, metrics, best_layer


if __name__ == "__main__":
    main()
