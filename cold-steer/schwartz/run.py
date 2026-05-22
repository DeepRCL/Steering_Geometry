#!/usr/bin/env python3
"""CLI entry point for the COLD-Steer Schwartz pipeline.

Usage examples (run from the ``cold-steer/`` directory):

    # Default: cold_fd on Qwen3.5-9B-Base
    python -m schwartz.run

    # cold_kernel instead of cold_fd
    python -m schwartz.run --method cold_kernel --kernel constant

    # Swap to a different model (must be Llama-shaped: Qwen, Llama-2/3, Mistral-v0.1, Gemma-2)
    python -m schwartz.run --model_name meta-llama/Llama-2-7b-hf

    # Few-shot regime sweep handled by N_TRAIN_LIST in sbatch_schwartz.slurm, or manually:
    python -m schwartz.run --n_training_samples 10

    # Fixed layer (skip the L2-separation sweep)
    python -m schwartz.run --no_layer_sweep --layer 22

    # Sweep over a custom candidate set
    python -m schwartz.run --layer_candidates 8 12 16 20 24
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(__file__)
_COLD_STEER_ROOT = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.abspath(os.path.join(_COLD_STEER_ROOT, ".."))
if _COLD_STEER_ROOT not in sys.path:
    sys.path.insert(0, _COLD_STEER_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from schwartz.config import SchwartzColdConfig
from schwartz.pipeline import SchwartzColdPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="COLD-Steer × Schwartz value-steering pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Steering method
    p.add_argument("--method", type=str, default=SchwartzColdConfig.method,
                   choices=["cold_fd", "cold_kernel"],
                   help="COLD-Steer variant: finite-difference (cold_fd) or kernel (cold_kernel).")
    p.add_argument("--kernel", type=str, default=SchwartzColdConfig.kernel,
                   help="Kernel type for cold_kernel (e.g. constant, unit, entk_proj_loss). "
                        "YAML 'none' maps to constant.")

    # Model
    p.add_argument("--model_name", type=str, default=SchwartzColdConfig.model_name)
    p.add_argument("--torch_dtype", type=str, default=SchwartzColdConfig.torch_dtype,
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", type=str, default=SchwartzColdConfig.device)

    # Dataset
    p.add_argument("--dataset_path", type=str, default=SchwartzColdConfig.dataset_path,
                   help="Path to final_dataset_v3.csv")
    p.add_argument("--relations_path", type=str, default=SchwartzColdConfig.relations_path,
                   help="Path to schwartz_relations.json")
    p.add_argument("--random_seed", type=int, default=SchwartzColdConfig.random_seed)
    p.add_argument("--no_chat_template", action="store_true",
                   help="Disable chat template (use the fallback prompt template).")

    # Layer selection
    p.add_argument("--no_layer_sweep", action="store_true")
    p.add_argument("--layer", type=int, default=None,
                   help="Fixed layer index (disables sweep).")
    p.add_argument("--layer_candidates", type=int, nargs="+", default=None,
                   help="Candidate layers for the sweep.")
    p.add_argument("--layer_sweep_n_samples", type=int,
                   default=SchwartzColdConfig.layer_sweep_n_samples)

    # COLD-Steer hyperparameters
    p.add_argument("--epsilon", type=float, default=SchwartzColdConfig.epsilon,
                   help="cold_fd only: θ' = θ + ε·mean_grad")
    p.add_argument("--eta", type=float, default=SchwartzColdConfig.eta)
    p.add_argument("--training_mode", type=str, default=SchwartzColdConfig.training_mode,
                   choices=["sft", "dpo", "negative_sft", "ce"],
                   help="Loss used to compute the steering gradient. "
                        "'sft'=NLL on positive (default), 'dpo'=DPO(pos vs neg).")
    p.add_argument("--steer_masking", type=str, default=SchwartzColdConfig.steer_masking,
                   choices=["all", "last"])
    p.add_argument("--gen_masking", type=str, default=SchwartzColdConfig.gen_masking,
                   choices=["prompt", "all"])
    p.add_argument("--n_training_samples", type=int,
                   default=SchwartzColdConfig.n_training_samples,
                   help="Training samples per value; remaining rows per value go to "
                        "validation. cold-steer's paper uses ~50.")

    # Evaluation
    p.add_argument(
        "--eval_metric",
        type=str,
        default=SchwartzColdConfig.eval_metric,
        choices=["full_logprob", "ab_next_token"],
        help="Steering eval: full-answer logprob or CAA-style A/B next-token",
    )
    p.add_argument("--n_eval_samples", type=int, default=SchwartzColdConfig.n_eval_samples,
                   help="Validation samples per value. Use -1 (or 0) to evaluate on "
                        "ALL remaining val rows for each value.")

    # Output
    p.add_argument("--output_dir", type=str, default=SchwartzColdConfig.output_dir)
    p.add_argument("--no_save_vectors", action="store_true")
    p.add_argument(
        "--force_retrain",
        action="store_true",
        help="Ignore cached vectors/steerers and train from scratch",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    layer_sweep_candidates = args.layer_candidates
    if args.layer is not None:
        layer_sweep_candidates = [args.layer]
    layer_sweep_enabled = (not args.no_layer_sweep) and (args.layer is None)

    config = SchwartzColdConfig(
        method=args.method,
        kernel=args.kernel,
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device=args.device,
        dataset_path=args.dataset_path,
        relations_path=args.relations_path,
        random_seed=args.random_seed,
        use_chat_template=not args.no_chat_template,
        layer_sweep_enabled=layer_sweep_enabled,
        layer_sweep_candidates=layer_sweep_candidates,
        layer_sweep_n_samples=args.layer_sweep_n_samples,
        epsilon=args.epsilon,
        eta=args.eta,
        training_mode=args.training_mode,
        steer_masking=args.steer_masking,
        gen_masking=args.gen_masking,
        n_training_samples=args.n_training_samples,
        eval_metric=args.eval_metric,
        n_eval_samples=args.n_eval_samples,
        output_dir=args.output_dir,
        save_vectors=not args.no_save_vectors,
        force_retrain=args.force_retrain,
        verbose=not args.quiet,
    )

    pipeline = SchwartzColdPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
