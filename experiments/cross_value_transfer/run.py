"""
CLI entry point for the Cross-Value Steering Transfer experiment.

Usage examples
--------------
Minimal (layer auto-discovered, model name read from config.json):

    python experiments/cross_value_transfer/run.py \\
        --caa_run_dir CAA/Geometry/outputs/qwen3_5_9b/Qwen__Qwen3.5-9B

With explicit overrides:

    python experiments/cross_value_transfer/run.py \\
        --caa_run_dir  CAA/Geometry/outputs/qwen3_5_9b/Qwen__Qwen3.5-9B \\
        --model_name   Qwen/Qwen3.5-9B \\
        --caa_layer    13 \\
        --alpha        20.0 \\
        --methods      caa \\
        --output_dir   experiments/cross_value_transfer/outputs \\
        --seed         42

Evaluate transformed geometry vectors from a geometry run:

    python experiments/cross_value_transfer/run.py \\
        --caa_run_dir  CAA/Geometry/outputs/qwen3_5_9b_base_best_dual_metrics_20260520_183805/Qwen__Qwen3.5-9B-Base \\
        --methods      caa_geometry \\
        --alpha        20.0

Evaluate SphericalSteer with its native geodesic hook:

    python experiments/cross_value_transfer/run.py \\
        --spherical_run_dir SphericalSteer/focused_tuning/k2_bneg0p6_base_final_dual_new_relations/Qwen__Qwen3.5-9B-Base \\
        --methods spherical \\
        --alpha 0.9

Load a saved config JSON (individual flags override it):

    python experiments/cross_value_transfer/run.py \\
        --config       my_run_config.json \\
        --force_recompute

The script resolves all relative paths against the project root
(the directory two levels above this file).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.cross_value_transfer.config import TransferExperimentConfig
from experiments.cross_value_transfer.caa_method import CAAMethod
from experiments.cross_value_transfer.spherical_method import SphericalSteerMethod
from experiments.cross_value_transfer.run_transfer_experiment import run_experiment


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_transfer_experiment",
        description="Cross-Value Steering Transfer experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file ──────────────────────────────────────────────────────────
    p.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a JSON config file (TransferExperimentConfig). "
             "Individual flags override fields from this file.",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        metavar="NAME",
        help="HuggingFace model identifier (e.g. 'Qwen/Qwen3.5-9B'). "
             "If omitted, read from caa_run_dir/config.json.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Torch device ('auto', 'cuda', 'cpu'). Default: auto.",
    )

    # ── Steering ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Steering strength α. Default: 20.0.",
    )

    # ── CAA-specific ─────────────────────────────────────────────────────────
    p.add_argument(
        "--caa_run_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the model-specific CAA output directory that contains "
             "vectors/ and config.json "
             "(e.g. CAA/Geometry/outputs/qwen3_5_9b/Qwen__Qwen3.5-9B).",
    )
    p.add_argument(
        "--caa_layer",
        type=int,
        default=None,
        metavar="N",
        help="Layer index for CAA vectors. If omitted, read from "
             "{caa_run_dir}/layer_selection/selected_layer.json.",
    )
    p.add_argument(
        "--caa_vector_source",
        type=str,
        default=None,
        choices=["vectors", "geometry_vectors"],
        help="Load ordinary CAA vectors from vectors/ or transformed vectors "
             "from geometry_vectors/. Default: vectors.",
    )

    # ── SphericalSteer-specific ──────────────────────────────────────────────
    p.add_argument(
        "--spherical_run_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the model-specific SphericalSteer output directory that "
             "contains vectors/ and config.json.",
    )
    p.add_argument(
        "--spherical_layer",
        type=int,
        default=None,
        metavar="N",
        help="Layer index for SphericalSteer vectors. If omitted, read from "
             "config.json layer_override or geometry_vectors/manifest.json.",
    )
    p.add_argument(
        "--spherical_kappa",
        type=float,
        default=None,
        metavar="FLOAT",
        help="SphericalSteer vMF concentration override. Default: read from "
             "spherical_run_dir/config.json.",
    )
    p.add_argument(
        "--spherical_beta",
        type=float,
        default=None,
        metavar="FLOAT",
        help="SphericalSteer trigger threshold override. Default: read from "
             "spherical_run_dir/config.json.",
    )
    p.add_argument(
        "--spherical_steer_position",
        type=str,
        default=None,
        choices=["last", "all"],
        help="SphericalSteer hook position. Default: read from "
             "spherical_run_dir/config.json.",
    )

    # ── Evaluation ───────────────────────────────────────────────────────────
    p.add_argument(
        "--eval_dataset",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the held-out MCQ evaluation CSV.",
    )
    p.add_argument(
        "--n_eval_samples",
        type=int,
        default=None,
        metavar="N",
        help="Max evaluation instances to sample per value. Default: 100.",
    )
    p.add_argument(
        "--eval_splits",
        type=str,
        default=None,
        metavar="S1[,S2,...]",
        help="Comma-separated CSV split labels to evaluate on. Use 'all' to "
             "disable split filtering. Default: validation,test.",
    )
    p.add_argument(
        "--eval_split_fraction",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Held-out fraction to use when the evaluation CSV has no split "
             "column. Mirrors the CAA Geometry eval_split. Default: 0.1.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducibility. Default: 42.",
    )

    # ── Relations ────────────────────────────────────────────────────────────
    p.add_argument(
        "--relations_path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to schwartz_relations-new.json.",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Root directory for experiment outputs.",
    )

    # ── Methods ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--methods",
        type=str,
        default=None,
        metavar="M1[,M2,...]",
        help="Comma-separated list of method names to run. Default: caa.",
    )

    # ── Flags ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--force_recompute",
        action="store_true",
        default=False,
        help="Recompute T matrix and metrics even if outputs already exist.",
    )

    return p


def _build_config(args: argparse.Namespace) -> TransferExperimentConfig:
    """Merge config file + CLI overrides into a TransferExperimentConfig."""
    # Start from file or defaults
    if args.config is not None:
        cfg = TransferExperimentConfig.load(args.config)
    else:
        cfg = TransferExperimentConfig()

    # Apply CLI overrides (only if explicitly provided)
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.device is not None:
        cfg.device = args.device
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.caa_run_dir is not None:
        cfg.caa_run_dir = args.caa_run_dir
    if args.caa_layer is not None:
        cfg.caa_layer = args.caa_layer
    if args.caa_vector_source is not None:
        cfg.caa_vector_source = args.caa_vector_source
    if args.spherical_run_dir is not None:
        cfg.spherical_run_dir = args.spherical_run_dir
    if args.spherical_layer is not None:
        cfg.spherical_layer = args.spherical_layer
    if args.spherical_kappa is not None:
        cfg.spherical_kappa = args.spherical_kappa
    if args.spherical_beta is not None:
        cfg.spherical_beta = args.spherical_beta
    if args.spherical_steer_position is not None:
        cfg.spherical_steer_position = args.spherical_steer_position
    if args.eval_dataset is not None:
        cfg.eval_dataset_path = args.eval_dataset
    if args.n_eval_samples is not None:
        cfg.n_eval_samples = args.n_eval_samples
    if args.eval_splits is not None:
        if args.eval_splits.strip().lower() == "all":
            cfg.eval_splits = None
        else:
            cfg.eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    if args.eval_split_fraction is not None:
        cfg.eval_split_fraction = args.eval_split_fraction
    if args.seed is not None:
        cfg.seed = args.seed
    if args.relations_path is not None:
        cfg.relations_path = args.relations_path
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.methods is not None:
        cfg.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.force_recompute:
        cfg.force_recompute = True

    return cfg


def _build_methods(cfg: TransferExperimentConfig):
    """Instantiate SteeringMethod objects from the method name list."""
    methods = []
    for method_name in cfg.methods:
        normalized_method_name = method_name.lower()
        if normalized_method_name == "caa":
            if not cfg.caa_run_dir:
                raise ValueError(
                    "Method 'caa' requires --caa_run_dir to be specified."
                )
            output_method_name = (
                "caa_geometry"
                if cfg.caa_vector_source == "geometry_vectors"
                else "caa"
            )
            method = CAAMethod(
                run_dir=cfg.caa_run_dir,
                layer=cfg.caa_layer,
                model_name=cfg.model_name,
                method_name=output_method_name,
                vector_source=cfg.caa_vector_source,
            )
            methods.append(method)
        elif normalized_method_name in {"caa_geometry", "caa_geometry_vectors"}:
            if not cfg.caa_run_dir:
                raise ValueError(
                    f"Method '{method_name}' requires --caa_run_dir to be specified."
                )
            method = CAAMethod(
                run_dir=cfg.caa_run_dir,
                layer=cfg.caa_layer,
                model_name=cfg.model_name,
                method_name="caa_geometry",
                vector_source="geometry_vectors",
            )
            methods.append(method)
        elif normalized_method_name in {"spherical", "sphericalsteer", "spherical_steer"}:
            if not cfg.spherical_run_dir:
                raise ValueError(
                    f"Method '{method_name}' requires --spherical_run_dir to be specified."
                )
            method = SphericalSteerMethod(
                run_dir=cfg.spherical_run_dir,
                layer=cfg.spherical_layer,
                model_name=cfg.model_name,
                method_name="spherical",
                kappa=cfg.spherical_kappa,
                beta=cfg.spherical_beta,
                steer_position=cfg.spherical_steer_position,
            )
            methods.append(method)
        else:
            raise ValueError(
                f"Unknown method '{method_name}'. "
                f"Currently supported: ['caa', 'caa_geometry', 'spherical']."
            )
    return methods


def main(argv=None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Build config
    cfg = _build_config(args)

    # Resolve paths relative to project root
    cfg = cfg.resolve_paths(_PROJECT_ROOT)

    print("TransferExperimentConfig:")
    print(f"  model_name     : {cfg.model_name!r}")
    print(f"  device         : {cfg.device}")
    print(f"  alpha          : {cfg.alpha}")
    print(f"  caa_run_dir    : {cfg.caa_run_dir}")
    print(f"  caa_layer      : {cfg.caa_layer!r}  (None = auto-discover)")
    print(f"  caa_vector_src : {cfg.caa_vector_source}")
    print(f"  spherical_dir  : {cfg.spherical_run_dir}")
    print(f"  spherical_layer: {cfg.spherical_layer!r}  (None = auto-discover)")
    print(f"  spherical_kappa: {cfg.spherical_kappa!r}  (None = run config)")
    print(f"  spherical_beta : {cfg.spherical_beta!r}  (None = run config)")
    print(f"  spherical_pos  : {cfg.spherical_steer_position!r}  (None = run config)")
    print(f"  eval_dataset   : {cfg.eval_dataset_path}")
    print(f"  n_eval_samples : {cfg.n_eval_samples}")
    print(f"  eval_splits    : {cfg.eval_splits if cfg.eval_splits else 'all'}")
    print(f"  eval_fraction  : {cfg.eval_split_fraction}")
    print(f"  seed           : {cfg.seed}")
    print(f"  relations_path : {cfg.relations_path}")
    print(f"  output_dir     : {cfg.output_dir}")
    print(f"  methods        : {cfg.methods}")
    print(f"  force_recompute: {cfg.force_recompute}")
    print()

    # Validate required fields
    if not cfg.methods:
        parser.error("--methods must specify at least one method name.")

    # Build method objects
    methods = _build_methods(cfg)

    # Run
    run_experiment(cfg, methods)


if __name__ == "__main__":
    main()
