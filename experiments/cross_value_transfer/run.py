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

<<<<<<< HEAD
Evaluate BiPO / optimized vectors with their additive pre-hook:

    python experiments/cross_value_transfer/run.py \\
        --bipo_run_dir BiPO/focused_tuning/qwen35_opt_20260520_221258/Qwen__Qwen3.5-9B-Base \\
        --methods bipo \\
        --alpha 10.0

Evaluate SparseCAA / SAS:

    python experiments/cross_value_transfer/run.py \\
        --sparsecaa_run_dir SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B-Base \\
        --methods sparsecaa \\
        --alpha 4.0

Evaluate QwenScopeCAA:

    python experiments/cross_value_transfer/run.py \\
        --qwenscope_run_dir SAE/QwenScopeCAA/outputs_qwenscope_l15_k100_final_20260520_1838051/Qwen__Qwen3.5-9B-Base_layer15_k100 \\
        --methods qwenscope \\
        --alpha 4.0
=======
Fit and evaluate ODESteer with its native nonlinear hook:

    python experiments/cross_value_transfer/run.py \\
        --model_name Qwen/Qwen3.5-9B-Base \\
        --methods odesteer
>>>>>>> 897e0f8ebb2f63ae425742d5af16c6246f50e83d

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
from experiments.cross_value_transfer.bipo_method import BiPOMethod
from experiments.cross_value_transfer.caa_method import CAAMethod
<<<<<<< HEAD
from experiments.cross_value_transfer.qwenscope_method import QwenScopeMethod
from experiments.cross_value_transfer.sparsecaa_method import SparseCAAMethod
=======
from experiments.cross_value_transfer.odesteer_method import ODESteerMethod
from experiments.cross_value_transfer.llm_steering_opt_method import LLMSteeringOptMethod
>>>>>>> 897e0f8ebb2f63ae425742d5af16c6246f50e83d
from experiments.cross_value_transfer.spherical_method import SphericalSteerMethod
from experiments.cross_value_transfer.run_transfer_experiment import (
    recompute_metrics_from_saved_results,
    run_experiment,
)

BEST_ALPHA_BY_METHOD = {
    # Best Qwen3.5-9B Base A/B next-token values recorded in accuracy_gains.md.
    "caa": 20.0,
    "caa_geometry": 20.0,
    "caa_geometry_vectors": 20.0,
    "spherical": 0.9,
    "sphericalsteer": 0.9,
    "spherical_steer": 0.9,
    "bipo": 10.0,
    "sparsecaa": 4.0,
    "odesteer": 20.0,
    "ode": 20.0,
    "odesteer_vectors": 20.0,
    "ode_vectors": 20.0,
    "llm_steering_opt": 40.0,
    "llm-steering-opt": 40.0,
    "llmsteeringopt": 40.0,
    "steering_opt": 40.0,
    "steering-opt": 40.0,
}


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
        help="Force steering strength α for all methods. If omitted, known "
             "method-specific best α values from accuracy_gains.md are used.",
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

<<<<<<< HEAD
    # ── BiPO / optimized-vector-specific ────────────────────────────────────
    p.add_argument(
        "--bipo_run_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the model-specific BiPO/optimized-vector output directory "
             "that contains vectors/ and config.json.",
    )
    p.add_argument(
        "--bipo_layer",
        type=int,
        default=None,
        metavar="N",
        help="Layer index for BiPO vectors. If omitted, read from config.json "
             "layer_override or geometry_vectors/manifest.json.",
    )
    p.add_argument(
        "--bipo_steer_position",
        type=str,
        default=None,
        choices=["all", "last"],
        help="BiPO additive pre-hook position. Default: read opt_steer_position "
             "from bipo_run_dir/config.json.",
    )
    p.add_argument(
        "--bipo_vector_source",
        type=str,
        default=None,
        choices=["vectors", "geometry_vectors"],
        help="Load ordinary BiPO vectors from vectors/ or transformed vectors "
             "from geometry_vectors/. Default: vectors.",
    )
    p.add_argument(
        "--bipo_normalize_vectors",
        action="store_true",
        default=False,
        help="Unit-normalise BiPO vectors before steering. By default raw "
             "optimized-vector magnitudes are preserved.",
    )

    # ── SparseCAA-specific ───────────────────────────────────────────────────
    p.add_argument(
        "--sparsecaa_run_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the SparseCAA output directory containing sparse_vectors/, "
             "pipeline_config.json, and sae_finetuned.pt.",
    )
    p.add_argument(
        "--sparsecaa_layer",
        type=int,
        default=None,
        metavar="N",
        help="Layer index whose MLP output SparseCAA hooks. If omitted, read "
             "mlp_layer from pipeline_config.json.",
    )
    p.add_argument(
        "--sparsecaa_sae_path",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional SAE checkpoint override. Default: sparsecaa_run_dir/sae_finetuned.pt.",
    )
    p.add_argument(
        "--sparsecaa_normalize_vectors",
        action="store_true",
        default=False,
        help="Unit-normalise SparseCAA vectors before steering. By default raw "
             "sparse-vector magnitudes are preserved.",
    )

    # ── QwenScopeCAA-specific ────────────────────────────────────────────────
    p.add_argument(
        "--qwenscope_run_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the QwenScopeCAA output directory containing "
             "sparse_vectors_caa_base/, pipeline_config.json, and "
             "sae_finetuned_layer{layer}.pt.",
    )
    p.add_argument(
        "--qwenscope_layer",
        type=int,
        default=None,
        metavar="N",
        help="Layer index whose post-layer residual stream QwenScopeCAA hooks. "
             "If omitted, read layer from pipeline_config.json.",
    )
    p.add_argument(
        "--qwenscope_sae_path",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional Qwen-Scope SAE checkpoint override. Default: run dir's "
             "fine-tuned SAE checkpoint.",
    )
    p.add_argument(
        "--qwenscope_vector_source",
        type=str,
        default=None,
        choices=["auto", "sparse_vectors_caa_base", "sparse_vectors"],
        help="Load QwenScope SAE persona vectors. Default auto prefers "
             "sparse_vectors_caa_base/ and falls back to sparse_vectors/.",
    )
    p.add_argument(
        "--qwenscope_normalize_vectors",
        action="store_true",
        default=False,
        help="Unit-normalise QwenScope vectors before steering. By default raw "
             "persona-vector magnitudes are preserved.",
=======
    # ── ODESteer-specific ───────────────────────────────────────────────────
    p.add_argument(
        "--odesteer_run_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="ODESteer Schwartz output directory. Required for odesteer_vectors.",
    )
    p.add_argument(
        "--odesteer_layer",
        type=int,
        default=None,
        metavar="N",
        help="Layer index for ODESteer. Default: 18.",
    )
    p.add_argument(
        "--odesteer_type",
        type=str,
        default=None,
        choices=["ODESteer", "StepODESteer"],
        help="ODESteer class to fit. Default: ODESteer.",
    )
    p.add_argument("--odesteer_solver", type=str, default=None, choices=["euler", "midpoint", "rk4"])
    p.add_argument("--odesteer_steps", type=int, default=None)
    p.add_argument("--odesteer_n_components", type=int, default=None)
    p.add_argument("--odesteer_degree", type=int, default=None)
    p.add_argument("--odesteer_gamma", type=float, default=None)
    p.add_argument("--odesteer_coef0", type=float, default=None)
    p.add_argument("--odesteer_lin_clf_type", type=str, default=None)
    # ── llm-steering-opt-specific ───────────────────────────────────────────
    p.add_argument(
        "--llm_steering_opt_run_dir",
        "--llm-steering-opt-run-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the llm-steering-opt run directory containing "
             "vectors/manifest.json and config.json.",
    )
    p.add_argument(
        "--llm_steering_opt_layer",
        "--llm-steering-opt-layer",
        type=int,
        default=None,
        metavar="N",
        help="Layer index for llm-steering-opt vectors. If omitted, read "
             "from vectors/manifest.json.",
    )
    p.add_argument(
        "--llm_steering_opt_normalize_vectors",
        "--llm-steering-opt-normalize-vectors",
        action="store_true",
        default=False,
        help="L2-normalize llm-steering-opt vectors before applying alpha. "
             "Default preserves native llm-steering-opt vector norms.",
>>>>>>> 897e0f8ebb2f63ae425742d5af16c6246f50e83d
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
    p.add_argument(
        "--postprocess_existing",
        action="store_true",
        default=False,
        help="Only recompute metrics/plots from existing T_matrix.npy files in "
             "--output_dir. Does not load the model or run inference.",
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
        cfg.use_method_default_alphas = False
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
<<<<<<< HEAD
    if args.bipo_run_dir is not None:
        cfg.bipo_run_dir = args.bipo_run_dir
    if args.bipo_layer is not None:
        cfg.bipo_layer = args.bipo_layer
    if args.bipo_steer_position is not None:
        cfg.bipo_steer_position = args.bipo_steer_position
    if args.bipo_vector_source is not None:
        cfg.bipo_vector_source = args.bipo_vector_source
    if args.bipo_normalize_vectors:
        cfg.bipo_normalize_vectors = True
    if args.sparsecaa_run_dir is not None:
        cfg.sparsecaa_run_dir = args.sparsecaa_run_dir
    if args.sparsecaa_layer is not None:
        cfg.sparsecaa_layer = args.sparsecaa_layer
    if args.sparsecaa_sae_path is not None:
        cfg.sparsecaa_sae_path = args.sparsecaa_sae_path
    if args.sparsecaa_normalize_vectors:
        cfg.sparsecaa_normalize_vectors = True
    if args.qwenscope_run_dir is not None:
        cfg.qwenscope_run_dir = args.qwenscope_run_dir
    if args.qwenscope_layer is not None:
        cfg.qwenscope_layer = args.qwenscope_layer
    if args.qwenscope_sae_path is not None:
        cfg.qwenscope_sae_path = args.qwenscope_sae_path
    if args.qwenscope_vector_source is not None:
        cfg.qwenscope_vector_source = args.qwenscope_vector_source
    if args.qwenscope_normalize_vectors:
        cfg.qwenscope_normalize_vectors = True
=======
    if args.odesteer_run_dir is not None:
        cfg.odesteer_run_dir = args.odesteer_run_dir
    if args.odesteer_layer is not None:
        cfg.odesteer_layer = args.odesteer_layer
    if args.odesteer_type is not None:
        cfg.odesteer_type = args.odesteer_type
    if args.odesteer_solver is not None:
        cfg.odesteer_solver = args.odesteer_solver
    if args.odesteer_steps is not None:
        cfg.odesteer_steps = args.odesteer_steps
    if args.odesteer_n_components is not None:
        cfg.odesteer_n_components = args.odesteer_n_components
    if args.odesteer_degree is not None:
        cfg.odesteer_degree = args.odesteer_degree
    if args.odesteer_gamma is not None:
        cfg.odesteer_gamma = args.odesteer_gamma
    if args.odesteer_coef0 is not None:
        cfg.odesteer_coef0 = args.odesteer_coef0
    if args.odesteer_lin_clf_type is not None:
        cfg.odesteer_lin_clf_type = args.odesteer_lin_clf_type
    if args.llm_steering_opt_run_dir is not None:
        cfg.llm_steering_opt_run_dir = args.llm_steering_opt_run_dir
    if args.llm_steering_opt_layer is not None:
        cfg.llm_steering_opt_layer = args.llm_steering_opt_layer
    if args.llm_steering_opt_normalize_vectors:
        cfg.llm_steering_opt_normalize_vectors = True
>>>>>>> 897e0f8ebb2f63ae425742d5af16c6246f50e83d
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


def _method_alpha(method_name: str, cfg: TransferExperimentConfig) -> float:
    if not cfg.use_method_default_alphas:
        return cfg.alpha
    return BEST_ALPHA_BY_METHOD.get(method_name.lower(), cfg.alpha)


def _attach_method_alpha(method, method_name: str, cfg: TransferExperimentConfig):
    method.alpha = _method_alpha(method_name, cfg)
    return method


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
            methods.append(_attach_method_alpha(method, output_method_name, cfg))
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
            methods.append(_attach_method_alpha(method, "caa_geometry", cfg))
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
<<<<<<< HEAD
            methods.append(method)
        elif normalized_method_name in {"bipo", "opt", "optimized"}:
            if not cfg.bipo_run_dir:
                raise ValueError(
                    f"Method '{method_name}' requires --bipo_run_dir to be specified."
                )
            output_method_name = (
                "bipo_geometry"
                if cfg.bipo_vector_source == "geometry_vectors"
                else "bipo"
            )
            method = BiPOMethod(
                run_dir=cfg.bipo_run_dir,
                layer=cfg.bipo_layer,
                model_name=cfg.model_name,
                method_name=output_method_name,
                steer_position=cfg.bipo_steer_position,
                normalize_vectors=cfg.bipo_normalize_vectors,
                vector_source=cfg.bipo_vector_source,
            )
            methods.append(method)
        elif normalized_method_name in {"bipo_geometry", "bipo_geometry_vectors"}:
            if not cfg.bipo_run_dir:
                raise ValueError(
                    f"Method '{method_name}' requires --bipo_run_dir to be specified."
                )
            method = BiPOMethod(
                run_dir=cfg.bipo_run_dir,
                layer=cfg.bipo_layer,
                model_name=cfg.model_name,
                method_name="bipo_geometry",
                steer_position=cfg.bipo_steer_position,
                normalize_vectors=cfg.bipo_normalize_vectors,
                vector_source="geometry_vectors",
            )
            methods.append(method)
        elif normalized_method_name in {"sparsecaa", "sparse_caa", "sas"}:
            if not cfg.sparsecaa_run_dir:
                raise ValueError(
                    f"Method '{method_name}' requires --sparsecaa_run_dir to be specified."
                )
            method = SparseCAAMethod(
                run_dir=cfg.sparsecaa_run_dir,
                layer=cfg.sparsecaa_layer,
                model_name=cfg.model_name,
                method_name="sparsecaa",
                sae_path=cfg.sparsecaa_sae_path,
                normalize_vectors=cfg.sparsecaa_normalize_vectors,
            )
            methods.append(method)
        elif normalized_method_name in {"qwenscope", "qwen_scope", "qwenscopecaa", "qscope"}:
            if not cfg.qwenscope_run_dir:
                raise ValueError(
                    f"Method '{method_name}' requires --qwenscope_run_dir to be specified."
                )
            method = QwenScopeMethod(
                run_dir=cfg.qwenscope_run_dir,
                layer=cfg.qwenscope_layer,
                model_name=cfg.model_name,
                method_name="qwenscope",
                sae_path=cfg.qwenscope_sae_path,
                vector_source=cfg.qwenscope_vector_source,
                normalize_vectors=cfg.qwenscope_normalize_vectors,
            )
            methods.append(method)
        else:
            raise ValueError(
                f"Unknown method '{method_name}'. "
                "Currently supported: "
                "['caa', 'caa_geometry', 'spherical', 'bipo', 'bipo_geometry', "
                "'sparsecaa', 'qwenscope']."
=======
            methods.append(_attach_method_alpha(method, "spherical", cfg))
        elif normalized_method_name in {"odesteer", "ode"}:
            method = ODESteerMethod(
                config=cfg,
                mode="exact",
                method_name="odesteer",
                model_name=cfg.model_name,
                position="last",
            )
            methods.append(_attach_method_alpha(method, "odesteer", cfg))
        elif normalized_method_name in {"odesteer_vectors", "ode_vectors"}:
            if not cfg.odesteer_run_dir:
                raise ValueError(
                    f"Method '{method_name}' requires --odesteer_run_dir to be specified."
                )
            method = ODESteerMethod(
                config=cfg,
                mode="vectors",
                method_name="odesteer_vectors",
                model_name=cfg.model_name,
                position="last",
            )
            methods.append(_attach_method_alpha(method, "odesteer_vectors", cfg))
        elif normalized_method_name in {
            "llm_steering_opt",
            "llm-steering-opt",
            "llmsteeringopt",
            "steering_opt",
            "steering-opt",
        }:
            if not cfg.llm_steering_opt_run_dir:
                raise ValueError(
                    f"Method '{method_name}' requires "
                    "--llm_steering_opt_run_dir to be specified."
                )
            method = LLMSteeringOptMethod(
                run_dir=cfg.llm_steering_opt_run_dir,
                layer=cfg.llm_steering_opt_layer,
                model_name=cfg.model_name,
                method_name="llm_steering_opt",
                normalize_vectors=cfg.llm_steering_opt_normalize_vectors,
            )
            methods.append(_attach_method_alpha(method, "llm_steering_opt", cfg))
        else:
            raise ValueError(
                f"Unknown method '{method_name}'. "
                "Currently supported: ['caa', 'caa_geometry', 'spherical', "
                "'odesteer', 'odesteer_vectors', 'llm_steering_opt']."
>>>>>>> 897e0f8ebb2f63ae425742d5af16c6246f50e83d
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
    print(f"  method_alphas  : {'best defaults' if cfg.use_method_default_alphas else 'forced global'}")
    print(f"  caa_run_dir    : {cfg.caa_run_dir}")
    print(f"  caa_layer      : {cfg.caa_layer!r}  (None = auto-discover)")
    print(f"  caa_vector_src : {cfg.caa_vector_source}")
    print(f"  spherical_dir  : {cfg.spherical_run_dir}")
    print(f"  spherical_layer: {cfg.spherical_layer!r}  (None = auto-discover)")
    print(f"  spherical_kappa: {cfg.spherical_kappa!r}  (None = run config)")
    print(f"  spherical_beta : {cfg.spherical_beta!r}  (None = run config)")
    print(f"  spherical_pos  : {cfg.spherical_steer_position!r}  (None = run config)")
<<<<<<< HEAD
    print(f"  bipo_run_dir   : {cfg.bipo_run_dir}")
    print(f"  bipo_layer     : {cfg.bipo_layer!r}  (None = auto-discover)")
    print(f"  bipo_pos       : {cfg.bipo_steer_position!r}  (None = run config)")
    print(f"  bipo_vector_src: {cfg.bipo_vector_source}")
    print(f"  bipo_norm_vecs : {cfg.bipo_normalize_vectors}")
    print(f"  sparsecaa_dir  : {cfg.sparsecaa_run_dir}")
    print(f"  sparsecaa_layer: {cfg.sparsecaa_layer!r}  (None = run config)")
    print(f"  sparsecaa_sae  : {cfg.sparsecaa_sae_path!r}  (None = run dir)")
    print(f"  sparsecaa_norm : {cfg.sparsecaa_normalize_vectors}")
    print(f"  qwenscope_dir  : {cfg.qwenscope_run_dir}")
    print(f"  qwenscope_layer: {cfg.qwenscope_layer!r}  (None = run config)")
    print(f"  qwenscope_sae  : {cfg.qwenscope_sae_path!r}  (None = run dir)")
    print(f"  qwenscope_vec  : {cfg.qwenscope_vector_source}")
    print(f"  qwenscope_norm : {cfg.qwenscope_normalize_vectors}")
=======
    print(f"  odesteer_dir   : {cfg.odesteer_run_dir}")
    print(f"  odesteer_layer : {cfg.odesteer_layer!r}")
    print(f"  odesteer_type  : {cfg.odesteer_type}")
    print(f"  odesteer_steps : {cfg.odesteer_steps}")
    print(
        f"  odesteer_kernel: n={cfg.odesteer_n_components}, "
        f"degree={cfg.odesteer_degree}, gamma={cfg.odesteer_gamma}"
    )
    print(f"  llm_opt_dir    : {cfg.llm_steering_opt_run_dir}")
    print(f"  llm_opt_layer  : {cfg.llm_steering_opt_layer!r}  (None = manifest)")
    print(f"  llm_opt_norm   : {cfg.llm_steering_opt_normalize_vectors}")
>>>>>>> 897e0f8ebb2f63ae425742d5af16c6246f50e83d
    print(f"  eval_dataset   : {cfg.eval_dataset_path}")
    print(f"  n_eval_samples : {cfg.n_eval_samples}")
    print(f"  eval_splits    : {cfg.eval_splits if cfg.eval_splits else 'all'}")
    print(f"  eval_fraction  : {cfg.eval_split_fraction}")
    print(f"  seed           : {cfg.seed}")
    print(f"  relations_path : {cfg.relations_path}")
    print(f"  output_dir     : {cfg.output_dir}")
    print(f"  methods        : {cfg.methods}")
    print(f"  force_recompute: {cfg.force_recompute}")
    print(f"  postprocess    : {args.postprocess_existing}")
    print()

    if args.postprocess_existing:
        recompute_metrics_from_saved_results(
            output_dir=Path(cfg.output_dir),
            relations_path=Path(cfg.relations_path),
        )
        return

    # Validate required fields
    if not cfg.methods:
        parser.error("--methods must specify at least one method name.")

    # Build method objects
    methods = _build_methods(cfg)

    # Run
    run_experiment(cfg, methods)


if __name__ == "__main__":
    main()
