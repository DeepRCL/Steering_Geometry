"""
Standalone CLI for running cross-value transfer with saved COLD-Steer vectors.

This keeps the normal cross-value-transfer runner untouched while using the
same core ``run_experiment`` implementation and output format.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.cross_value_transfer.cold_steer_method import ColdSteerMethod
from experiments.cross_value_transfer.config import TransferExperimentConfig
from experiments.cross_value_transfer.run_transfer_experiment import run_experiment


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_cold_steer_transfer",
        description="Cross-Value Steering Transfer experiment for COLD-Steer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--cold_steer_run_dir", type=str, required=True)
    p.add_argument("--cold_steer_layer", type=int, default=None)
    p.add_argument("--cold_steer_position", choices=["all", "last"], default="all")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--eval_dataset", type=str, default="CAA/value_data/final_dataset_200.csv")
    p.add_argument("--eval_split_fraction", type=float, default=0.9)
    p.add_argument("--eval_splits", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--relations_path", type=str, default="CAA/value_data/schwartz_relations-new.json")
    p.add_argument("--output_dir", type=str, default="experiments/cross_value_transfer/outputs_final200")
    p.add_argument("--force_recompute", action="store_true", default=False)
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)

    eval_splits = ["validation", "test"]
    if args.eval_splits is not None:
        if args.eval_splits.strip().lower() == "all":
            eval_splits = None
        else:
            eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]

    cfg = TransferExperimentConfig(
        model_name=args.model_name,
        device=args.device,
        eval_dataset_path=args.eval_dataset,
        eval_splits=eval_splits,
        eval_split_fraction=args.eval_split_fraction,
        seed=args.seed,
        relations_path=args.relations_path,
        output_dir=args.output_dir,
        methods=["cold_steer"],
        force_recompute=args.force_recompute,
    ).resolve_paths(_PROJECT_ROOT)

    method = ColdSteerMethod(
        run_dir=Path(args.cold_steer_run_dir),
        layer=args.cold_steer_layer,
        model_name=cfg.model_name,
        method_name="cold_steer",
        position=args.cold_steer_position,
    )
    method.alpha = args.alpha if args.alpha is not None else method.recommended_alpha

    print("ColdSteer cross-value transfer:")
    print(f"  model_name     : {cfg.model_name!r}")
    print(f"  cold_run_dir   : {Path(args.cold_steer_run_dir).resolve()}")
    print(f"  cold_layer     : {method.layer}")
    print(f"  cold_position  : {args.cold_steer_position}")
    print(f"  alpha          : {method.alpha}")
    print(f"  eval_dataset   : {cfg.eval_dataset_path}")
    print(f"  eval_fraction  : {cfg.eval_split_fraction}")
    print(f"  output_dir     : {cfg.output_dir}")
    print()

    run_experiment(cfg, [method])


if __name__ == "__main__":
    main()
