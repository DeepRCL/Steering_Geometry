"""
Schwartz value steering pipeline using ODESteer's native API.

Uses HuggingFaceLM for model management, fit_steer_model() for fitting,
extract_prompt_eos_activations() for activation extraction, and
compute_answer_prob() for evaluation — matching the ODESteer repo exactly.

Geometry vectors are mean ODE displacements on positive training
activations: mean(steer(pos_X, T) - pos_X), using the same T and ODE
steps as inference steering.

Usage:
    python scripts/schwartz/schwartz_pipeline.py \\
        --model Qwen2.5-7B-Base --steer_type ODESteer --layer_idx 13 \\
        --T 5.0 --train_ratio 0.1
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

# Path setup: scripts/schwartz and odesteer src must precede everything else so
# ``import config`` resolves to ``scripts/schwartz/config.py`` (not the repo
# top-level ``config.py``), and so ``schwartz_eval`` resolves locally.
_SCRIPT_DIR = Path(__file__).resolve().parent
_SCHWARTZ_DIR = str(_SCRIPT_DIR)
_ODESTEER_SRC = str(_SCRIPT_DIR.parents[2] / "src")

for _path in (_SCHWARTZ_DIR, _ODESTEER_SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from odesteer.lm import HuggingFaceLM
from odesteer.steer import get_steer_model
from odesteer.utils import get_project_dir

import config  # noqa: E402 — scripts/schwartz/config.py
import geometry as geometry_module  # noqa: E402
import schwartz_eval  # noqa: E402 — scripts/schwartz/schwartz_eval.py (self-contained)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Schwartz Value Steering Pipeline (ODESteer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B-Base",
                        help="Model name (short name from _config.py or full HF path)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "bfloat16", "float32"])

    # Dataset
    parser.add_argument("--dataset_path", type=str,
                        default=str(get_project_dir() / "data" / "final_dataset_v3.csv"))
    parser.add_argument("--relations_path", type=str,
                        default=str(get_project_dir() / "data" / "schwartz_relations.json"))
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=10)

    # Steering method
    parser.add_argument("--steer_type", type=str, default="ODESteer",
                        choices=["ODESteer", "StepODESteer", "CAA", "RepE", "ITI", "NoSteer"])
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--n_components", type=int, default=8000)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--coef0", type=float, default=1.0)
    parser.add_argument("--lin_clf_type", type=str, default="lr")
    parser.add_argument("--T", type=float, default=5.0, help="Steering strength")

    # Layer
    parser.add_argument("--layer_idx", type=int, default=13)
    parser.add_argument("--layer_sweep", action="store_true",
                        help="Run layer sweep instead of using fixed layer_idx")
    parser.add_argument("--layer_candidates", type=int, nargs="+", default=None)
    parser.add_argument("--layer_sweep_n_samples", type=int, default=20)

    # Evaluation
    parser.add_argument("--n_eval_samples", type=int)
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="full_logprob",
        choices=["full_logprob", "ab_next_token"],
        help="Steering eval: full-answer logprob or CAA-style A/B next-token",
    )
    parser.add_argument("--skip_eval", action="store_true")

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def get_steer_kwargs(args) -> dict:
    """Build kwargs for get_steer_model / HuggingFaceLM."""
    if args.steer_type in ("ODESteer", "RFFODESteer"):
        return dict(solver=args.solver, steps=args.steps,
                    n_components=args.n_components, degree=args.degree,
                    gamma=args.gamma, coef0=args.coef0, lin_clf_type=args.lin_clf_type)
    elif args.steer_type in ("StepODESteer", "RFFStepODESteer"):
        return dict(n_components=args.n_components, degree=args.degree,
                    gamma=args.gamma, coef0=args.coef0, lin_clf_type=args.lin_clf_type)
    else:
        return {}


# ─── Activation Extraction ──────────────────────────────────────────────────

@torch.no_grad()
def extract_activations(
    hf_lm: HuggingFaceLM,
    rows: List[dict],
    layer_idx: int,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract pos/neg activations using HuggingFaceLM.extract_prompt_eos_activations().

    For instruct models, the question is wrapped in the tokenizer's chat
    template before the answer is appended so the activation distribution
    matches the CAA-aligned A/B eval prompt (which also uses the chat
    template). For base models we fall back to plain ``Q: ...\\nA: ...``.
    """
    if n_samples is not None and n_samples < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, n_samples)

    tokenizer = hf_lm.tokenizer
    pos_prompts = [
        config.format_qa_prompt(r["question"], r["positive_answer"], tokenizer=tokenizer)
        for r in rows
    ]
    neg_prompts = [
        config.format_qa_prompt(r["question"], r["negative_answer"], tokenizer=tokenizer)
        for r in rows
    ]

    pos_X = hf_lm.extract_prompt_eos_activations(pos_prompts, layer_idx).cpu().float()
    neg_X = hf_lm.extract_prompt_eos_activations(neg_prompts, layer_idx).cpu().float()

    return pos_X, neg_X


# ─── Layer Selection ────────────────────────────────────────────────────────

def select_layer(hf_lm, train_rows, values, args, verbose=True) -> Tuple[int, dict]:
    """Layer sweep using classifier accuracy on fitted steer models."""
    if args.layer_candidates:
        candidates = args.layer_candidates
    else:
        n_layers = hf_lm.model.config.num_hidden_layers
        n_cand = 12
        start = max(1, int(n_layers * 0.15))
        end = int(n_layers * 0.85)
        step = max(1, (end - start) // (n_cand - 1))
        candidates = list(range(start, end + 1, step))[:n_cand]

    if verbose:
        print(f"Layer sweep over candidates: {candidates}")

    sweep_values = [v for v in values if v in config.SCHWARTZ_CIRCUMPLEX_ORDER]
    steer_kwargs = get_steer_kwargs(args)
    mean_scores = {}

    for layer in candidates:
        layer_accs = {}
        for value in tqdm(sweep_values, desc=f"Layer {layer}", leave=False):
            value_rows = config.get_rows_for_value(train_rows, value)
            if len(value_rows) < 2:
                continue
            try:
                pos_X, neg_X = extract_activations(
                    hf_lm, value_rows, layer,
                    n_samples=args.layer_sweep_n_samples, seed=args.seed,
                )
                steer = get_steer_model(args.steer_type, **steer_kwargs)
                steer.fit(pos_X, neg_X)
                # Measure accuracy
                all_X = torch.cat([pos_X, neg_X])
                all_y = torch.cat([torch.ones(len(pos_X)), torch.zeros(len(neg_X))])
                if hasattr(steer, 'clf') and hasattr(steer.clf, 'predict'):
                    preds = steer.clf.predict(all_X)
                    acc = float((preds == all_y).float().mean())
                else:
                    acc = float(torch.norm(pos_X.mean(0) - neg_X.mean(0)))
                layer_accs[value] = acc
            except Exception as e:
                if verbose:
                    print(f"  WARNING: layer {layer}, {value}: {e}")

        scores = list(layer_accs.values())
        mean_scores[layer] = float(np.mean(scores)) if scores else 0.0
        if verbose:
            print(f"  Layer {layer}: mean_score = {mean_scores[layer]:.4f}")

    best = max(candidates, key=lambda l: mean_scores.get(l, 0.0))
    if verbose:
        print(f"\n  Best layer: {best} (score={mean_scores[best]:.4f})\n")
    return best, {"candidates": candidates, "scores": mean_scores, "best_layer": best}


# ─── Value vector extraction ────────────────────────────────────────────────

@torch.no_grad()
def extract_displacement_vector(
    steer: Any,
    pos_X: torch.Tensor,
    T: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ODE steering displacement on positive training activations.

    Runs the same steer() used at inference (full ODE integration for
    ODESteer, single step for StepODESteer). Returns (mean displacement,
    per-sample displacement norms).
    """
    displacements = steer.steer(pos_X, T=T) - pos_X
    return displacements.mean(dim=0), displacements.norm(dim=-1)


# ─── Training ───────────────────────────────────────────────────────────────

def train_all_values(
    hf_lm: HuggingFaceLM,
    train_rows: List[dict],
    values: List[str],
    layer_idx: int,
    args,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    For each value:
        1. Extract pos/neg activations using HuggingFaceLM
        2. Fit an ODESteer model using those activations
        3. Extract representative vector = mean(steer(pos_X, T) - pos_X)
    """
    steer_kwargs = get_steer_kwargs(args)
    vectors = {}
    steer_models = {}
    train_info = {}

    for value in values:
        value_rows = config.get_rows_for_value(train_rows, value)
        if len(value_rows) < 2:
            if verbose:
                print(f"  {value}: skipping (< 2 rows)")
            continue

        if verbose:
            print(f"  {value} ({len(value_rows)} rows):", end=" ")

        t0 = time.time()
        pos_X, neg_X = extract_activations(hf_lm, value_rows, layer_idx, seed=args.seed)

        if len(pos_X) < 2 or len(neg_X) < 2:
            if verbose:
                print("skipped (not enough activations)")
            continue

        steer = get_steer_model(args.steer_type, **steer_kwargs)
        steer.fit(pos_X, neg_X)
        steer_models[value] = steer

        rep_vec, per_sample_disp = extract_displacement_vector(steer, pos_X, args.T)
        vectors[value] = rep_vec.detach().cpu().float()

        elapsed = time.time() - t0
        acc = -1.0
        if hasattr(steer, 'clf') and hasattr(steer.clf, 'predict'):
            all_X = torch.cat([pos_X, neg_X])
            all_y = torch.cat([torch.ones(len(pos_X)), torch.zeros(len(neg_X))])
            preds = steer.clf.predict(all_X)
            acc = float((preds == all_y).float().mean())

        train_info[value] = {
            "n_pos": len(pos_X), "n_neg": len(neg_X),
            "clf_accuracy": round(acc, 4),
            "displacement_norm": round(float(rep_vec.norm()), 4),
            "mean_per_sample_displacement_norm": round(float(per_sample_disp.mean()), 4),
            "T": args.T,
            "time_sec": round(elapsed, 2),
        }
        if verbose:
            print(
                f"✓ {elapsed:.1f}s | clf_acc={acc:.3f} | "
                f"disp_norm={rep_vec.norm():.4f}"
            )

    if verbose:
        print(f"\n  Fitted {len(vectors)}/{len(values)} models\n")
    return vectors, steer_models, train_info


# ─── Evaluation ──────────────────────────────────────────────────────────────
# All evaluation logic lives in ``schwartz_eval.py`` (single self-contained
# file). Re-exported here as a thin wrapper so call-sites stay unchanged.


def evaluate_steering(
    hf_lm: HuggingFaceLM,
    steer_models: Dict[str, Any],
    val_rows: List[dict],
    values: List[str],
    args,
    verbose: bool = True,
) -> Dict[str, Any]:
    """See ``schwartz_eval.evaluate_steering`` for the implementation."""
    return schwartz_eval.evaluate_steering(
        hf_lm,
        steer_models,
        val_rows,
        values,
        args,
        get_rows_for_value=config.get_rows_for_value,
        verbose=verbose,
    )


# ─── Save Helpers ────────────────────────────────────────────────────────────

def save_vectors(vectors, out_dir, layer_idx, args):
    vec_dir = os.path.join(out_dir, "vectors")
    os.makedirs(vec_dir, exist_ok=True)
    manifest = {
        "vector_type": "ode_displacement",
        "definition": "mean(steer(pos_X, T) - pos_X)",
        "T": args.T,
        "steps": getattr(args, "steps", None),
        "steer_type": args.steer_type,
        "values": {},
    }
    for value, vector in vectors.items():
        safe = value.lower().replace(": ", "_").replace(":", "_").replace(" ", "_").replace("-", "_")
        torch.save(vector.detach().cpu(), os.path.join(vec_dir, f"{safe}.pt"))
        manifest["values"][value] = {
            "file": f"{safe}.pt",
            "layer": layer_idx,
            "norm": round(vector.norm().item(), 4),
        }
    with open(os.path.join(vec_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(vectors)} displacement vectors to {vec_dir}/")


def _default_schwartz_output_dir(args, layer_idx: int) -> str:
    """Default results path under ``results/schwartz/{model}/…/seed_{seed}``."""
    eval_slug = args.eval_metric.replace("_", "-")
    run_name = (
        f"{args.steer_type}-layer_{layer_idx}-T_{args.T}-train_{args.train_ratio}-"
        f"eval_{eval_slug}"
    )
    return str(
        get_project_dir() / "results" / "schwartz" / args.model / run_name / f"seed_{args.seed}"
    )


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    verbose = not args.quiet
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    steer_kwargs = get_steer_kwargs(args)

    # Output directory
    if args.output_dir is None:
        args.output_dir = _default_schwartz_output_dir(args, args.layer_idx)
    os.makedirs(args.output_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("  Schwartz Value Steering Pipeline (ODESteer)")
        print("=" * 60)

    # 1. Load data
    if verbose:
        print(f"\nLoading dataset: {args.dataset_path}")
    all_rows = config.load_dataset(args.dataset_path)
    values = config.get_unique_values(all_rows)
    train_rows, val_rows = config.stratified_split(all_rows, args.train_ratio, args.seed)
    if verbose:
        config.print_split_summary(train_rows, val_rows, values)

    # 2. Load model via HuggingFaceLM (ODESteer's official wrapper)
    if verbose:
        print(f"Loading model: {args.model} (dtype={args.dtype})")
        print(f"  steer_type={args.steer_type}, layer_idx={args.layer_idx}")

    hf_lm = HuggingFaceLM(
        args.model,
        steer_name=None,  # We'll fit models per-value manually
        steer_model_kwargs={},
        steer_layer_idx=args.layer_idx,
        device="auto",
        dtype=dtype,
    )

    if torch.cuda.is_available():
        # mem_get_info returns (free_memory, total_memory) in bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1024**3
        total_gb = total_bytes / 1024**3
        
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Free Memory: {free_gb:.2f} GB")
        print(f"Total Memory: {total_gb:.2f} GB")
    else:
        print("CUDA is not available.")

    
    # Ensure model is in float32 to avoid BFloat16 errors in ODESteer kernels
    if dtype == torch.float32:
        hf_lm.model = hf_lm.model.to(torch.float32)

    if verbose:
        n_layers = hf_lm.model.config.num_hidden_layers
        d_model = hf_lm.model.config.hidden_size
        print(f"  Loaded: {n_layers} layers, d_model={d_model}\n")

    # 3. Layer selection (optional)
    layer_idx = args.layer_idx
    if args.layer_sweep:
        layer_idx, sweep_info = select_layer(hf_lm, train_rows, values, args, verbose)
        hf_lm.steer_layer_idx = layer_idx
        # Update output dir with selected layer
        args.output_dir = _default_schwartz_output_dir(args, layer_idx)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "layer_sweep.json"), "w") as f:
            json.dump(sweep_info, f, indent=2)

    # 4. Train all values
    if verbose:
        print(f"Training {args.steer_type} models at layer {layer_idx}")
    vectors, steer_models, train_info = train_all_values(
        hf_lm, train_rows, values, layer_idx, args, verbose
    )

    # Save vectors
    save_vectors(vectors, args.output_dir, layer_idx, args)

    # Save training info
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(
            {
                "vector_type": "ode_displacement",
                "T": args.T,
                "steps": args.steps,
                "steer_type": args.steer_type,
                "per_value": train_info,
            },
            f,
            indent=2,
        )

    # 5. Evaluate steering
    if not args.skip_eval and steer_models:
        eval_metrics = evaluate_steering(hf_lm, steer_models, val_rows, values, args, verbose)
        with open(os.path.join(args.output_dir, "steering_eval_metrics.json"), "w") as f:
            json.dump(eval_metrics, f, indent=2)

    # 6. Geometry analysis
    geo_metrics = geometry_module.analyze_geometry(
        vectors=vectors,
        relations_path=args.relations_path,
        output_dir=args.output_dir,
        random_seed=args.seed,
        verbose=verbose,
    )

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    if verbose:
        print("=" * 60)
        print("  Pipeline complete!")
        print(f"  Results saved to: {args.output_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
