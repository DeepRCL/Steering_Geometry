"""
Evaluate steering via the SAE sparse latent space.

The steering hook intercepts the MLP output at layer config.mlp_layer, passes
it through the fine-tuned SAE encoder, adds the sparse persona vector scaled
by alpha, then decodes back to dense MLP space.  The modified MLP output is
returned to the transformer; the layer's own residual connection adds it
normally.  No other part of the forward pass is altered.

Hook mechanic (per forward call at layer 16):
    mlp_out  (batch, seq, 4096)
      ↓  sae.encode
    z        (batch, seq, 16384)   — sparse feature activations
      ↓  z = z + α * persona_vec
    z_steered
      ↓  sae.decode
    mlp_out_steered  (batch, seq, 4096)   ← returned to transformer

Output format is identical to CAA/Geometry/evaluate.py so the results are
directly comparable.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..sae_model import SparseAutoencoder, load_sae
from .config import SparseCAAPipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from .data_loader import (
    ContrastivePair,
    format_eval_prompt,
    load_combined,
    split_dataset,
)


# ──────────────────────────────────────────────────────────────────────────────
# Sparse steering hook
# ──────────────────────────────────────────────────────────────────────────────
def make_sparse_steer_hook(
    sae: SparseAutoencoder,
    persona_vec: torch.Tensor,   # (d_sae,) — the sparse persona direction
    alpha: float,
    d_in: int,
):
    """
    Returns a forward hook that:
      1. Takes the MLP output (batch, seq, d_in).
      2. Encodes into sparse space: z = sae.encode(act).
      3. Adds: z_steered = z + alpha * persona_vec.
      4. Decodes back: mlp_out_steered = sae.decode(z_steered).
      5. Returns mlp_out_steered (same shape as original output).
    """
    def hook(module, inp, output):
        act = output[0] if isinstance(output, tuple) else output
        original_shape = act.shape                       # (batch, seq, d_in)
        dtype = act.dtype
        flat = act.reshape(-1, d_in).to(torch.float32)  # (batch*seq, d_in)

        # Encode → sparse space
        z = sae.encode(flat)                             # (batch*seq, d_sae)
        # Add persona direction in sparse space
        pv = persona_vec.to(device=z.device, dtype=z.dtype)
        z_steered = z + alpha * pv
        # Decode → back to dense MLP space
        recon = sae.decode(z_steered)                    # (batch*seq, d_in)
        recon = recon.reshape(original_shape).to(dtype)

        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon

    return hook


# ──────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────────────
def _score_pair(
    model,
    tokens: List[int],
    a_token_id: int,
    b_token_id: int,
    pos_is_a: bool,
    device: torch.device,
) -> Dict[str, Any]:
    input_ids = torch.tensor([tokens]).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits

    last_logits = logits[0, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    prob_a = probs[a_token_id].item()
    prob_b = probs[b_token_id].item()
    prob_pos = prob_a if pos_is_a else prob_b
    prob_neg = prob_b if pos_is_a else prob_a
    margin = prob_pos - prob_neg
    chose_a = prob_a > prob_b

    return {
        "prob_a": prob_a,
        "prob_b": prob_b,
        "prob_positive": prob_pos,
        "prob_negative": prob_neg,
        "positive_margin": margin,
        "chose_a": chose_a,
        "pos_is_a": pos_is_a,
        "is_correct": (chose_a == pos_is_a),
    }


def _format_generation_prompt(pair: ContrastivePair, tokenizer, is_instruct: bool) -> str:
    if is_instruct:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": pair.question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return pair.question + "\nResponse:"


def _mean_completion_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
) -> float:
    """Mean per-token log-probability computed autoregressively."""
    completion_text = " " + completion.lstrip()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    if not completion_ids:
        return 0.0

    prefix_ids = list(prompt_ids)
    token_logprobs = []
    with torch.no_grad():
        for token_id in completion_ids:
            input_ids = torch.tensor([prefix_ids]).to(device)
            logits = model(input_ids).logits[0, -1, :]
            token_logprobs.append(F.log_softmax(logits, dim=-1)[token_id].item())
            prefix_ids.append(token_id)

    return float(np.mean(token_logprobs)) if token_logprobs else 0.0


def _score_full_logprob(
    model,
    tokenizer,
    pair: ContrastivePair,
    is_instruct: bool,
    device: torch.device,
) -> Dict[str, Any]:
    prompt = _format_generation_prompt(pair, tokenizer, is_instruct)
    lp_pos = _mean_completion_logprob(model, tokenizer, prompt, pair.positive_answer, device)
    lp_neg = _mean_completion_logprob(model, tokenizer, prompt, pair.negative_answer, device)
    margin = lp_pos - lp_neg
    return {
        "mean_logprob_positive": lp_pos,
        "mean_logprob_negative": lp_neg,
        "logprob_positive_margin": margin,
        "is_correct": lp_pos > lp_neg,
    }


def _summarize(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not details:
        return {"accuracy": 0.0, "num_eval": 0}
    return {
        "accuracy": float(np.mean([d["is_correct"] for d in details])),
        "num_eval": len(details),
        "mean_prob_positive": float(np.mean([d["prob_positive"] for d in details])),
        "mean_prob_negative": float(np.mean([d["prob_negative"] for d in details])),
        "mean_positive_margin": float(np.mean([d["positive_margin"] for d in details])),
        "details": details,
    }


def _summarize_full_logprob(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not details:
        return {"accuracy": 0.0, "num_eval": 0}
    return {
        "accuracy": float(np.mean([d["is_correct"] for d in details])),
        "num_eval": len(details),
        "mean_logprob_positive": float(np.mean([d["mean_logprob_positive"] for d in details])),
        "mean_logprob_negative": float(np.mean([d["mean_logprob_negative"] for d in details])),
        "mean_logprob_positive_margin": float(np.mean([d["logprob_positive_margin"] for d in details])),
        "details": details,
    }


def _metric_summary(
    results_all: Dict,
    config: SparseCAAPipelineConfig,
    baseline_key: str,
    steered_key: str,
    metric_name: str,
) -> Dict[str, Any]:
    value_order = [v for v in SCHWARTZ_CIRCUMPLEX_ORDER if v in results_all]
    alpha_labels = [str(a) for a in config.alpha_values]
    baseline_acc = np.array([results_all[v][baseline_key]["accuracy"] for v in value_order])
    steered_acc = np.array(
        [[results_all[v][steered_key][a]["accuracy"] for a in alpha_labels] for v in value_order]
    )
    acc_gain = steered_acc - baseline_acc[:, None]
    mean_steered = steered_acc.mean(axis=0)
    mean_gain = acc_gain.mean(axis=0)
    best_idx = int(np.argmax(mean_gain))

    return {
        "metric": metric_name,
        "layer": config.mlp_layer,
        "alpha_values": config.alpha_values,
        "mean_baseline_accuracy": float(baseline_acc.mean()),
        "mean_steered_accuracy_by_alpha": {
            alpha: float(mean_steered[idx]) for idx, alpha in enumerate(alpha_labels)
        },
        "mean_accuracy_gain_by_alpha": {
            alpha: float(mean_gain[idx]) for idx, alpha in enumerate(alpha_labels)
        },
        "best_overall_alpha_by_mean_accuracy_gain": {
            "alpha": alpha_labels[best_idx],
            "mean_accuracy_gain_vs_baseline": float(mean_gain[best_idx]),
            "mean_accuracy_at_alpha": float(mean_steered[best_idx]),
            "mean_baseline_accuracy": float(baseline_acc.mean()),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def _save_eval_plots(results_all: Dict, config: SparseCAAPipelineConfig, out_dir: str):
    value_order = [v for v in SCHWARTZ_CIRCUMPLEX_ORDER if v in results_all]
    alpha_labels = [str(a) for a in config.alpha_values]

    if not value_order:
        return

    baseline_acc = np.array([results_all[v]["baseline"]["accuracy"] for v in value_order])
    steered_acc = np.array(
        [[results_all[v]["steered"][a]["accuracy"] for a in alpha_labels] for v in value_order]
    )
    acc_gain = steered_acc - baseline_acc[:, None]
    mean_baseline = float(baseline_acc.mean())
    mean_steered = steered_acc.mean(axis=0)
    mean_gain = acc_gain.mean(axis=0)
    alpha_vals = config.alpha_values

    # Accuracy curves
    plt.figure(figsize=(8, 5))
    plt.axhline(mean_baseline, color="gray", linestyle="--", linewidth=2, label="Baseline")
    plt.plot(alpha_vals, mean_steered, marker="o", linewidth=2, label="Steered (sparse SAE)")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Accuracy")
    plt.title("Baseline vs Sparse-SAE Steering Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "baseline_vs_steered_accuracy.png"), dpi=200)
    plt.close()

    # Accuracy gain
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_vals, mean_gain, marker="o", linewidth=2)
    plt.axhline(0.0, color="gray", linestyle="--")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Accuracy Gain vs Baseline")
    plt.title("Accuracy Gain from Sparse-SAE Steering")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_gain_vs_baseline.png"), dpi=200)
    plt.close()

    # Per-value heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        acc_gain,
        xticklabels=alpha_labels,
        yticklabels=value_order,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
    )
    plt.xlabel("Alpha")
    plt.ylabel("Value")
    plt.title("Accuracy Gain by Value and Alpha (Sparse-SAE Steering)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_gain_heatmap.png"), dpi=200)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_sparse_steering(
    config: SparseCAAPipelineConfig,
    vectors: Dict[str, torch.Tensor],
    sae: Optional[SparseAutoencoder] = None,
) -> Dict:
    """
    Evaluate sparse-SAE steering on the held-out eval split.

    Args:
        config:  Pipeline configuration.
        vectors: {value → (d_sae,) sparse persona tensor}
        sae:     Optionally pass an already-loaded SAE; otherwise loaded from disk.

    Returns the full results dict and saves eval_results.json + plots.
    """
    out_dir = config.subdir("evaluation")
    out_path = os.path.join(out_dir, "eval_results.json")

    if os.path.exists(out_path):
        print(f"Evaluation results already exist at {out_path} — loading.")
        with open(out_path) as f:
            return json.load(f)

    # ── Load SAE ──────────────────────────────────────────────────────────────
    if sae is None:
        ckpt = (
            config.finetuned_sae_path
            if os.path.exists(config.finetuned_sae_path)
            else config.sae_checkpoint
        )
        sae = load_sae(ckpt, config.d_in, config.d_sae, device="cpu")
    # Keep SAE on CPU; the hook will move tensors as needed

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {config.model_name}")
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if config.device == "auto" else config.device,
    )
    model.eval()

    name_lower = config.model_name.lower()
    is_instruct = "base" not in name_lower and "pt" not in name_lower

    mlp_module = model.model.layers[config.mlp_layer].mlp

    # Move SAE to same device as model for hook efficiency
    sae_device = device
    sae = sae.to(sae_device).eval()

    # ── Load eval data ────────────────────────────────────────────────────────
    df = load_combined(config)
    _, eval_data = split_dataset(df, config)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    results_all: Dict = {}
    print(f"Evaluating with alphas: {config.alpha_values}")

    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        eval_pairs: List[ContrastivePair] = eval_data.get(val, [])
        if not eval_pairs:
            continue

        persona_vec = vectors[val].to(sae_device)

        # Baseline (no steering)
        baseline_details = []
        baseline_logprob_details = []
        for pair in eval_pairs:
            tokens, a_id, b_id = format_eval_prompt(pair, tokenizer, is_instruct)
            result = _score_pair(model, tokens, a_id, b_id, pair.pos_is_a, device)
            result["sample_id"] = pair.sample_id
            baseline_details.append(result)

            lp_result = _score_full_logprob(model, tokenizer, pair, is_instruct, device)
            lp_result["sample_id"] = pair.sample_id
            baseline_logprob_details.append(lp_result)
        baseline_summary = _summarize(baseline_details)
        baseline_logprob_summary = _summarize_full_logprob(baseline_logprob_details)

        steered_results: Dict = {}
        steered_logprob_results: Dict = {}

        for alpha in config.alpha_values:
            hook_fn = make_sparse_steer_hook(sae, persona_vec, alpha, config.d_in)
            handle = mlp_module.register_forward_hook(hook_fn)

            try:
                print(f"  {val} (alpha={alpha}) …")
                steered_details = []
                steered_logprob_details = []
                for pair, bsl, bsl_lp in zip(eval_pairs, baseline_details, baseline_logprob_details):
                    tokens, a_id, b_id = format_eval_prompt(pair, tokenizer, is_instruct)
                    detail = _score_pair(model, tokens, a_id, b_id, pair.pos_is_a, device)
                    detail["sample_id"] = pair.sample_id
                    # Delta vs baseline
                    detail["baseline_prob_positive"] = bsl["prob_positive"]
                    detail["baseline_is_correct"] = bsl["is_correct"]
                    detail["delta_prob_positive"] = (
                        detail["prob_positive"] - bsl["prob_positive"]
                    )
                    detail["delta_positive_margin"] = (
                        detail["positive_margin"] - bsl["positive_margin"]
                    )
                    detail["became_correct"] = (
                        not bsl["is_correct"] and detail["is_correct"]
                    )
                    detail["became_incorrect"] = (
                        bsl["is_correct"] and not detail["is_correct"]
                    )
                    steered_details.append(detail)

                    lp_detail = _score_full_logprob(model, tokenizer, pair, is_instruct, device)
                    lp_detail["sample_id"] = pair.sample_id
                    lp_detail["baseline_mean_logprob_positive"] = bsl_lp["mean_logprob_positive"]
                    lp_detail["baseline_mean_logprob_negative"] = bsl_lp["mean_logprob_negative"]
                    lp_detail["baseline_logprob_positive_margin"] = bsl_lp["logprob_positive_margin"]
                    lp_detail["baseline_is_correct"] = bsl_lp["is_correct"]
                    lp_detail["delta_logprob_positive"] = (
                        lp_detail["mean_logprob_positive"] - bsl_lp["mean_logprob_positive"]
                    )
                    lp_detail["delta_logprob_negative"] = (
                        lp_detail["mean_logprob_negative"] - bsl_lp["mean_logprob_negative"]
                    )
                    lp_detail["delta_logprob_positive_margin"] = (
                        lp_detail["logprob_positive_margin"] - bsl_lp["logprob_positive_margin"]
                    )
                    lp_detail["became_correct"] = (
                        not bsl_lp["is_correct"] and lp_detail["is_correct"]
                    )
                    lp_detail["became_incorrect"] = (
                        bsl_lp["is_correct"] and not lp_detail["is_correct"]
                    )
                    steered_logprob_details.append(lp_detail)
            finally:
                handle.remove()

            steered_summary = _summarize(steered_details)
            steered_summary["accuracy_gain_vs_baseline"] = (
                steered_summary["accuracy"] - baseline_summary["accuracy"]
            )
            steered_summary["mean_prob_positive_gain_vs_baseline"] = (
                steered_summary["mean_prob_positive"]
                - baseline_summary["mean_prob_positive"]
            )
            steered_summary["mean_positive_margin_gain_vs_baseline"] = (
                steered_summary["mean_positive_margin"]
                - baseline_summary["mean_positive_margin"]
            )
            steered_results[str(alpha)] = steered_summary

            steered_logprob_summary = _summarize_full_logprob(steered_logprob_details)
            steered_logprob_summary["accuracy_gain_vs_baseline"] = (
                steered_logprob_summary["accuracy"] - baseline_logprob_summary["accuracy"]
            )
            steered_logprob_summary["mean_logprob_positive_gain_vs_baseline"] = (
                steered_logprob_summary["mean_logprob_positive"]
                - baseline_logprob_summary["mean_logprob_positive"]
            )
            steered_logprob_summary["mean_logprob_positive_margin_gain_vs_baseline"] = (
                steered_logprob_summary["mean_logprob_positive_margin"]
                - baseline_logprob_summary["mean_logprob_positive_margin"]
            )
            steered_logprob_results[str(alpha)] = steered_logprob_summary

        results_all[val] = {
            "baseline": baseline_summary,
            "steered": steered_results,
            "baseline_full_logprob": baseline_logprob_summary,
            "steered_full_logprob": steered_logprob_results,
        }

    # ── Save results ──────────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)

    with open(os.path.join(out_dir, "eval_results_full_logprob.json"), "w") as f:
        json.dump(
            {
                val: {
                    "baseline": payload["baseline_full_logprob"],
                    "steered": payload["steered_full_logprob"],
                }
                for val, payload in results_all.items()
            },
            f,
            indent=2,
        )

    ab_summary = _metric_summary(results_all, config, "baseline", "steered", "ab_next_token")
    full_logprob_summary = _metric_summary(
        results_all,
        config,
        "baseline_full_logprob",
        "steered_full_logprob",
        "full_answer_mean_logprob",
    )
    ab_summary["full_answer_mean_logprob"] = full_logprob_summary
    with open(os.path.join(out_dir, "evaluation_summary.json"), "w") as f:
        json.dump(ab_summary, f, indent=2)
    with open(os.path.join(out_dir, "evaluation_summary_full_logprob.json"), "w") as f:
        json.dump(full_logprob_summary, f, indent=2)

    _save_eval_plots(results_all, config, out_dir)
    print(f"Evaluation complete. Results → {out_dir}")
    return results_all
