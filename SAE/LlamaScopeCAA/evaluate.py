"""
Evaluate steering via the Llama-Scope SAE sparse latent space.

The steering hook intercepts the residual stream at config.layer, passes it
through the SAE encoder (TopK), adds the persona vector scaled by alpha,
then decodes back to the dense residual space.  The modified residual replaces
the layer output; all subsequent transformer layers see the steered hidden state.

Pre-TopK hook mechanic (config.use_pre_topk_personas=True, the default):
    residual       (batch, seq, 4096)
      ↓  sae.pre_encode  [dense, linear]
    pre            (batch, seq, 32768)

    [optional Δ correction — config.use_delta_correction=True]
      ↓  z_u = TopK(pre, k=50)
      ↓  act_recon = sae.decode(z_u)
      delta = residual - act_recon      (batch, seq, 4096)

      ↓  pre_steered = pre + α · persona_vec
      ↓  z_steered = TopK(pre_steered, k=50)
      ↓  recon = sae.decode(z_steered)
    [if delta correction]
      recon = recon + delta             ← SAE reconstruction error corrected
    residual_steered  (batch, seq, 4096)  ← returned as layer output

The Δ correction adds back the portion of the original residual that the SAE
cannot reconstruct.  Without it the reconstruction error is injected directly
into the transformer's hidden state on every steered forward pass.

Output format is compatible with CAA/Geometry/evaluate.py so results are
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

from .config import LlamaScopePipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from .data_loader import (
    ContrastivePair,
    format_eval_prompt,
    load_steering_split,
)
from .topk_sae_model import TopKSparseAutoencoder, get_or_download_sae


# ──────────────────────────────────────────────────────────────────────────────
# Steering hooks
# ──────────────────────────────────────────────────────────────────────────────
def make_pre_topk_steer_hook(
    sae: TopKSparseAutoencoder,
    persona_vec: torch.Tensor,          # (d_sae,) — pre-TopK persona direction
    alpha: float,
    d_in: int,
    use_delta_correction: bool = True,
):
    """
    Pre-TopK steering hook (recommended, matches persona vector computation).

    Injects the persona direction into the pre-activation space BEFORE the TopK
    gate, so the value signal biases which 50 features are selected:

        residual  (batch, seq, d_in)
          ↓  sae.pre_encode  [dense, continuous]
        pre       (batch, seq, d_sae)

        [if use_delta_correction]
          z_u = TopK(pre, k=50)
          act_recon = sae.decode(z_u)
          delta = flat - act_recon          ← unsteered reconstruction error

          ↓  pre_steered = pre + α · persona_vec
          ↓  TopK(pre_steered, k=50)
        z_steered (batch, seq, d_sae)  — 50 active, value-biased
          ↓  sae.decode
        recon     (batch, seq, d_in)
        [if use_delta_correction]
          recon = recon + delta             ← add reconstruction error back
        residual_steered  (batch, seq, d_in)   ← replaces layer output

    Args:
        use_delta_correction: When True (default), compute Δ = act − decode(encode(act))
            from the unsteered pass and add it to the steered reconstruction.
            This cancels SAE reconstruction error that would otherwise be injected
            into the residual stream on every forward pass.

    Use this when config.use_pre_topk_personas=True (the default).
    """
    def hook(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        original_shape = hidden.shape                          # (batch, seq, d_in)
        dtype = hidden.dtype
        flat = hidden.reshape(-1, d_in).to(torch.float32)     # (batch*seq, d_in)

        # Compute dense pre-activations (before TopK)
        pre = sae.pre_encode(flat)                             # (batch*seq, d_sae)

        # Δ correction: compute unsteered reconstruction error from this pass
        if use_delta_correction:
            z_unsteered = sae.sparsify(pre)
            act_recon = sae.decode(z_unsteered)                # (batch*seq, d_in)
            delta = flat - act_recon                           # (batch*seq, d_in)

        # Inject persona direction in pre-activation space
        pv = persona_vec.to(device=pre.device, dtype=pre.dtype)
        pre_steered = pre + alpha * pv
        # Re-apply the SAE sparsifier to the steered pre-activations.
        z_steered = sae.sparsify(pre_steered)
        # Decode → back to dense residual space
        recon = sae.decode(z_steered)                          # (batch*seq, d_in)

        # Add back the reconstruction error to preserve unmodelled information
        if use_delta_correction:
            recon = recon + delta

        recon = recon.reshape(original_shape).to(dtype)

        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon

    return hook


def make_topk_steer_hook(
    sae: TopKSparseAutoencoder,
    persona_vec: torch.Tensor,          # (d_sae,) — post-TopK persona direction
    alpha: float,
    d_in: int,
    use_delta_correction: bool = True,
):
    """
    SAS-faithful post-TopK steering hook (used when use_pre_topk_personas=False).

    Implements the full SAS inference equation:
        ã = â(σ(f(a) + λ · v)) + Δ

    Step by step:
        z     = f(a) = encode(a)              ← sparse post-TopK encoding, k active
        Δ     = a − â(z)                      ← reconstruction residual (if enabled)
        s     = z + λ · v                     ← inject in sparse space (SAS Step C)
        σ(s)  = re-apply TopK to s            ← SAS Step D: keep exactly k features
        recon = decode(σ(s))                  ← SAS Step E
        ã     = recon + Δ                     ← SAS Step F

    Re-applying TopK after the addition (Step D) is the critical SAS requirement:
    without it the decoder receives inputs with an arbitrary number of non-zero
    entries, far outside the distribution it was trained on.

    Args:
        use_delta_correction: When True (default), compute Δ = act − decode(encode(act))
            from the unsteered pass and add it to the steered reconstruction.
            This cancels SAE reconstruction error injected on every forward pass.
    """
    def hook(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        original_shape = hidden.shape                          # (batch, seq, d_in)
        dtype = hidden.dtype
        flat = hidden.reshape(-1, d_in).to(torch.float32)     # (batch*seq, d_in)

        # SAS Step B — encode (post-TopK sparse, exactly k non-zeros)
        z = sae.encode(flat)                                   # (batch*seq, d_sae)

        # SAS Step B — Δ correction (reuse z to avoid a second encode call)
        if use_delta_correction:
            act_recon = sae.decode(z)                          # (batch*seq, d_in)
            delta = flat - act_recon                           # (batch*seq, d_in)

        # SAS Step C — add persona direction in sparse space
        pv = persona_vec.to(device=z.device, dtype=z.dtype)
        z_steered_raw = z + alpha * pv                         # (batch*seq, d_sae)

        # SAS Step D — re-apply the SAE sparsifier / σ.
        # Without this the decoder receives out-of-distribution dense inputs.
        z_steered = sae.sparsify(z_steered_raw)                # (batch*seq, d_sae)

        # SAS Step E — decode back to dense residual space
        recon = sae.decode(z_steered)                          # (batch*seq, d_in)

        # SAS Step F — add back reconstruction residual
        if use_delta_correction:
            recon = recon + delta

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
    """
    Compute mean per-token log-probability autoregressively so the metric
    matches generation-time steering behavior.
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def _save_eval_plots(results_all: Dict, config: LlamaScopePipelineConfig, out_dir: str):
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
    plt.plot(alpha_vals, mean_steered, marker="o", linewidth=2, label="Steered (Llama-Scope SAE)")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Accuracy")
    plt.title("Baseline vs Llama-Scope SAE Steering Accuracy")
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
    plt.title("Accuracy Gain from Llama-Scope SAE Steering")
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
    plt.title("Accuracy Gain by Value and Alpha (Llama-Scope SAE Steering)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_gain_heatmap.png"), dpi=200)
    plt.close()

    baseline_full_acc = np.array(
        [results_all[v]["baseline_full_logprob"]["accuracy"] for v in value_order]
    )
    steered_full_acc = np.array(
        [
            [results_all[v]["steered_full_logprob"][a]["accuracy"] for a in alpha_labels]
            for v in value_order
        ]
    )
    full_gain = steered_full_acc - baseline_full_acc[:, None]

    plt.figure(figsize=(8, 5))
    plt.axhline(
        float(baseline_full_acc.mean()),
        color="gray",
        linestyle="--",
        linewidth=2,
        label="Baseline",
    )
    plt.plot(
        alpha_vals,
        steered_full_acc.mean(axis=0),
        marker="o",
        linewidth=2,
        label="Steered (Llama-Scope SAE)",
    )
    plt.xlabel("Alpha")
    plt.ylabel("Mean Accuracy")
    plt.title("Baseline vs Full-Answer Logprob Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "full_logprob_baseline_vs_steered_accuracy.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(alpha_vals, full_gain.mean(axis=0), marker="o", linewidth=2)
    plt.axhline(0.0, color="gray", linestyle="--")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Accuracy Gain vs Baseline")
    plt.title("Full-Answer Logprob Accuracy Gain from Llama-Scope SAE Steering")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "full_logprob_accuracy_gain_vs_baseline.png"), dpi=200)
    plt.close()


def _metric_summary(
    results_all: Dict,
    config: LlamaScopePipelineConfig,
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

    per_value_best = {}
    for row_idx, value_name in enumerate(value_order):
        value_best_idx = int(np.argmax(acc_gain[row_idx]))
        value_best_alpha = alpha_labels[value_best_idx]
        per_value_best[value_name] = {
            "baseline_accuracy": float(baseline_acc[row_idx]),
            "best_alpha": value_best_alpha,
            "best_accuracy": float(steered_acc[row_idx, value_best_idx]),
            "best_accuracy_gain_vs_baseline": float(acc_gain[row_idx, value_best_idx]),
        }

    return {
        "metric": metric_name,
        "layer": config.layer,
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
        "per_value_best_alpha_by_accuracy_gain": per_value_best,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_sparse_steering(
    config: LlamaScopePipelineConfig,
    vectors: Dict[str, torch.Tensor],
    sae: Optional[TopKSparseAutoencoder] = None,
) -> Dict:
    """
    Evaluate Llama-Scope SAE steering on the held-out eval split.

    Args:
        config:  Pipeline configuration.
        vectors: {value → (d_sae,) sparse persona tensor}
        sae:     Optionally pass an already-loaded SAE; otherwise loaded/downloaded.

    Returns the full results dict and saves eval_results.json + plots.
    """
    out_dir = config.evaluation_dir
    out_path = os.path.join(out_dir, "eval_results.json")

    if os.path.exists(out_path):
        print(f"Evaluation results already exist at {out_path} — loading.")
        with open(out_path) as f:
            return json.load(f)

    # ── Load SAE ──────────────────────────────────────────────────────────────
    if sae is None:
        sae = get_or_download_sae(config, device="cpu", use_finetuned=True)

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
    is_instruct = "instruct" in name_lower and bool(getattr(tokenizer, "chat_template", None))

    layer_module = model.model.layers[config.layer]

    # Move SAE to same device as model for efficiency
    sae_device = device
    sae = sae.to(sae_device).eval()

    # ── Load eval data ────────────────────────────────────────────────────────
    # Use the same base-only held-out split as CAA/SphericalSteer.
    _, eval_data = load_steering_split(config)

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

        # Use pre-TopK hook when persona vectors are in pre-activation space
        hook_factory = (
            make_pre_topk_steer_hook
            if config.use_pre_topk_personas
            else make_topk_steer_hook
        )

        for alpha in config.alpha_values:
            hook_fn = hook_factory(
                sae, persona_vec, alpha, config.d_in,
                use_delta_correction=config.use_delta_correction,
            )
            handle = layer_module.register_forward_hook(hook_fn)

            try:
                print(f"  {val} (alpha={alpha}) …")
                steered_details = []
                steered_logprob_details = []
                for pair, bsl, bsl_lp in zip(eval_pairs, baseline_details, baseline_logprob_details):
                    tokens, a_id, b_id = format_eval_prompt(pair, tokenizer, is_instruct)
                    detail = _score_pair(model, tokens, a_id, b_id, pair.pos_is_a, device)
                    detail["sample_id"] = pair.sample_id
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
