"""
Evaluate steering via the Qwen-Scope SAE sparse latent space.

The steering hook intercepts the residual stream at config.layer, passes it
through the SAE encoder (TopK), adds the persona vector scaled by alpha,
then decodes back to the dense residual space.  The modified residual replaces
the layer output; all subsequent transformer layers see the steered hidden state.

Pre-TopK hook mechanic (config.use_pre_topk_personas=True, the default):
    residual       (batch, seq, 4096)
      ↓  sae.pre_encode  [dense, linear]
    pre            (batch, seq, 65536)

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

from .config import QwenScopePipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from .data_loader import (
    ContrastivePair,
    format_eval_prompt,
    load_combined,
    split_dataset,
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
            topk_vals_u, topk_idx_u = pre.topk(sae.k, dim=-1)
            z_unsteered = torch.zeros_like(pre)
            z_unsteered.scatter_(-1, topk_idx_u, topk_vals_u)
            act_recon = sae.decode(z_unsteered)                # (batch*seq, d_in)
            delta = flat - act_recon                           # (batch*seq, d_in)

        # Inject persona direction in pre-activation space
        pv = persona_vec.to(device=pre.device, dtype=pre.dtype)
        pre_steered = pre + alpha * pv
        # Apply TopK to the steered pre-activations
        topk_vals, topk_idx = pre_steered.topk(sae.k, dim=-1)
        z_steered = torch.zeros_like(pre_steered)
        z_steered.scatter_(-1, topk_idx, topk_vals)
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
    Post-TopK steering hook (legacy, use make_pre_topk_steer_hook instead).

    Adds the persona direction to the post-TopK sparse z AFTER TopK selection.
    This bypasses the TopK gate, so the value signal cannot change which features
    are selected — it only modulates already-selected features.
    Use this only when config.use_pre_topk_personas=False.

    Args:
        use_delta_correction: When True (default), compute Δ = act − decode(encode(act))
            from the unsteered pass and add it to the steered reconstruction.
            This cancels SAE reconstruction error that would otherwise be injected
            into the residual stream on every forward pass.
    """
    def hook(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        original_shape = hidden.shape                          # (batch, seq, d_in)
        dtype = hidden.dtype
        flat = hidden.reshape(-1, d_in).to(torch.float32)     # (batch*seq, d_in)

        # Encode → sparse space (TopK, k=50)
        z = sae.encode(flat)                                   # (batch*seq, d_sae)

        # Δ correction: unsteered reconstruction error (z is already the
        # unsteered encoding, so reuse it directly)
        if use_delta_correction:
            act_recon = sae.decode(z)                          # (batch*seq, d_in)
            delta = flat - act_recon                           # (batch*seq, d_in)

        # Add persona direction in sparse space
        pv = persona_vec.to(device=z.device, dtype=z.dtype)
        z_steered = z + alpha * pv
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


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def _save_eval_plots(results_all: Dict, config: QwenScopePipelineConfig, out_dir: str):
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
    plt.plot(alpha_vals, mean_steered, marker="o", linewidth=2, label="Steered (Qwen-Scope SAE)")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Accuracy")
    plt.title("Baseline vs Qwen-Scope SAE Steering Accuracy")
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
    plt.title("Accuracy Gain from Qwen-Scope SAE Steering")
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
    plt.title("Accuracy Gain by Value and Alpha (Qwen-Scope SAE Steering)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_gain_heatmap.png"), dpi=200)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_sparse_steering(
    config: QwenScopePipelineConfig,
    vectors: Dict[str, torch.Tensor],
    sae: Optional[TopKSparseAutoencoder] = None,
) -> Dict:
    """
    Evaluate Qwen-Scope SAE steering on the held-out eval split.

    Args:
        config:  Pipeline configuration.
        vectors: {value → (d_sae,) sparse persona tensor}
        sae:     Optionally pass an already-loaded SAE; otherwise loaded/downloaded.

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
    is_instruct = "base" not in name_lower and "pt" not in name_lower

    layer_module = model.model.layers[config.layer]

    # Move SAE to same device as model for efficiency
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
        for pair in eval_pairs:
            tokens, a_id, b_id = format_eval_prompt(pair, tokenizer, is_instruct)
            result = _score_pair(model, tokens, a_id, b_id, pair.pos_is_a, device)
            result["sample_id"] = pair.sample_id
            baseline_details.append(result)
        baseline_summary = _summarize(baseline_details)

        steered_results: Dict = {}

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
                for pair, bsl in zip(eval_pairs, baseline_details):
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

        results_all[val] = {
            "baseline": baseline_summary,
            "steered": steered_results,
        }

    # ── Save results ──────────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)

    _save_eval_plots(results_all, config, out_dir)
    print(f"Evaluation complete. Results → {out_dir}")
    return results_all
