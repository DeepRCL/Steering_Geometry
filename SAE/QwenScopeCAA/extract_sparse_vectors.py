"""
Extract per-value CAA persona vectors in the Qwen-Scope SAE sparse latent space.

For each contrastive pair (pos/neg) and each Schwartz value:
  1. Run Qwen forward pass.
  2. Hook model.model.layers[config.layer] — capture the LAST TOKEN residual-
     stream output (output[0][0, -1, :], shape: (4096,)).
  3. Encode through the SAE using pre_encode (dense, pre-TopK):
       pre = encoder(act)               # (65536,) — dense, continuous
  4. Accumulate pre for pos and neg prompts separately into pos_z / neg_z.

After all pairs for a value (three sequential steps):
  Step 2 — τ frequency-masked non-zero mean (config.tau):
       v_pos[c] = mean of pos_z[:, c] over non-zero rows  if freq_pos[c] ≥ τ
               = 0                                         otherwise
       (same for v_neg)
  Step 3 — common feature removal (config.remove_common_features,
       post-TopK sparse mode only):
       Features non-zero in BOTH v_pos and v_neg are zeroed on both sides.
       These are likely syntactic/positional artifacts shared across all values.
       Skipped in pre-TopK (dense) mode because all features are non-zero,
       which would wipe both vectors entirely; the dense subtraction already
       cancels shared activations naturally.
  Step 4 — difference vector:
       persona_vec[value] = v_pos - v_neg          shape: (65536,)

Results are cached to disk; already-computed values are loaded from cache.
Metadata (including tau, feature counts above threshold, and common-feature
removal count) is written to value_metadata.json.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import QwenScopePipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from .data_loader import (
    ContrastivePair,
    format_prompts,
    load_steering_split,
    print_dataset_summary,
)
from .topk_sae_model import TopKSparseAutoencoder, get_or_download_sae


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _vec_path(config: QwenScopePipelineConfig, val: str) -> str:
    return os.path.join(config.sparse_vectors_dir, f"{safe_name(val)}.pt")


def _all_cached(config: QwenScopePipelineConfig) -> bool:
    return all(
        os.path.exists(_vec_path(config, v)) and os.path.exists(_act_path(config, v))
        for v in SCHWARTZ_CIRCUMPLEX_ORDER
    )


def _load_cached(config: QwenScopePipelineConfig) -> Dict[str, torch.Tensor]:
    return {v: torch.load(_vec_path(config, v), map_location="cpu")
            for v in SCHWARTZ_CIRCUMPLEX_ORDER}


def _act_path(config: QwenScopePipelineConfig, val: str) -> str:
    return os.path.join(config.steering_activations_dir, f"{safe_name(val)}.pt")


# ──────────────────────────────────────────────────────────────────────────────
# Frequency-masked mean
# ──────────────────────────────────────────────────────────────────────────────
def _tau_mean(z_list: List[torch.Tensor], tau: float) -> torch.Tensor:
    """
    Frequency-masked non-zero mean over a list of feature vectors.

    For each feature dimension c:
      freq[c] = count(z_i[c] != 0) / N
      If freq[c] >= tau  →  mean_vec[c] = sum_i(z_i[c]) / count(z_i[c] != 0)
      If freq[c] <  tau  →  mean_vec[c] = 0

    When tau=0.0 all features are retained and the result equals the
    non-zero-row mean (sum divided by non-zero count, not by N).  For the
    default dense pre-TopK case every entry is non-zero, so this is numerically
    identical to torch.stack(z_list).mean(dim=0) when tau ≤ 1.0.

    Args:
        z_list : list of N tensors, each of shape (d_sae,)
        tau    : frequency threshold in [0, 1]

    Returns:
        mean_vec : (d_sae,) float32 tensor with below-tau features zeroed.
    """
    S = torch.stack(z_list)                         # (N, d_sae)
    N = S.shape[0]
    nonzero_counts = (S != 0).float().sum(dim=0)    # (d_sae,)
    freq = nonzero_counts / N                        # (d_sae,)
    mask = freq >= tau                               # (d_sae,) bool
    col_sums = S.sum(dim=0)                         # (d_sae,)
    safe_counts = nonzero_counts.clamp(min=1.0)     # avoid division by zero
    nz_mean = col_sums / safe_counts                # (d_sae,)
    nz_mean = nz_mean * mask.float()                # zero below-tau features
    return nz_mean


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def extract_sparse_vectors(
    config: QwenScopePipelineConfig,
    sae: Optional[TopKSparseAutoencoder] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-value sparse CAA persona vectors in the Qwen-Scope feature space.

    If sae is None, loads the fine-tuned checkpoint when present; otherwise
    downloads and uses the pre-trained Qwen-Scope SAE directly.

    Already-computed vectors are loaded from cache; only missing values are
    (re-)computed.

    Extraction applies three sequential steps controlled by config fields:
      1. Collect pre_encode activations for all training pairs.
      2. _tau_mean (config.tau): non-zero mean, zeroing features that fired in
         fewer than tau*N samples on either side.
      3. Common removal (config.remove_common_features): zero features active in
         both v_pos and v_neg before subtraction.
      4. Difference: persona_vec = v_pos − v_neg.

    Returns:
        {value: tensor of shape (d_sae,)}  — one float32 persona vector per
        Schwartz value, with tau-filtered and common features zeroed as configured.
    """
    os.makedirs(config.sparse_vectors_dir, exist_ok=True)

    # SAE fine-tuning may use the larger base+Touche dataset, but persona-vector
    # extraction is locked to the same base-only split as CAA/SphericalSteer.
    train_data, eval_data = load_steering_split(config)
    print_dataset_summary(train_data, eval_data)

    if _all_cached(config):
        print("All sparse persona vectors cached — loading from disk.")
        return _load_cached(config)

    missing = [
        v for v in SCHWARTZ_CIRCUMPLEX_ORDER
        if not os.path.exists(_vec_path(config, v)) or not os.path.exists(_act_path(config, v))
    ]
    print(f"{len(missing)}/{len(SCHWARTZ_CIRCUMPLEX_ORDER)} values need extraction.")

    # ── Load SAE (CPU) ────────────────────────────────────────────────────────
    if sae is None:
        sae = get_or_download_sae(config, device="cpu", use_finetuned=True)
    sae = sae.cpu().eval()

    # ── Load Qwen ─────────────────────────────────────────────────────────────
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
    torch.set_grad_enabled(False)

    name_lower = config.model_name.lower()
    is_instruct = "base" not in name_lower and "pt" not in name_lower

    # ── Hook: last-token residual-stream activation ───────────────────────────
    _current: Dict[str, torch.Tensor] = {}

    def _resid_hook(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # (1, seq_len, d_in) → last token → cpu float32
        _current["act"] = hidden[0, -1, :].detach().to(dtype=torch.float32, device="cpu")

    layer_module = model.model.layers[config.layer]
    handle = layer_module.register_forward_hook(_resid_hook)

    # ── Extraction loop ───────────────────────────────────────────────────────
    vectors: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, dict] = {}

    try:
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            cache_p = _vec_path(config, val)
            if os.path.exists(cache_p) and os.path.exists(_act_path(config, val)):
                vectors[val] = torch.load(cache_p, map_location="cpu")
                print(f"  [cache] {val}")
                continue

            pairs: List[ContrastivePair] = train_data.get(val, [])
            if not pairs:
                print(f"  [WARNING] No training pairs for '{val}' — zero vector.")
                vec = torch.zeros(config.d_sae)
                torch.save(vec, cache_p)
                vectors[val] = vec
                metadata[val] = {"n_train": 0, "n_eval": len(eval_data.get(val, []))}
                continue

            pos_z: List[torch.Tensor] = []
            neg_z: List[torch.Tensor] = []
            pos_acts: List[torch.Tensor] = []
            neg_acts: List[torch.Tensor] = []

            # Choose pre-TopK (dense) or post-TopK (sparse) representation.
            # pre_encode is strongly preferred: post-TopK vectors have ~99.9%
            # zero entries, so pairwise cosine similarities collapse onto shared
            # "common" features rather than value-discriminative ones.
            encode_fn = sae.pre_encode if config.use_pre_topk_personas else sae.encode

            for pair in tqdm(pairs, desc=f"  [{val}]", leave=False):
                pos_tokens, neg_tokens = format_prompts(pair, tokenizer, is_instruct)

                # Positive prompt
                _current.clear()
                model(torch.tensor([pos_tokens]).to(device))
                act_pos = _current["act"]
                z_pos = encode_fn(act_pos)    # (d_sae,)
                pos_z.append(z_pos.detach())
                pos_acts.append(act_pos.detach())

                # Negative prompt
                _current.clear()
                model(torch.tensor([neg_tokens]).to(device))
                act_neg = _current["act"]
                z_neg = encode_fn(act_neg)    # (d_sae,)
                neg_z.append(z_neg.detach())
                neg_acts.append(act_neg.detach())

            # Step 2 — τ frequency-masked non-zero mean
            v_pos = _tau_mean(pos_z, config.tau)    # (d_sae,)
            v_neg = _tau_mean(neg_z, config.tau)    # (d_sae,)

            # Record tau survivors before common removal
            n_pos_above_tau = int((v_pos != 0).sum().item())
            n_neg_above_tau = int((v_neg != 0).sum().item())

            # Step 3 — common feature removal
            # Only meaningful for post-TopK (sparse) representations where
            # non-zero entries indicate actual feature activation.  In the
            # default pre-TopK (dense) mode every dimension is non-zero, so
            # common_mask would be all-True and wipe both vectors entirely,
            # producing zero persona vectors.  The dense subtraction already
            # cancels shared activations naturally (mean_pos[c] - mean_neg[c]),
            # so this step is skipped in that mode.
            n_common = 0
            if config.remove_common_features and not config.use_pre_topk_personas:
                common_mask = (v_pos != 0) & (v_neg != 0)
                n_common = int(common_mask.sum().item())
                v_pos = v_pos.clone()
                v_neg = v_neg.clone()
                v_pos[common_mask] = 0.0
                v_neg[common_mask] = 0.0

            # Step 4 — difference vector
            vec = v_pos - v_neg                     # (d_sae,) persona vector

            torch.save(vec, cache_p)
            os.makedirs(config.steering_activations_dir, exist_ok=True)
            torch.save(
                {
                    "value": val,
                    "source": "caa_compatible_base_train_split",
                    "layer": config.layer,
                    "sample_ids": [p.sample_id for p in pairs],
                    "pos": torch.stack(pos_acts).cpu(),
                    "neg": torch.stack(neg_acts).cpu(),
                },
                _act_path(config, val),
            )
            vectors[val] = vec
            metadata[val] = {
                "n_train": len(pairs),
                "n_eval": len(eval_data.get(val, [])),
                "vec_norm": float(vec.norm().item()),
                "n_positive_features": int((vec > 0).sum().item()),
                "n_negative_features": int((vec < 0).sum().item()),
                "tau": config.tau,
                "n_pos_features_above_tau": n_pos_above_tau,
                "n_neg_features_above_tau": n_neg_above_tau,
                "n_common_features_removed": n_common,
            }
            print(
                f"  {val}: n={len(pairs)}, "
                f"norm={vec.norm():.4f}, "
                f"+feats={(vec > 0).sum()}, "
                f"-feats={(vec < 0).sum()}, "
                f"tau_pos={n_pos_above_tau}, "
                f"tau_neg={n_neg_above_tau}, "
                f"common_removed={n_common}"
            )

    finally:
        handle.remove()

    # ── Save metadata ─────────────────────────────────────────────────────────
    if metadata:
        meta_path = os.path.join(config.sparse_vectors_dir, "value_metadata.json")
        existing: Dict = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                existing = json.load(f)
        existing.update(metadata)
        with open(meta_path, "w") as f:
            json.dump(existing, f, indent=2)

    return vectors
