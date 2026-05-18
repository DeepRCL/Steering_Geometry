"""
Extract per-value CAA persona vectors in the Qwen-Scope SAE sparse latent space.

For each contrastive pair (pos/neg) and each Schwartz value:
  1. Run Qwen forward pass.
  2. Hook model.model.layers[config.layer] — capture the LAST TOKEN residual-
     stream output (output[0][0, -1, :], shape: (4096,)).
  3. Encode through the SAE:
       pre = encoder(act)               # (65536,)
       z = TopK(pre, k=50)              # exactly 50 non-zero values
  4. Accumulate z for pos and neg prompts separately.

After all pairs for a value:
  persona_vec[value] = mean(z_pos_list) - mean(z_neg_list)   shape: (65536,)

This is the standard CAA difference vector, computed in the Qwen-Scope
sparse feature space (residual stream) rather than the dense MLP space.

Results are cached to disk; already-computed values are loaded from cache.
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
    load_combined,
    print_dataset_summary,
    split_dataset,
)
from .topk_sae_model import TopKSparseAutoencoder, get_or_download_sae


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _vec_path(config: QwenScopePipelineConfig, val: str) -> str:
    return os.path.join(config.sparse_vectors_dir, f"{safe_name(val)}.pt")


def _all_cached(config: QwenScopePipelineConfig) -> bool:
    return all(os.path.exists(_vec_path(config, v)) for v in SCHWARTZ_CIRCUMPLEX_ORDER)


def _load_cached(config: QwenScopePipelineConfig) -> Dict[str, torch.Tensor]:
    return {v: torch.load(_vec_path(config, v), map_location="cpu")
            for v in SCHWARTZ_CIRCUMPLEX_ORDER}


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

    Returns:
        {value: tensor of shape (d_sae,)}  — one float32 sparse persona vector
        per Schwartz value.
    """
    os.makedirs(config.sparse_vectors_dir, exist_ok=True)

    if _all_cached(config):
        print("All sparse persona vectors cached — loading from disk.")
        return _load_cached(config)

    missing = [v for v in SCHWARTZ_CIRCUMPLEX_ORDER
               if not os.path.exists(_vec_path(config, v))]
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

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_combined(config)
    train_data, eval_data = split_dataset(df, config)
    print_dataset_summary(train_data, eval_data)

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
            if os.path.exists(cache_p):
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
                z_pos = encode_fn(_current["act"])    # (d_sae,)
                pos_z.append(z_pos.detach())

                # Negative prompt
                _current.clear()
                model(torch.tensor([neg_tokens]).to(device))
                z_neg = encode_fn(_current["act"])    # (d_sae,)
                neg_z.append(z_neg.detach())

            pos_mean = torch.stack(pos_z).mean(dim=0)
            neg_mean = torch.stack(neg_z).mean(dim=0)
            vec = pos_mean - neg_mean   # (d_sae,) persona vector

            torch.save(vec, cache_p)
            vectors[val] = vec
            metadata[val] = {
                "n_train": len(pairs),
                "n_eval": len(eval_data.get(val, [])),
                "vec_norm": float(vec.norm().item()),
                "n_positive_features": int((vec > 0).sum().item()),
                "n_negative_features": int((vec < 0).sum().item()),
            }
            print(
                f"  {val}: n={len(pairs)}, "
                f"norm={vec.norm():.4f}, "
                f"+feats={(vec > 0).sum()}, "
                f"-feats={(vec < 0).sum()}"
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
