"""
Fine-tune the Qwen-Scope pre-trained SAE on value-specific residual activations.

WHY FINE-TUNE?
──────────────────────────────────────────────────────────────────────────────
The Qwen-Scope SAE was trained on general text.  Fine-tuning on the
argumentative value dataset adapts the SAE features to be more sensitive to
value-related language patterns, which should give more discriminative persona
vectors in the sparse latent space.

WHY MSE ONLY (NO L1)?
──────────────────────────────────────────────────────────────────────────────
The TopK constraint (k=50) enforces sparsity structurally: exactly 50 features
are kept active per token, regardless of loss.  Unlike ReLU SAEs, which use
L1 regularisation to push activations toward zero, TopK needs no penalty term.
Using L1 on top of TopK would add an unnecessary bias without any benefit.

PROCEDURE
──────────────────────────────────────────────────────────────────────────────
Step A — Activation collection (one Qwen forward pass per prompt):
  For every row in the combined dataset, run BOTH the positive and negative
  prompts through Qwen.  Hook `model.model.layers[config.layer]` and collect
  ALL token-position residual-stream outputs (shape: (seq_len, 4096)) — not
  just the last token.  Each position gives the SAE one training example.
  Activations are written to an HDF5 cache file.

Step B — SAE fine-tuning (CPU or GPU, Qwen not needed):
  Read the cached activations in shuffled mini-batches and minimise:
    L = MSE(x_hat, x)
  using Adam with a small learning rate, starting from the pre-trained weights.
  The checkpoint is saved in the Qwen-Scope format {W_enc, W_dec, b_enc, b_dec}.
"""
from __future__ import annotations

import os
from typing import Dict, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import QwenScopePipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER
from .data_loader import (
    ContrastivePair,
    format_prompts,
    load_combined,
    split_dataset,
)
from .topk_sae_model import (
    TopKSparseAutoencoder,
    get_or_download_sae,
    save_qwenscope_sae,
)


# ──────────────────────────────────────────────────────────────────────────────
# Step A: Collect residual-stream activations from all prompts
# ──────────────────────────────────────────────────────────────────────────────
def _collect_activations(
    config: QwenScopePipelineConfig,
    train_data: Dict[str, List[ContrastivePair]],
    tokenizer,
    model,
    device: torch.device,
    is_instruct: bool,
    cache_path: str,
) -> int:
    """
    Run all training prompts through Qwen, capture every token's residual-stream
    activation at config.layer, and write them to an HDF5 file.

    Hook target: model.model.layers[config.layer]
    Captured:    output[0]  — full residual stream, shape (1, seq_len, d_in)

    Returns the total number of activation vectors written.
    """
    if os.path.exists(cache_path):
        with h5py.File(cache_path, "r") as f:
            n = f["activations"].shape[0]
        print(f"  [cache] Activation cache found: {n:,} vectors at {cache_path}")
        return n

    print(f"  Collecting residual-stream activations (layer {config.layer}) → {cache_path}")

    _buf: Dict[str, torch.Tensor] = {}

    def _hook(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden shape: (1, seq_len, d_in) — batch size is always 1 here
        _buf["act"] = hidden[0].detach().to(dtype=torch.float32, device="cpu")

    layer_module = model.model.layers[config.layer]
    handle = layer_module.register_forward_hook(_hook)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    all_pairs: List[ContrastivePair] = []
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        all_pairs.extend(train_data.get(val, []))

    total_vectors = 0

    with h5py.File(cache_path, "w") as hf:
        ds = hf.create_dataset(
            "activations",
            shape=(0, config.d_in),
            maxshape=(None, config.d_in),
            dtype="float32",
            chunks=(min(4096, max(1, len(all_pairs))), config.d_in),
        )

        model.eval()
        for pair in tqdm(all_pairs, desc="  Collecting activations"):
            for tokens in format_prompts(pair, tokenizer, is_instruct):
                _buf.clear()
                with torch.no_grad():
                    model(torch.tensor([tokens]).to(device))

                if "act" not in _buf:
                    continue

                seq_acts = _buf["act"].numpy()   # (seq_len, d_in)
                n_tok = seq_acts.shape[0]

                ds.resize(total_vectors + n_tok, axis=0)
                ds[total_vectors : total_vectors + n_tok] = seq_acts
                total_vectors += n_tok

    handle.remove()
    print(f"  Collected {total_vectors:,} activation vectors.")
    return total_vectors


# ──────────────────────────────────────────────────────────────────────────────
# Step B: Fine-tune SAE on cached activations
# ──────────────────────────────────────────────────────────────────────────────
def _finetune_sae(
    config: QwenScopePipelineConfig,
    sae: TopKSparseAutoencoder,
    cache_path: str,
    n_vectors: int,
    sae_device: torch.device,
) -> TopKSparseAutoencoder:
    """
    Fine-tune `sae` for config.finetune_epochs epochs on the cached activations.
    Loss = MSE(x_hat, x)  — no L1 because TopK enforces sparsity structurally.
    """
    sae = sae.to(sae_device)
    sae.train()

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.finetune_lr)

    batch_size = config.finetune_batch_size
    indices = np.arange(n_vectors)

    print(f"\n  Fine-tuning SAE for {config.finetune_epochs} epoch(s)")
    print(f"  Activation vectors : {n_vectors:,}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Steps per epoch    : {n_vectors // batch_size}")

    with h5py.File(cache_path, "r") as hf:
        ds = hf["activations"]

        for epoch in range(config.finetune_epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(
                range(0, n_vectors - batch_size + 1, batch_size),
                desc=f"  Epoch {epoch + 1}/{config.finetune_epochs}",
            )
            for start in pbar:
                batch_idx = np.sort(indices[start : start + batch_size])
                x = torch.tensor(ds[batch_idx], dtype=torch.float32).to(sae_device)

                x_hat, _ = sae(x)
                loss = F.mse_loss(x_hat, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.5f}")

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.5f}")

    sae.eval()
    return sae


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def finetune_sae(config: QwenScopePipelineConfig) -> TopKSparseAutoencoder:
    """
    Fine-tune the Qwen-Scope SAE on value-specific residual activations.

    If a fine-tuned checkpoint already exists at config.finetuned_sae_path,
    load and return it without re-training.

    Returns the fine-tuned TopKSparseAutoencoder in eval mode on CPU.
    """
    if os.path.exists(config.finetuned_sae_path):
        print(f"Fine-tuned SAE already exists at {config.finetuned_sae_path} — loading.")
        from .topk_sae_model import load_qwenscope_sae
        return load_qwenscope_sae(config.finetuned_sae_path, k=config.k, device="cpu")

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

    name_lower = config.model_name.lower()
    is_instruct = "base" not in name_lower and "pt" not in name_lower

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading combined dataset …")
    df = load_combined(config)
    train_data, _ = split_dataset(df, config)

    # ── Collect activations ───────────────────────────────────────────────────
    cache_path = os.path.join(config.run_dir, "activation_cache.h5")
    os.makedirs(config.run_dir, exist_ok=True)

    torch.set_grad_enabled(False)
    n_vectors = _collect_activations(
        config, train_data, tokenizer, model, device, is_instruct, cache_path
    )
    torch.set_grad_enabled(True)

    # Free GPU memory before fine-tuning
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Load pre-trained Qwen-Scope SAE ───────────────────────────────────────
    sae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading pre-trained Qwen-Scope SAE (layer {config.layer}) …")
    sae = get_or_download_sae(config, device=str(sae_device), use_finetuned=False)

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    sae = _finetune_sae(config, sae, cache_path, n_vectors, sae_device)

    # ── Save in Qwen-Scope format ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(config.finetuned_sae_path), exist_ok=True)
    save_qwenscope_sae(sae, config.finetuned_sae_path)

    return sae.cpu().eval()
