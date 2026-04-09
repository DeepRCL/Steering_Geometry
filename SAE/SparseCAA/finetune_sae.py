"""
Fine-tune the pre-trained SAE on value-specific MLP activations.

WHY FINE-TUNE?
──────────────────────────────────────────────────────────────────────────────
The released SAE was trained on ~50M tokens of general text (Pile-uncopyrighted).
Fine-tuning on the argumentative value dataset adapts the SAE features to be
more sensitive to value-related language patterns, which should give more
discriminative persona vectors in the sparse latent space.

PROCEDURE
──────────────────────────────────────────────────────────────────────────────
Step A — Activation collection (one Qwen forward pass):
  For every row in the combined dataset, run BOTH the positive and negative
  prompts through Qwen.  Hook `model.model.layers[mlp_layer].mlp` and collect
  ALL token-position MLP outputs (shape: (seq_len, 4096)) — not just the last
  token.  Each position gives the SAE one training example.  Activations are
  written to an HDF5 cache file to avoid holding ~638K vectors in RAM.

Step B — SAE fine-tuning (CPU or GPU, no Qwen needed):
  Read the cached activations in shuffled mini-batches and minimise:
    L = MSE(x_hat, x) + λ · ||z||₁
  using Adam with a small learning rate, starting from the pre-trained weights.
  The checkpoint is saved in the same format as sae_base_best.pt.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..sae_model import SparseAutoencoder, load_sae
from .config import SparseCAAPipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER
from .data_loader import (
    ContrastivePair,
    format_prompts,
    load_combined,
    split_dataset,
)


# ──────────────────────────────────────────────────────────────────────────────
# Step A: Collect MLP activations from all prompts
# ──────────────────────────────────────────────────────────────────────────────
def _collect_activations(
    config: SparseCAAPipelineConfig,
    train_data: Dict[str, List[ContrastivePair]],
    tokenizer,
    model,
    device: torch.device,
    is_instruct: bool,
    cache_path: str,
) -> int:
    """
    Run all training prompts through Qwen, capture every token's MLP activation
    at layer config.mlp_layer, and write them to an HDF5 file.

    Returns the total number of activation vectors written.
    """
    if os.path.exists(cache_path):
        with h5py.File(cache_path, "r") as f:
            n = f["activations"].shape[0]
        print(f"  [cache] Activation cache found: {n:,} vectors at {cache_path}")
        return n

    print(f"  Collecting MLP activations → {cache_path}")
    mlp_module = model.model.layers[config.mlp_layer].mlp

    _buf: Dict[str, torch.Tensor] = {}

    def _hook(module, inp, output):
        act = output[0] if isinstance(output, tuple) else output
        # act shape: (1, seq_len, d_in)  — batch size is always 1 here
        _buf["act"] = act[0].detach().to(dtype=torch.float32, device="cpu")

    handle = mlp_module.register_forward_hook(_hook)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Collect all pairs from all values
    all_pairs: List[ContrastivePair] = []
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        all_pairs.extend(train_data.get(val, []))

    total_vectors = 0

    with h5py.File(cache_path, "w") as hf:
        # We write in chunks; pre-allocate a resizable dataset
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

                seq_acts = _buf["act"].numpy()  # (seq_len, d_in)
                n_tok = seq_acts.shape[0]

                # Resize and append
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
    config: SparseCAAPipelineConfig,
    sae: SparseAutoencoder,
    cache_path: str,
    n_vectors: int,
    sae_device: torch.device,
) -> SparseAutoencoder:
    """
    Fine-tune `sae` for config.finetune_epochs epochs on the cached activations.
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

                x_hat, z = sae(x)
                loss_recon = F.mse_loss(x_hat, x)
                loss_l1 = config.l1_coefficient * z.abs().mean()
                loss = loss_recon + loss_l1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(
                    loss=f"{loss.item():.5f}",
                    recon=f"{loss_recon.item():.5f}",
                    l1=f"{loss_l1.item():.5f}",
                )

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.5f}")

    sae.eval()
    return sae


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def finetune_sae(config: SparseCAAPipelineConfig) -> SparseAutoencoder:
    """
    Fine-tune the SAE on value-specific MLP activations.

    If a fine-tuned checkpoint already exists at config.finetuned_sae_path,
    load and return it without re-training.

    Returns the fine-tuned SparseAutoencoder in eval mode on CPU.
    """
    if os.path.exists(config.finetuned_sae_path):
        print(f"Fine-tuned SAE already exists at {config.finetuned_sae_path} — loading.")
        return load_sae(config.finetuned_sae_path, config.d_in, config.d_sae)

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
    torch.set_grad_enabled(False)

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

    # ── Load base SAE ─────────────────────────────────────────────────────────
    sae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading base SAE: {config.sae_checkpoint}")
    sae = load_sae(config.sae_checkpoint, config.d_in, config.d_sae, device=str(sae_device))

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    sae = _finetune_sae(config, sae, cache_path, n_vectors, sae_device)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(config.finetuned_sae_path), exist_ok=True)
    torch.save({"model_state_dict": sae.state_dict()}, config.finetuned_sae_path)
    print(f"\nFine-tuned SAE saved to {config.finetuned_sae_path}")

    return sae.cpu().eval()
