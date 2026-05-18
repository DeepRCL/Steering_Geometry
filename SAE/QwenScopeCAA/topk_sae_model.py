"""
TopK Sparse Autoencoder matching the Qwen-Scope checkpoint format.

Qwen-Scope checkpoint layout (layer{n}.sae.pt):
  W_enc : (d_sae, d_in)   encoder weight matrix
  b_enc : (d_sae,)         encoder bias
  W_dec : (d_in,  d_sae)  decoder weight matrix
  b_dec : (d_in,)          decoder bias

Encoding (matches the Qwen-Scope demo exactly):
  pre  = x @ W_enc.T + b_enc          # (..., d_sae)
  keep top-k values, zero the rest
  z    = sparse feature activations    # (..., d_sae)

Decoding:
  x_hat = z @ W_dec.T + b_dec         # (..., d_in)

The TopK constraint replaces L1 regularisation: sparsity is enforced
structurally rather than via a penalty term, so fine-tuning uses MSE only.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class TopKSparseAutoencoder(nn.Module):
    """
    TopK SAE matching the Qwen-Scope architecture.

    Parameters
    ----------
    d_in  : input dimension (4096 for Qwen3.5-9B)
    d_sae : feature dimension (65536 for Qwen-Scope W64K)
    k     : number of active features kept per token (50 for Qwen-Scope)
    """

    def __init__(self, d_in: int = 4096, d_sae: int = 65536, k: int = 50):
        super().__init__()
        self.k = k
        # nn.Linear stores weight as (out, in), matching (d_sae, d_in) for encoder
        # and (d_in, d_sae) for decoder — same shapes as W_enc and W_dec.
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in)

    # ── Pre-activation (dense) ────────────────────────────────────────────────
    def pre_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the DENSE pre-TopK encoder activations: x @ W_enc.T + b_enc.

        This is the continuous representation BEFORE sparsification.  Use this
        for persona vector computation and for steering injection — it avoids
        the two failure modes of post-TopK vectors:
          1. ~99.9% of entries are exactly zero (only 50/65536 per sample),
             making cosine similarity between persona vectors dominated by
             "common" features rather than value-discriminative ones.
          2. TopK can keep negative pre-activations, which carry the wrong
             semantics (feature presence vs. absence) in the difference vector.

        Args:
            x : (..., d_in)
        Returns:
            pre : (..., d_sae) — dense, continuous, can be negative
        """
        return self.encoder(x)                         # x @ W_enc.T + b_enc

    # ── Encode ────────────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard TopK encode: exactly k non-zero values per token.

        Use this for reconstruction and when you need the SAE's native sparse
        representation.  For persona vectors and steering, prefer pre_encode().

        Args:
            x : (..., d_in)
        Returns:
            z : (..., d_sae) — exactly k non-zero values per token
        """
        pre = self.pre_encode(x)                       # (..., d_sae)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    # ── Decode ────────────────────────────────────────────────────────────────
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z    : (..., d_sae)
        Returns:
            x_hat: (..., d_in)
        """
        return self.decoder(z)

    # ── Full forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_hat : (..., d_in)  – reconstruction
            z     : (..., d_sae) – sparse feature activations
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ──────────────────────────────────────────────────────────────────────────────
def load_qwenscope_sae(
    path: str,
    k: int = 50,
    device: str = "cpu",
) -> TopKSparseAutoencoder:
    """
    Load a TopKSparseAutoencoder from a Qwen-Scope .pt checkpoint.

    The file must be a dict with keys W_enc, W_dec, b_enc, b_dec.

    Args:
        path   : Path to ``layer{n}.sae.pt``
        k      : TopK budget (must match the SAE's training configuration)
        device : ``"cpu"``, ``"cuda"``, or ``"mps"``

    Returns:
        Loaded SAE in eval mode on the requested device.
    """
    ckpt: dict = torch.load(path, map_location=device, weights_only=True)

    W_enc = ckpt["W_enc"]   # (d_sae, d_in)
    b_enc = ckpt["b_enc"]   # (d_sae,)
    W_dec = ckpt["W_dec"]   # (d_in, d_sae)
    b_dec = ckpt["b_dec"]   # (d_in,)

    d_sae, d_in = W_enc.shape

    sae = TopKSparseAutoencoder(d_in=d_in, d_sae=d_sae, k=k)
    sae.encoder.weight.data = W_enc.float()
    sae.encoder.bias.data   = b_enc.float()
    sae.decoder.weight.data = W_dec.float()
    sae.decoder.bias.data   = b_dec.float()

    sae = sae.to(device)
    sae.eval()
    print(f"Qwen-Scope SAE loaded: {path}  (d_in={d_in}, d_sae={d_sae}, k={k}, device={device})")
    return sae


def save_qwenscope_sae(sae: TopKSparseAutoencoder, path: str) -> None:
    """
    Save a TopKSparseAutoencoder in the Qwen-Scope dict format so that
    load_qwenscope_sae can reload it after fine-tuning.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "W_enc": sae.encoder.weight.data.cpu(),
        "b_enc": sae.encoder.bias.data.cpu(),
        "W_dec": sae.decoder.weight.data.cpu(),
        "b_dec": sae.decoder.bias.data.cpu(),
    }
    torch.save(ckpt, path)
    print(f"Fine-tuned Qwen-Scope SAE saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace Hub download helper
# ──────────────────────────────────────────────────────────────────────────────
def download_layer_sae(
    repo_id: str,
    layer: int,
    cache_dir: str,
) -> str:
    """
    Download ``layer{layer}.sae.pt`` from a HuggingFace Hub repo.

    The file is cached locally; subsequent calls return immediately without
    re-downloading.

    Args:
        repo_id   : HuggingFace repo, e.g. ``"Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"``
        layer     : Layer index (0–31 for Qwen-Scope)
        cache_dir : Directory to store downloaded .pt files

    Returns:
        Absolute path to the local .pt file.
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)
    filename = f"layer{layer}.sae.pt"
    local_target = os.path.join(cache_dir, filename)

    if os.path.exists(local_target):
        print(f"  [cache] SAE layer {layer} already at {local_target}")
        return local_target

    print(f"  Downloading {repo_id}/{filename} …")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=cache_dir,
    )
    return downloaded


def get_or_download_sae(
    config,
    device: str = "cpu",
    use_finetuned: bool = True,
) -> TopKSparseAutoencoder:
    """
    Convenience: return the fine-tuned SAE if it exists (and use_finetuned),
    otherwise download and load the pre-trained Qwen-Scope checkpoint.

    Args:
        config       : QwenScopePipelineConfig instance
        device       : Target device string
        use_finetuned: If True, prefer config.finetuned_sae_path when present.
    """
    if use_finetuned and os.path.exists(config.finetuned_sae_path):
        print(f"Loading fine-tuned SAE from {config.finetuned_sae_path}")
        return load_qwenscope_sae(config.finetuned_sae_path, k=config.k, device=device)

    # Download pre-trained layer checkpoint if not already cached
    layer_path = download_layer_sae(config.sae_repo, config.layer, config.sae_cache_dir)
    return load_qwenscope_sae(layer_path, k=config.k, device=device)
