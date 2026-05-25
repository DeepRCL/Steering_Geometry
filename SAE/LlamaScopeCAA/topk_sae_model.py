"""
TopK Sparse Autoencoder matching the Llama-Scope checkpoint format.

Llama-Scope checkpoint layout:
  encoder.weight : (d_sae, d_in)
  encoder.bias   : (d_sae,)
  decoder.weight : (d_in, d_sae)
  decoder.bias   : (d_in,)

Encoding (matches the Llama-Scope demo exactly):
  pre  = x @ W_enc.T + b_enc          # (..., d_sae)
  keep top-k values, zero the rest
  z    = sparse feature activations    # (..., d_sae)

Decoding:
  x_hat = z @ W_dec.T + b_dec         # (..., d_in)

The TopK constraint replaces L1 regularisation: sparsity is enforced
structurally rather than via a penalty term, so fine-tuning uses MSE only.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class TopKSparseAutoencoder(nn.Module):
    """
    TopK SAE matching the Llama-Scope architecture.

    Parameters
    ----------
    d_in  : input dimension (4096 for Llama-3.1-8B)
    d_sae : feature dimension (32768 for Llama-Scope 8x)
    k     : number of active features kept per token (50 for Llama-Scope)
    """

    def __init__(
        self,
        d_in: int = 4096,
        d_sae: int = 32768,
        k: int = 50,
        activation_fn: str = "jumprelu",
        jump_relu_threshold: float | None = None,
    ):
        super().__init__()
        self.k = k
        self.activation_fn = activation_fn
        self.jump_relu_threshold = jump_relu_threshold
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
          1. ~99.9% of entries are exactly zero (only 50/32768 per sample),
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

    def sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        """Apply the sparsifier used by the loaded Llama-Scope SAE."""
        if self.activation_fn == "jumprelu":
            threshold = 0.0 if self.jump_relu_threshold is None else self.jump_relu_threshold
            return pre * (pre > threshold).to(pre.dtype)

        if self.activation_fn == "relu_topk":
            pre = torch.relu(pre)

        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

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
        return self.sparsify(pre)

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
def load_llamascope_sae(
    path: str,
    k: int = 50,
    device: str = "cpu",
    hyperparams_path: str | None = None,
) -> TopKSparseAutoencoder:
    """
    Load a TopKSparseAutoencoder from a Llama-Scope checkpoint.

    Pre-trained checkpoints are safetensors files with keys
    encoder.weight, encoder.bias, decoder.weight, decoder.bias. Fine-tuned
    checkpoints saved by this pipeline are torch files with W_enc-style keys.

    Args:
        path   : Path to a Llama-Scope ``final.safetensors`` or fine-tuned .pt
        k      : TopK budget (must match the SAE's training configuration)
        device : ``"cpu"``, ``"cuda"``, or ``"mps"``

    Returns:
        Loaded SAE in eval mode on the requested device.
    """
    activation_fn = "topk"
    jump_relu_threshold = None
    if hyperparams_path is None:
        candidate = os.path.join(os.path.dirname(os.path.dirname(path)), "hyperparams.json")
        if os.path.exists(candidate):
            hyperparams_path = candidate
    if hyperparams_path and os.path.exists(hyperparams_path):
        with open(hyperparams_path, encoding="utf-8") as f:
            hyperparams = json.load(f)
        activation_fn = str(hyperparams.get("act_fn", activation_fn)).lower()
        jump_relu_threshold = hyperparams.get("jump_relu_threshold")
        k = int(hyperparams.get("top_k", k))

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        ckpt: dict = load_file(path, device=device)
        W_enc = ckpt["encoder.weight"]
        b_enc = ckpt["encoder.bias"]
        W_dec = ckpt["decoder.weight"]
        b_dec = ckpt["decoder.bias"]
    else:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        W_enc = ckpt["W_enc"]
        b_enc = ckpt["b_enc"]
        W_dec = ckpt["W_dec"]
        b_dec = ckpt["b_dec"]
        activation_fn = ckpt.get("activation_fn", activation_fn)
        jump_relu_threshold = ckpt.get("jump_relu_threshold", jump_relu_threshold)

    d_sae, d_in = W_enc.shape

    sae = TopKSparseAutoencoder(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        activation_fn=activation_fn,
        jump_relu_threshold=jump_relu_threshold,
    )
    sae.encoder.weight.data = W_enc.float()
    sae.encoder.bias.data   = b_enc.float()
    sae.decoder.weight.data = W_dec.float()
    sae.decoder.bias.data   = b_dec.float()

    sae = sae.to(device)
    sae.eval()
    print(
        f"Llama-Scope SAE loaded: {path}  "
        f"(d_in={d_in}, d_sae={d_sae}, k={k}, act={activation_fn}, "
        f"threshold={jump_relu_threshold}, device={device})"
    )
    return sae


def save_llamascope_sae(sae: TopKSparseAutoencoder, path: str) -> None:
    """
    Save a TopKSparseAutoencoder in the Llama-Scope dict format so that
    load_llamascope_sae can reload it after fine-tuning.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "W_enc": sae.encoder.weight.data.cpu(),
        "b_enc": sae.encoder.bias.data.cpu(),
        "W_dec": sae.decoder.weight.data.cpu(),
        "b_dec": sae.decoder.bias.data.cpu(),
        "activation_fn": sae.activation_fn,
        "jump_relu_threshold": sae.jump_relu_threshold,
        "k": sae.k,
    }
    torch.save(ckpt, path)
    print(f"Fine-tuned Llama-Scope SAE saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace Hub download helper
# ──────────────────────────────────────────────────────────────────────────────
def download_layer_sae(
    repo_id: str,
    layer: int,
    cache_dir: str,
    expansion: int = 8,
    site: str = "R",
) -> str:
    """
    Download a layer SAE from a HuggingFace Hub repo.

    The file is cached locally; subsequent calls return immediately without
    re-downloading.

    Args:
        repo_id   : HuggingFace repo, e.g. ``"OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x"``
        layer     : Layer index (0–31 for Llama-Scope)
        cache_dir : Directory to store downloaded .pt files

    Returns:
        Absolute path to the local .pt file.
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)
    filename = (
        f"Llama3_1-8B-Base-L{layer}{site}-{expansion}x/"
        "checkpoints/final.safetensors"
    )

    print(f"  Downloading/cache-checking {repo_id}/{filename} …")
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )
    hf_hub_download(
        repo_id=repo_id,
        filename=f"Llama3_1-8B-Base-L{layer}{site}-{expansion}x/hyperparams.json",
        cache_dir=cache_dir,
    )
    return checkpoint_path


def get_or_download_sae(
    config,
    device: str = "cpu",
    use_finetuned: bool = True,
) -> TopKSparseAutoencoder:
    """
    Convenience: return the fine-tuned SAE if it exists (and use_finetuned),
    otherwise download and load the pre-trained Llama-Scope checkpoint.

    Args:
        config       : LlamaScopePipelineConfig instance
        device       : Target device string
        use_finetuned: If True, prefer config.finetuned_sae_path when present.
    """
    if use_finetuned and os.path.exists(config.finetuned_sae_path):
        print(f"Loading fine-tuned SAE from {config.finetuned_sae_path}")
        return load_llamascope_sae(config.finetuned_sae_path, k=config.k, device=device)

    # Download pre-trained layer checkpoint if not already cached
    layer_path = download_layer_sae(
        config.sae_repo,
        config.layer,
        config.sae_cache_dir,
        expansion=getattr(config, "sae_expansion", 8),
        site=getattr(config, "sae_site", "R"),
    )
    return load_llamascope_sae(layer_path, k=config.k, device=device)
    def sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        if self.activation_fn == "jumprelu":
            threshold = 0.0 if self.jump_relu_threshold is None else self.jump_relu_threshold
            return pre * (pre > threshold).to(pre.dtype)
        if self.activation_fn == "relu_topk":
            pre = torch.relu(pre)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z
