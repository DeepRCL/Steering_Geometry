"""
Sparse Autoencoder architecture and checkpoint loader.

Architecture (from the released checkpoint):
  Input (4096) → Encoder (4096 → 16384) + ReLU → Decoder (16384 → 4096)

The pre-encoder bias is subtracted before encoding and re-added after decoding,
which centres the distribution and allows the encoder to learn sparse features
relative to the mean activation.
"""
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Standard ReLU sparse autoencoder matching the released checkpoint format."""

    def __init__(self, d_in: int = 4096, d_sae: int = 16384):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d_in))        # pre-encoder bias
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)

    # ── Full forward ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (..., d_in)
        Returns:
            x_hat: (..., d_in)  – reconstruction
            z:     (..., d_sae) – sparse feature activations
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # ── Separate encode / decode ─────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project into feature space.  Returns sparse activations ≥ 0."""
        return torch.relu(self.encoder(x - self.bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from (possibly modified) feature activations."""
        return self.decoder(z) + self.bias


def load_sae(
    checkpoint_path: str,
    d_in: int = 4096,
    d_sae: int = 16384,
    device: str = "cpu",
) -> SparseAutoencoder:
    """Load a SparseAutoencoder from a .pt checkpoint.

    The checkpoint must contain a ``model_state_dict`` key, matching the format
    produced by the released training code.

    Args:
        checkpoint_path: Path to ``sae_base_best.pt`` (or similar).
        d_in:            Input / hidden dimension (4096 for Qwen 3.5 9B).
        d_sae:           Feature dimension (16384 by default – 4x expansion).
        device:          ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        Loaded SAE in eval mode on the requested device.
    """
    sae = SparseAutoencoder(d_in=d_in, d_sae=d_sae)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()
    print(f"SAE loaded: {checkpoint_path}  (d_in={d_in}, d_sae={d_sae}, device={device})")
    return sae
