import os
from functools import partial
from typing import Any, Dict, List

import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import SteeringMethod
from ..data_loader import ContrastivePair, PromptFormatter
from ..model_loader import ModelInfo, get_decoder_layers


def _safe_unit(vector: torch.Tensor) -> torch.Tensor:
    norm = vector.norm().clamp_min(1e-12)
    return vector / norm


def _spherical_update(
    hidden_states: torch.Tensor,
    mu_t: torch.Tensor,
    kappa: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    orig_dtype = hidden_states.dtype
    x = hidden_states.float()
    mu_t = _safe_unit(mu_t.to(device=x.device, dtype=torch.float32))
    mu_h = -mu_t

    orig_norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    x_hat = x / orig_norm

    cos_t = (x_hat * mu_t).sum(dim=-1).clamp(-1.0, 1.0)
    cos_h = (x_hat * mu_h).sum(dim=-1).clamp(-1.0, 1.0)
    logits = torch.stack([kappa * cos_t, kappa * cos_h], dim=-1)
    probs = F.softmax(logits, dim=-1)

    delta = probs[..., 1] - probs[..., 0]
    trigger_mask = delta > beta
    if not trigger_mask.any():
        return hidden_states

    denom = max(1e-6, 1.0 - beta)
    t = alpha * (delta - beta) / denom
    t = torch.clamp(t, 0.0, 1.0)

    theta = torch.acos(cos_t)
    valid_mask = trigger_mask & (theta >= 1e-4)
    if not valid_mask.any():
        return hidden_states

    sin_theta = torch.sin(theta).clamp_min(1e-12)
    tangent_dir = (x_hat - cos_t.unsqueeze(-1) * mu_t) / sin_theta.unsqueeze(-1)
    theta_new = (1.0 - t) * theta

    x_new_hat = (
        torch.cos(theta_new).unsqueeze(-1) * mu_t
        + torch.sin(theta_new).unsqueeze(-1) * tangent_dir
    )
    candidate = x_new_hat * orig_norm

    x_new = x.clone()
    x_new[valid_mask] = candidate[valid_mask]
    return x_new.to(orig_dtype)


def _spherical_steering_hook(
    module,
    inputs,
    output,
    mu_t: torch.Tensor,
    kappa: float,
    alpha: float,
    beta: float,
    position: str,
):
    is_tuple = isinstance(output, tuple)
    hidden_states = output[0] if is_tuple else output
    if position == "all":
        steered = _spherical_update(hidden_states, mu_t, kappa, alpha, beta)
    elif position == "last":
        steered = hidden_states.clone()
        steered[:, -1, :] = _spherical_update(
            hidden_states[:, -1, :],
            mu_t,
            kappa,
            alpha,
            beta,
        )
    else:
        raise ValueError(f"Unknown spherical steer position: {position}")
    if is_tuple:
        return (steered,) + output[1:]
    return steered


class SphericalSteeringMethod(SteeringMethod):
    """
    Cobras/SphericalSteer-style geodesic steering for value vectors.

    Training stores the truthful prototype mu_T = normalize(mean(pos) - mean(neg)).
    Evaluation rotates activations on the sphere toward mu_T when the antipodal
    prototype has sufficiently higher vMF probability.
    """

    def __init__(
        self,
        kappa: float = 20.0,
        beta: float = -0.15,
        steer_position: str = "last",
    ):
        self.kappa = kappa
        self.beta = beta
        self.steer_position = steer_position
        self.activations = {"pos": {}, "neg": {}}

    def compute_vectors(
        self,
        contrastive_pairs: List[ContrastivePair],
        model_info: ModelInfo,
        layers: List[int],
        value_name: str,
    ) -> Dict[int, torch.Tensor]:
        if not contrastive_pairs:
            raise ValueError(f"No contrastive pairs available for value '{value_name}'.")

        formatter = PromptFormatter(model_info.tokenizer, model_info.is_instruct)
        decoder_layers = get_decoder_layers(model_info)

        for polarity in ("pos", "neg"):
            for layer_idx in layers:
                self.activations[polarity].setdefault(layer_idx, {})

        pos_activations = {layer_idx: [] for layer_idx in layers}
        neg_activations = {layer_idx: [] for layer_idx in layers}
        current_activations = {}

        def get_activation_hook(layer_idx):
            def hook(module, inputs, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                current_activations[layer_idx] = hidden_states[0, -1, :].detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )

            return hook

        handles = [
            decoder_layers[layer_idx].register_forward_hook(get_activation_hook(layer_idx))
            for layer_idx in layers
        ]

        model_info.model.eval()
        try:
            for pair in tqdm(contrastive_pairs, desc=f"Extracting {value_name}"):
                pos_tokens, neg_tokens = formatter.format_extraction_prompts(pair)

                current_activations.clear()
                input_ids = torch.tensor([pos_tokens]).to(model_info.device)
                with torch.no_grad():
                    model_info.model(input_ids)
                for layer_idx in layers:
                    if layer_idx not in current_activations:
                        raise RuntimeError(
                            f"Activation hook did not fire for layer {layer_idx} on "
                            f"positive prompt for sample_id={pair.sample_id}."
                        )
                    act = current_activations[layer_idx]
                    self.activations["pos"][layer_idx][pair.sample_id] = act
                    pos_activations[layer_idx].append(act)

                current_activations.clear()
                input_ids = torch.tensor([neg_tokens]).to(model_info.device)
                with torch.no_grad():
                    model_info.model(input_ids)
                for layer_idx in layers:
                    if layer_idx not in current_activations:
                        raise RuntimeError(
                            f"Activation hook did not fire for layer {layer_idx} on "
                            f"negative prompt for sample_id={pair.sample_id}."
                        )
                    act = current_activations[layer_idx]
                    self.activations["neg"][layer_idx][pair.sample_id] = act
                    neg_activations[layer_idx].append(act)
        finally:
            for handle in handles:
                handle.remove()

        steering_vectors = {}
        for layer_idx in layers:
            pos_mean = torch.stack(pos_activations[layer_idx]).mean(dim=0)
            neg_mean = torch.stack(neg_activations[layer_idx]).mean(dim=0)
            diff = pos_mean - neg_mean
            steering_vectors[layer_idx] = _safe_unit(diff)

        return steering_vectors

    def apply(
        self,
        model_info: ModelInfo,
        layer_idx: int,
        vector: torch.Tensor,
        alpha: float,
    ) -> Any:
        decoder_layers = get_decoder_layers(model_info)
        mu_t = _safe_unit(vector.detach()).to(model_info.device)
        hook_fn = partial(
            _spherical_steering_hook,
            mu_t=mu_t,
            kappa=self.kappa,
            alpha=alpha,
            beta=self.beta,
            position=self.steer_position,
        )
        handle = decoder_layers[layer_idx].register_forward_hook(hook_fn)
        return [handle]

    def compute_displacement_vectors(
        self,
        vectors: Dict[str, torch.Tensor],
        activations: Dict[str, Dict[str, Dict[int, Dict[str, torch.Tensor]]]],
        layer_idx: int,
        alpha: float,
        source: str = "neg",
    ) -> Dict[str, torch.Tensor]:
        from ..config import SCHWARTZ_CIRCUMPLEX_ORDER

        displacement_vectors = {}
        for value in SCHWARTZ_CIRCUMPLEX_ORDER:
            if source == "all":
                samples = []
                for polarity in ("pos", "neg"):
                    samples.extend(activations[value][polarity][layer_idx].values())
            else:
                samples = list(activations[value][source][layer_idx].values())

            if not samples:
                raise ValueError(
                    f"No {source} activations found for {value} at layer {layer_idx}."
                )

            x = torch.stack([sample.float() for sample in samples])
            mu_t = _safe_unit(vectors[value].detach().cpu().float())
            x_steered = _spherical_update(x, mu_t, self.kappa, alpha, self.beta)
            displacement_vectors[value] = (x_steered - x).mean(dim=0)

        return displacement_vectors

    def cleanup(self, handles: Any):
        for handle in handles:
            handle.remove()

    def save_activations(self, output_dir: str, value_name: str):
        from ..config import safe_name

        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{safe_name(value_name)}.h5")

        with h5py.File(filename, "w") as f:
            for polarity in ["pos", "neg"]:
                grp_p = f.create_group(polarity)
                for layer_idx, sample_dict in self.activations[polarity].items():
                    grp_l = grp_p.create_group(f"layer_{layer_idx}")
                    for sample_id, tensor in sample_dict.items():
                        grp_l.create_dataset(
                            str(sample_id),
                            data=tensor.detach().to(dtype=torch.float32).numpy(),
                        )

    def clear_activations(self):
        self.activations = {"pos": {}, "neg": {}}
