import logging
from functools import partial
from typing import List

import torch

from .model_loader import ModelInfo, get_decoder_layers
from .steering.caa import _steering_hook

logger = logging.getLogger(__name__)


def add_steering_hooks(
    model_info: ModelInfo,
    vector: torch.Tensor,
    alpha: float,
    layer_indices: List[int],
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register steering hooks on a set of decoder layers."""
    handles = []
    layers = get_decoder_layers(model_info)

    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(layers):
            logger.warning("Layer %s out of range, skipping", layer_idx)
            continue

        layer_vec = vector[layer_idx + 1].float()
        vec_norm = layer_vec.norm()
        if vec_norm > 0:
            layer_vec = layer_vec / vec_norm

        logger.info("Layer %s: raw norm=%.2f, injecting alpha=%.2f", layer_idx, vec_norm, alpha)
        hook_fn = partial(
            _steering_hook,
            vector=layer_vec.to(model_info.device).to(model_info.model.dtype),
            alpha=alpha,
        )
        handles.append(layers[layer_idx].register_forward_hook(hook_fn))

    return handles
