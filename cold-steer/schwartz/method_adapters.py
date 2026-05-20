"""Adapters that drive cold-steer's ``LossFDSteerer`` from Python.

Why an adapter?
    cold-steer's ``LossFDSteerer.__init__`` calls
    ``hydra.utils.instantiate(steerable_llm)`` which loads a *fresh* copy
    of the LLM. For our Schwartz benchmark we want to load the model
    once and train one steerer per value, so we subclass it and accept a
    preloaded ``SteerableLLM`` instance.

What does ``extract_representative_vector`` do?
    cold_fd does not produce an explicit per-layer steering vector. The
    intervention it applies at the chosen layer ``L`` is

        v(x) = (z(θ', x) − z(θ, x)) / ε     where  θ' = θ + ε·∇L

    For geometry analysis we need one vector per value. We take that
    direction at the last response token, averaged over the *training*
    prompts (the same prompts cold_fd uses), then send it to the geometry
    module exactly like the explicit-vector methods.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

import sys
import os

_THIS_DIR = os.path.dirname(__file__)
_COLD_STEER_ROOT = os.path.dirname(_THIS_DIR)
if _COLD_STEER_ROOT not in sys.path:
    sys.path.insert(0, _COLD_STEER_ROOT)

from src.steerer import BaseSteerer, KernelLossSteerer, LossFDSteerer  # noqa: E402
from src.llm import SteerableLLM  # noqa: E402

Steerer = Union["PreloadedLossFDSteerer", "PreloadedKernelLossSteerer"]


class PreloadedLossFDSteerer(LossFDSteerer):
    """LossFDSteerer that accepts an already-instantiated SteerableLLM.

    Bypasses ``hydra.utils.instantiate(steerable_llm)`` in the parent
    ``__init__`` so we can share one model across many values.
    """

    def __init__(
        self,
        steerable_llm: SteerableLLM,
        epsilon: float = 1e-6,
        eta: float = 1.0,
        training: str = "sft",
        training_batch_size: int = 1,
        test_batch_size: int = 1,
        log_dir: str = ".",
        steer_masking: str = "all",
        gen_masking: str = "prompt",
    ) -> None:
        BaseSteerer.__init__(
            self,
            steerable_llm,
            log_dir=log_dir,
            batch_size=test_batch_size,
            steer_masking=steer_masking,
            gen_masking=gen_masking,
        )
        self.epsilon = epsilon
        self.eta = eta
        self.training = training
        self.training_batch_size = training_batch_size
        self.z_eps = None
        self.steered_params: Optional[dict] = None

    @torch.no_grad()
    def extract_representative_vector(
        self,
        dataset,
        layer_idx: int,
    ) -> torch.Tensor:
        """Average activation displacement at the steering layer.

        Returns a 1-D tensor of shape ``(d_model,)`` on CPU, computed as

            (1/N) Σ_i  (z(θ', x_i) − z(θ, x_i))[..., last_response_token, :] / ε

        where ``θ' = self.steered_params`` (set by ``train()``).

        We use the **last non-pad token of the prompt+answer sequence**
        as the readout — that is the token where the model would emit
        its next prediction during inference, and the same position
        cold-steer's hook conceptually targets when ``steer_masking='last'``.
        """
        if self.steered_params is None:
            raise RuntimeError(
                "extract_representative_vector called before train(); "
                "self.steered_params is None."
            )
        if len(dataset) == 0:
            raise ValueError("Empty training dataset; cannot extract vector.")

        accum: Optional[torch.Tensor] = None
        count = 0
        for datum in DataLoader(dataset, batch_size=1, shuffle=False):
            # We forward the FULL prompt+answer ("matching") sequence so
            # that the displacement at the last token is informed by the
            # value-aligned answer, which is what the steerer's gradient
            # depends on.
            inputs = {
                "input_ids": datum["matching_input_ids"],
                "attention_mask": datum["matching_attention_mask"],
            }
            # 1. z(θ, x) — original activations
            with self.bypass_steering():
                self.layer_outputs = {}
                handles = self.steerable_llm.register_steering_hooks(
                    lambda lidx: lambda m, i, o: self._capture_hook(m, i, o, layer_idx=lidx)
                )
                try:
                    self.steerable_llm(**inputs)
                    z_orig = {k: v.detach() for k, v in self.layer_outputs.items()}
                finally:
                    for h in handles:
                        h.remove()

            # 2. z(θ', x) — activations under perturbed params (functional)
            z_eps_dict = self.get_intermediate_activations(
                params=self.steered_params, inputs=inputs
            )
            z_eps = {k: v.detach() for k, v in z_eps_dict.items()}

            if layer_idx not in z_orig or layer_idx not in z_eps:
                raise RuntimeError(
                    f"Layer {layer_idx} not captured. Captured original "
                    f"layers: {list(z_orig.keys())}, perturbed: {list(z_eps.keys())}"
                )

            # Last non-pad token of the matching sequence (left padded)
            attn = datum["matching_attention_mask"][0]
            # left-padded → the last index where attn==1 is simply -1
            last_pos = int(attn.shape[0] - 1)
            # But guard: if there is right-side padding too, walk back.
            while last_pos > 0 and attn[last_pos].item() == 0:
                last_pos -= 1

            diff = (z_eps[layer_idx][0, last_pos, :] - z_orig[layer_idx][0, last_pos, :]) / self.epsilon
            diff = diff.detach().to("cpu").float()
            if accum is None:
                accum = torch.zeros_like(diff)
            accum += diff
            count += 1

        assert accum is not None and count > 0
        return accum / count

    def _capture_hook(self, module, inp, out, layer_idx: int = -1):
        # Hybrid architectures (Qwen3-Next / Qwen3.5) return a bare tensor;
        # standard transformer layers return a tuple. Handle both.
        hidden = out[0] if isinstance(out, tuple) else out
        self.layer_outputs[layer_idx] = hidden
        return out


class PreloadedKernelLossSteerer(KernelLossSteerer):
    """KernelLossSteerer that accepts an already-instantiated SteerableLLM."""

    def __init__(
        self,
        steerable_llm: SteerableLLM,
        eta: float = 1.0,
        training: str = "sft",
        kernel: str = "constant",
        training_batch_size: int = 1,
        test_batch_size: int = 1,
        log_dir: str = ".",
        steer_masking: str = "all",
        gen_masking: str = "prompt",
    ) -> None:
        if kernel == "none":
            kernel = "constant"
        KernelLossSteerer.__init__(
            self,
            steerable_llm=steerable_llm,
            eta=eta,
            training_batch_size=training_batch_size,
            training=training,
            kernel=kernel,
            log_dir=log_dir,
            steer_masking=steer_masking,
            gen_masking=gen_masking,
        )
        self.loss_data: Optional[
            Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]
        ] = None

    @torch.no_grad()
    def extract_representative_vector(self, dataset, layer_idx: int) -> torch.Tensor:
        """Mean η·v_steer at the last token (same readout as ``steer_output_hook``)."""
        if self.loss_data is None:
            raise RuntimeError("extract_representative_vector called before train()")
        if len(dataset) == 0:
            raise ValueError("Empty training dataset; cannot extract vector.")

        accum: Optional[torch.Tensor] = None
        count = 0
        for datum in DataLoader(dataset, batch_size=1, shuffle=False):
            inputs = {
                "input_ids": datum["matching_input_ids"],
                "attention_mask": datum["matching_attention_mask"],
            }
            self.layer_outputs = {}
            handles = self.steerable_llm.register_steering_hooks(
                lambda lidx: lambda m, i, o: self._capture_hook(m, i, o, layer_idx=lidx)
            )
            try:
                self.steerable_llm(**inputs)
                activation = self.layer_outputs[layer_idx].detach()
            finally:
                for h in handles:
                    h.remove()

            delta = self._steering_delta_at_last_token(activation, inputs, layer_idx)
            accum = delta if accum is None else accum + delta
            count += 1

        assert accum is not None and count > 0
        return accum / count

    def _steering_delta_at_last_token(
        self, activation: torch.Tensor, inputs: dict, layer_idx: int
    ) -> torch.Tensor:
        kappa_t, loss_v = self.loss_data
        kappa = kappa_t[layer_idx].to(activation.device)
        loss_v_layer = loss_v[layer_idx].to(activation.device)
        z_last = activation[:, -1, :]

        vector = loss_v_layer if self.kernel == "entk_proj_loss" else None
        inputs_kappa = self.kernel_fn(
            output=z_last, inputs=inputs, vector=vector, layer_idx=layer_idx
        ).to(activation.device)
        if inputs_kappa.dim() == 1:
            inputs_kappa = inputs_kappa.unsqueeze(0)

        sim = torch.einsum("bd,Nd->bN", inputs_kappa, kappa)
        v_steer = torch.einsum("bN,Nd->bd", sim, loss_v_layer)
        return (self.eta * v_steer.squeeze(0)).detach().to("cpu").float()

    def _capture_hook(self, module, inp, out, layer_idx: int = -1):
        hidden = out[0] if isinstance(out, tuple) else out
        self.layer_outputs[layer_idx] = hidden
        return out


def make_steerer(
    method: str,
    steerable_llm: SteerableLLM,
    *,
    epsilon: float,
    eta: float,
    training: str,
    steer_masking: str,
    gen_masking: str,
    training_batch_size: int = 1,
    kernel: str = "constant",
    log_dir: str = ".",
) -> Steerer:
    if method == "cold_fd":
        return make_cold_fd_steerer(
            steerable_llm=steerable_llm,
            epsilon=epsilon,
            eta=eta,
            training=training,
            steer_masking=steer_masking,
            gen_masking=gen_masking,
            training_batch_size=training_batch_size,
            log_dir=log_dir,
        )
    if method == "cold_kernel":
        return make_cold_kernel_steerer(
            steerable_llm=steerable_llm,
            eta=eta,
            training=training,
            kernel=kernel,
            steer_masking=steer_masking,
            gen_masking=gen_masking,
            training_batch_size=training_batch_size,
            log_dir=log_dir,
        )
    raise ValueError(f"Unknown method {method!r}; use cold_fd or cold_kernel")


def make_cold_fd_steerer(
    steerable_llm: SteerableLLM,
    epsilon: float,
    eta: float,
    training: str,
    steer_masking: str,
    gen_masking: str,
    training_batch_size: int = 1,
    log_dir: str = ".",
) -> PreloadedLossFDSteerer:
    """Factory used by the pipeline."""
    return PreloadedLossFDSteerer(
        steerable_llm=steerable_llm,
        epsilon=epsilon,
        eta=eta,
        training=training,
        training_batch_size=training_batch_size,
        steer_masking=steer_masking,
        gen_masking=gen_masking,
        log_dir=log_dir,
    )


def make_cold_kernel_steerer(
    steerable_llm: SteerableLLM,
    eta: float,
    training: str,
    kernel: str,
    steer_masking: str,
    gen_masking: str,
    training_batch_size: int = 1,
    log_dir: str = ".",
) -> PreloadedKernelLossSteerer:
    return PreloadedKernelLossSteerer(
        steerable_llm=steerable_llm,
        eta=eta,
        training=training,
        kernel=kernel,
        training_batch_size=training_batch_size,
        steer_masking=steer_masking,
        gen_masking=gen_masking,
        log_dir=log_dir,
    )


def reset_steerer_state(steerer: BaseSteerer) -> None:
    """Clear ephemeral hook/activation cache."""
    if hasattr(steerer, "reset_steering"):
        steerer.reset_steering()
    if hasattr(steerer, "layer_outputs"):
        steerer.layer_outputs = {}


def offload_steerer_state(steerer: BaseSteerer) -> None:
    """Move trained steerer state to CPU between values."""
    if getattr(steerer, "steered_params", None) is not None:
        steerer.steered_params = {
            k: v.detach().cpu() for k, v in steerer.steered_params.items()
        }
    if getattr(steerer, "loss_data", None) is not None:
        kappa, loss_v = steerer.loss_data
        steerer.loss_data = (
            {k: v.detach().cpu() for k, v in kappa.items()},
            {k: v.detach().cpu() for k, v in loss_v.items()},
        )
    reset_steerer_state(steerer)


def load_steerer_state_to_device(steerer: BaseSteerer, device: torch.device) -> None:
    """Load CPU-cached steerer state onto ``device`` for eval forwards."""
    if getattr(steerer, "steered_params", None) is not None:
        steerer.steered_params = {
            k: v.to(device, non_blocking=True) for k, v in steerer.steered_params.items()
        }
    if getattr(steerer, "loss_data", None) is not None:
        kappa, loss_v = steerer.loss_data
        steerer.loss_data = (
            {k: v.to(device, non_blocking=True) for k, v in kappa.items()},
            {k: v.to(device, non_blocking=True) for k, v in loss_v.items()},
        )


offload_steered_params = offload_steerer_state
load_steered_params_to_device = load_steerer_state_to_device


def set_steering_layers(steerable_llm: SteerableLLM, layers: Sequence[int]) -> None:
    """Re-bind the SteerableLLM's steering layers in place.

    cold-steer's ``SteerableLLM`` caches steering layer indices, modules,
    and parameter slices at construction time. We need to update those
    when the layer-selection step picks a different layer than the one
    used during construction.
    """
    steerable_llm.steering_layer_indices = list(layers)
    steerable_llm.num_steering_layers = len(layers)
    model = steerable_llm.model
    steerable_llm.steering_layers = [
        model.model.layers[lidx - 1] for lidx in steerable_llm.steering_layer_indices
    ]
    steerable_llm.steering_params = [
        {
            k: v
            for k, v in steerable_llm.params.items()
            if ("model.embed" in k)
            or any(f"model.layers.{i}." in k for i in range(lidx + 1))
        }
        for lidx in steerable_llm.steering_layer_indices
    ]
    # Use `hidden_size` directly so we work on hybrid architectures
    # (Qwen3-Next / Qwen3.5) where some layers lack `self_attn`.
    hidden_size = model.config.hidden_size
    steerable_llm.steering_in_dims = [hidden_size for _ in steerable_llm.steering_layer_indices]
    steerable_llm.steering_out_dims = [hidden_size for _ in steerable_llm.steering_layer_indices]
