import os
import random
import sys
import hashlib
from contextlib import contextmanager
from types import ModuleType
from functools import partial
from typing import Any, Dict, List, Optional

import h5py
import torch
from tqdm import tqdm

from .base import SteeringMethod
from ..data_loader import ContrastivePair, PromptFormatter
from ..model_loader import ModelInfo, get_decoder_layers


def _load_steering_opt():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    steering_opt_path = os.path.join(repo_root, "llm-steering-opt")
    if steering_opt_path not in sys.path:
        sys.path.insert(0, steering_opt_path)

    if "mdmm" not in sys.modules:
        try:
            import mdmm  # noqa: F401
        except ModuleNotFoundError:
            sys.modules["mdmm"] = ModuleType("mdmm")

    import steering_opt  # type: ignore

    return steering_opt


@contextmanager
def _default_device(device: torch.device):
    if not hasattr(torch, "get_default_device") or not hasattr(torch, "set_default_device"):
        yield
        return

    previous_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(previous_device)


def _additive_pre_hook(
    module,
    inputs,
    vector: torch.Tensor,
    alpha: float,
    position: str,
):
    hidden_states = inputs[0]
    delta = alpha * vector.to(device=hidden_states.device, dtype=hidden_states.dtype)

    if position == "all":
        steered = hidden_states + delta
    elif position == "last":
        steered = hidden_states.clone()
        steered[:, -1, :] = steered[:, -1, :] + delta
    else:
        raise ValueError(f"Unknown opt steer position: {position}")

    return (steered,) + inputs[1:]


class OptimizedSteeringMethod(SteeringMethod):
    """
    Gradient-optimized steering vectors using the cloned llm-steering-opt code.

    This adapter keeps the same CAA Geometry interface: one vector per Schwartz
    value/layer, additive inference-time steering, and optional base activation
    caching for the existing layer-selection metrics.
    """

    def __init__(
        self,
        lr: float = 0.3,
        max_iters: int = 10,
        starting_norm: float = 1.0,
        max_norm: Optional[float] = None,
        n_training_samples: Optional[int] = None,
        seed: int = 42,
        steer_position: str = "all",
    ):
        self.lr = lr
        self.max_iters = max_iters
        self.starting_norm = starting_norm
        self.max_norm = max_norm
        self.n_training_samples = n_training_samples
        self.seed = seed
        self.steer_position = steer_position
        self.activations = {"pos": {}, "neg": {}}
        self.training_info: Dict[str, Dict[int, Dict[str, float]]] = {}

    def _sample_pairs(self, pairs: List[ContrastivePair], value_name: str) -> List[ContrastivePair]:
        if self.n_training_samples is None or self.n_training_samples >= len(pairs):
            return list(pairs)

        rng = random.Random(f"{self.seed}:{value_name}")
        return rng.sample(pairs, self.n_training_samples)

    def _cache_base_activations(
        self,
        contrastive_pairs: List[ContrastivePair],
        model_info: ModelInfo,
        layers: List[int],
        value_name: str,
    ) -> None:
        formatter = PromptFormatter(model_info.tokenizer, model_info.is_instruct)
        decoder_layers = get_decoder_layers(model_info)

        for polarity in ("pos", "neg"):
            for layer_idx in layers:
                self.activations[polarity].setdefault(layer_idx, {})

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
            for pair in tqdm(contrastive_pairs, desc=f"Caching activations {value_name}"):
                pos_tokens, neg_tokens = formatter.format_extraction_prompts(pair)

                current_activations.clear()
                input_ids = torch.tensor([pos_tokens]).to(model_info.device)
                with torch.no_grad():
                    model_info.model(input_ids)
                for layer_idx in layers:
                    self.activations["pos"][layer_idx][pair.sample_id] = current_activations[layer_idx]

                current_activations.clear()
                input_ids = torch.tensor([neg_tokens]).to(model_info.device)
                with torch.no_grad():
                    model_info.model(input_ids)
                for layer_idx in layers:
                    self.activations["neg"][layer_idx][pair.sample_id] = current_activations[layer_idx]
        finally:
            for handle in handles:
                handle.remove()

    def _make_datapoints(self, pairs: List[ContrastivePair], formatter: PromptFormatter):
        steering_opt = _load_steering_opt()
        datapoints = []
        for pair in pairs:
            prompt = formatter._format_base_prompt(pair.question)
            datapoints.append(
                steering_opt.TrainingDatapoint(
                    prompt=prompt,
                    src_completions=[" " + pair.negative_answer.lstrip()],
                    dst_completions=[" " + pair.positive_answer.lstrip()],
                )
            )
        return datapoints

    def compute_vectors(
        self,
        contrastive_pairs: List[ContrastivePair],
        model_info: ModelInfo,
        layers: List[int],
        value_name: str,
    ) -> Dict[int, torch.Tensor]:
        if not contrastive_pairs:
            raise ValueError(f"No contrastive pairs available for value '{value_name}'.")

        sampled_pairs = self._sample_pairs(contrastive_pairs, value_name)
        self._cache_base_activations(sampled_pairs, model_info, layers, value_name)

        formatter = PromptFormatter(model_info.tokenizer, model_info.is_instruct)
        datapoints = self._make_datapoints(sampled_pairs, formatter)
        steering_opt = _load_steering_opt()

        steering_vectors = {}
        self.training_info.setdefault(value_name, {})

        for layer_idx in layers:
            seed_material = f"{self.seed}:{value_name}:{layer_idx}".encode("utf-8")
            layer_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
            generator = torch.Generator(device=model_info.device)
            generator.manual_seed(layer_seed)
            starting_vec = torch.randn(
                model_info.hidden_dim,
                generator=generator,
                device=model_info.device,
                dtype=torch.float32,
            )
            starting_vec = self.starting_norm * starting_vec / starting_vec.norm().clamp_min(1e-12)

            print(
                f"Optimizing {value_name} at layer {layer_idx} "
                f"(n={len(datapoints)}, lr={self.lr}, max_iters={self.max_iters})"
            )
            with torch.enable_grad(), _default_device(model_info.device):
                vector, info = steering_opt.optimize_vector(
                    model_info.model,
                    datapoints,
                    layer_idx,
                    tokenizer=model_info.tokenizer,
                    lr=self.lr,
                    max_iters=self.max_iters,
                    max_norm=self.max_norm,
                    starting_norm=self.starting_norm,
                    starting_vec=starting_vec,
                    return_info=True,
                    show_iter_progress=False,
                    use_transformer_lens=False,
                )

            steering_vectors[layer_idx] = vector.detach().cpu().float()
            self.training_info[value_name][layer_idx] = {
                "iters": int(info.get("iters", -1)),
                "loss": float(info.get("loss", 0.0)),
                "norm": float(info.get("norm", steering_vectors[layer_idx].norm().item())),
                "n_training_samples": float(len(datapoints)),
            }

        return steering_vectors

    def apply(self, model_info: ModelInfo, layer_idx: int, vector: torch.Tensor, alpha: float) -> Any:
        decoder_layers = get_decoder_layers(model_info)
        hook_fn = partial(
            _additive_pre_hook,
            vector=vector.detach(),
            alpha=alpha,
            position=self.steer_position,
        )
        handle = decoder_layers[layer_idx].register_forward_pre_hook(hook_fn)
        return [handle]

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
