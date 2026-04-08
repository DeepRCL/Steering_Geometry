import os
import torch
import h5py
from typing import List, Dict, Any
from tqdm import tqdm
from functools import partial

from .base import SteeringMethod
from ..data_loader import ContrastivePair, PromptFormatter
from ..model_loader import ModelInfo, get_decoder_layers

def _steering_hook(module, inputs, output, vector: torch.Tensor, alpha: float):
    """
    Hook to add the steering vector to the residual stream.

    PyTorch forward hooks registered via ``register_forward_hook`` receive
    ``(module, inputs, output)`` unless ``with_kwargs=True`` is explicitly
    enabled. The original implementation expected kwargs as a positional
    argument, which causes a runtime TypeError during the forward pass.

    output is either a tensor or a tuple where the first element is the hidden states.
    hidden_states shape: [batch, seq_len, hidden_dim]
    We want to add to all token positions after the prompt,
    but the simplest is adding to the last position or all positions.
    CAA usually adds to all positions starting from some point.
    To match the generic case without knowing prompt length, we add it to all positions.
    """
    is_tuple = isinstance(output, tuple)
    hidden_states = output[0] if is_tuple else output
    
    # vector shape: [hidden_dim]
    # hidden_states shape: [batch, seq_len, hidden_dim]
    hidden_states = hidden_states + (alpha * vector)
    
    if is_tuple:
        return (hidden_states,) + output[1:]
    return hidden_states

class CAASteeringMethod(SteeringMethod):
    def __init__(self):
        # Store activations as dict: {pos|neg: {layer_idx: {sample_id: tensor}}}
        self.activations = {"pos": {}, "neg": {}}
        
    def compute_vectors(self, 
                        contrastive_pairs: List[ContrastivePair], 
                        model_info: ModelInfo, 
                        layers: List[int],
                        value_name: str) -> Dict[int, torch.Tensor]:
        if not contrastive_pairs:
            raise ValueError(f"No contrastive pairs available for value '{value_name}'.")
        
        formatter = PromptFormatter(model_info.tokenizer, model_info.is_instruct)
        decoder_layers = get_decoder_layers(model_info)
        
        for p in ("pos", "neg"):
            for l in layers:
                self.activations[p].setdefault(l, {})

        pos_activations_mean = {l: [] for l in layers}
        neg_activations_mean = {l: [] for l in layers}
        
        # Cache for storing current forward pass activations
        current_activations = {}
        
        def get_activation_hook(layer_idx):
            def hook(module, inputs, output):
                # output is tuple (hidden_states, ...)
                hidden_states = output[0] if isinstance(output, tuple) else output
                # Store activations in float32 on CPU for HDF5 compatibility and
                # more stable downstream averaging/cosine computations.
                last_token_activation = hidden_states[0, -1, :].detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
                current_activations[layer_idx] = last_token_activation
            return hook

        # Register hooks for extraction
        handles = []
        for layer_idx in layers:
            handle = decoder_layers[layer_idx].register_forward_hook(get_activation_hook(layer_idx))
            handles.append(handle)
            
        model_info.model.eval()
        
        for pair in tqdm(contrastive_pairs, desc=f"Extracting {value_name}"):
            pos_tokens, neg_tokens = formatter.format_extraction_prompts(pair)
            
            # --- POSITIVE ---
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
                pos_activations_mean[layer_idx].append(act)
                
            # --- NEGATIVE ---
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
                neg_activations_mean[layer_idx].append(act)

        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Compute mean difference
        steering_vectors = {}
        for layer_idx in layers:
            pos_mean = torch.stack(pos_activations_mean[layer_idx]).mean(dim=0)
            neg_mean = torch.stack(neg_activations_mean[layer_idx]).mean(dim=0)
            steering_vectors[layer_idx] = pos_mean - neg_mean
            
        return steering_vectors

    def apply(self, model_info: ModelInfo, layer_idx: int, vector: torch.Tensor, alpha: float) -> Any:
        decoder_layers = get_decoder_layers(model_info)
        
        # Ensure vector is normalized
        vec_norm = vector.norm()
        if vec_norm > 0:
            vector_norm = vector / vec_norm
        else:
            vector_norm = vector
            
        target_vec = vector_norm.to(model_info.device).to(model_info.model.dtype)
        
        hook_fn = partial(_steering_hook, vector=target_vec, alpha=alpha)
        handle = decoder_layers[layer_idx].register_forward_hook(hook_fn)
        return [handle]
        
    def cleanup(self, handles: Any):
        for handle in handles:
            handle.remove()

    def save_activations(self, output_dir: str, value_name: str):
        from ..config import safe_name
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{safe_name(value_name)}.h5")
        
        with h5py.File(filename, "w") as f:
            for p in ["pos", "neg"]:
                grp_p = f.create_group(p)
                for l_idx, sample_dict in self.activations[p].items():
                    grp_l = grp_p.create_group(f"layer_{l_idx}")
                    for sample_id, tensor in sample_dict.items():
                        grp_l.create_dataset(
                            str(sample_id),
                            data=tensor.detach().to(dtype=torch.float32).numpy(),
                        )

    def clear_activations(self):
        self.activations = {"pos": {}, "neg": {}}
