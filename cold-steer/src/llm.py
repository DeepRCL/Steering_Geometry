from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from torch.func import functional_call
from typing import Union, List
import torch

class SteerableLLM(nn.Module):
    def __init__(
        self, 
        model_name: str = "/workingdir/ksharma323/llama7b",
        steering_layer_indices: List[int] = [10],
        temperature: float = 0.7,
    ):
        super().__init__()
        # Load model & tokenizer
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='balanced')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', add_eos_token=False)
        self.tokenizer.padding_side = 'left'
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sampling_temperature = temperature
        
        self.params = {k: v.detach() for k, v in self.model.named_parameters()}
        
        self.steering_layer_indices = steering_layer_indices
        self.num_steering_layers = len(steering_layer_indices)
        
        # Llama-shaped architectures expose `model.model.embed_tokens` and
        # `model.model.layers[i].self_attn.{q,o}_proj`. Qwen2/Qwen3, Llama-3,
        # Llama-2 and Mistral-v0.1 all match this layout, so we treat them
        # uniformly. Add new families here only if their module path differs.
        name = self.model_name
        llama_shaped = (
            'llama7b' in name
            or 'Llama-2-7b' in name
            or 'Llama-3' in name
            or 'Meta-Llama-3' in name
            or 'gemma-2' in name
            or ('Mistral-7B' in name and 'v0.1' in name)
            or 'Qwen' in name
        )
        if llama_shaped:
            self.input_layer = self.model.model.embed_tokens
            self.steering_layers = [self.model.model.layers[layer_idx-1] for layer_idx in steering_layer_indices]
            self.steering_params = [{k: v for k, v in self.params.items() if ('model.embed' in k) or (any([(f'model.layers.{i}.' in k) \
                                        for i in range(layer_idx+1)]))} for layer_idx in steering_layer_indices]

            # Residual-stream dim. The original code read this off
            # `self_attn.{q,o}_proj`, but on hybrid architectures (e.g. Qwen3-Next / Qwen3.5)
            # some decoder blocks are linear-attention layers without a `self_attn` module.
            # `hidden_size` is identical for all standard layer types and survives the swap.
            hidden_size = self.model.config.hidden_size
            self.steering_in_dims = [hidden_size for _ in steering_layer_indices]
            self.steering_out_dims = [hidden_size for _ in steering_layer_indices]
        else:
            raise NotImplementedError(
                f"SteerableLLM does not know the module layout for '{name}'. "
                "Extend src/llm.py with the appropriate branch."
            )
            
        
    def get_layers_params_steering(self, layers):
        return [{k: v for k, v in params.items() if any([(f'model.layers.{i}.' in k) for i in layers_i])} \
                    for layers_i, params in zip(layers, self.steering_params)]
        
    def set_steering_params(self, params=None, requires_grad=False):
        if params is not None:
            for k, v in params.items():
                self.params[k] = v
            
        for k, v in self.model.named_parameters():
            for param in self.steering_params:
                if k in param:
                    v.requires_grad_(requires_grad)
                 
    def get_steering_params(self):
        return [{k: v for k, v in self.model.named_parameters() if k in params_steering} for params_steering in self.steering_params]

    def get_params(self, keys):
        return [v for k, v in self.model.named_parameters() if k in keys]

    def functional_forward(self, params=None, inputs=None):
        return functional_call(self.model, self.params if params is None else params, 
                               kwargs={k: v for k, v in inputs.items()} if inputs is None else inputs)
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def generate(self, decode=True, *args, **kwargs):
        generated_ids = self.model.generate(*args, **kwargs)
        prompt_length = kwargs['input_ids'].shape[1]
        
        if decode:
            return self.tokenizer.batch_decode(generated_ids[:, prompt_length:])
        return generated_ids[:, prompt_length:]
        
    def register_input_hook(self, hook_fn):
        return self.input_layer.register_forward_hook(hook_fn)
            
    def make_peft(self, peft_config):
        from peft import get_peft_model
        self.model = get_peft_model(self.model, peft_config)
        
    def register_steering_hooks(self, hook_fn):
        return [layer.register_forward_hook(hook_fn(layer_idx)) for layer_idx, layer in zip(self.steering_layer_indices, self.steering_layers)]
            