import torch
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any

@dataclass
class ModelInfo:
    model: Any
    tokenizer: Any
    family: str
    n_layers: int
    hidden_dim: int
    is_instruct: bool
    device: torch.device

def load_model(model_name: str, device: str = "auto") -> ModelInfo:
    print(f"Loading model: {model_name}...")
    
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "auto" else device
    )
    
    # Auto-detect family
    name_lower = model_name.lower()
    
    if "qwen" in name_lower:
        family = "qwen3"
        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        is_instruct = "base" not in name_lower
    elif "gemma" in name_lower:
        family = "gemma3"
        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        is_instruct = "pt" not in name_lower # Gemma-3 uses 'pt' for pre-trained (base), 'it' for instruct
    else:
        # Fallback based on config
        family = "unknown"
        n_layers = getattr(model.config, "num_hidden_layers", 0)
        hidden_dim = getattr(model.config, "hidden_size", 0)
        is_instruct = "base" not in name_lower
        
    print(f"Model loaded. Family: {family}, Layers: {n_layers}, Hidden Dim: {hidden_dim}, Instruct: {is_instruct}")
        
    return ModelInfo(
        model=model,
        tokenizer=tokenizer,
        family=family,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        is_instruct=is_instruct,
        device=device_obj
    )
    
def get_decoder_layers(model_info: ModelInfo) -> list:
    """Returns the list of transformer decoder layers where we can attach hooks."""
    # Both Qwen3 and Gemma (and LLaMA) use `model.layers`
    if hasattr(model_info.model, "model") and hasattr(model_info.model.model, "layers"):
        return model_info.model.model.layers
    elif hasattr(model_info.model, "layers"):
        return model_info.model.layers
    else:
        raise ValueError(f"Could not find decoder layers in model. Available attributes: {dir(model_info.model)}")
