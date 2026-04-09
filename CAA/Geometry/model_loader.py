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


def _get_text_config(config: Any) -> Any:
    """Return the language-model config, handling multimodal wrapper configs like Gemma 3."""
    return getattr(config, "text_config", config)


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
        dtype=torch.bfloat16,
        device_map="auto" if device == "auto" else device
    )
    
    # Auto-detect family
    name_lower = model_name.lower()
    text_config = _get_text_config(model.config)
    
    tokenizer_has_chat_template = bool(getattr(tokenizer, "chat_template", None))

    if "qwen" in name_lower:
        family = "qwen3"
        n_layers = getattr(text_config, "num_hidden_layers", getattr(model.config, "num_hidden_layers", 0))
        hidden_dim = getattr(text_config, "hidden_size", getattr(model.config, "hidden_size", 0))
        is_instruct = "base" not in name_lower
    elif "gemma" in name_lower:
        family = "gemma4" if "gemma-4" in name_lower else "gemma3"
        n_layers = getattr(text_config, "num_hidden_layers", getattr(model.config, "num_hidden_layers", 0))
        hidden_dim = getattr(text_config, "hidden_size", getattr(model.config, "hidden_size", 0))
        if "-it" in name_lower:
            is_instruct = True
        elif "-pt" in name_lower:
            is_instruct = False
        else:
            is_instruct = tokenizer_has_chat_template
    else:
        # Fallback based on config
        family = "unknown"
        n_layers = getattr(text_config, "num_hidden_layers", getattr(model.config, "num_hidden_layers", 0))
        hidden_dim = getattr(text_config, "hidden_size", getattr(model.config, "hidden_size", 0))
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
    # Qwen and LLaMA-style causal LMs commonly use `model.layers`.
    if hasattr(model_info.model, "model") and hasattr(model_info.model.model, "layers"):
        return model_info.model.model.layers
    # Gemma 3 conditional generation wraps the text stack under `model.language_model`.
    elif hasattr(model_info.model, "model") and hasattr(model_info.model.model, "language_model"):
        language_model = model_info.model.model.language_model
        if hasattr(language_model, "layers"):
            return language_model.layers
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            return language_model.model.layers
    # Some multimodal/text wrappers expose the language model directly.
    elif hasattr(model_info.model, "language_model"):
        language_model = model_info.model.language_model
        if hasattr(language_model, "layers"):
            return language_model.layers
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            return language_model.model.layers
    elif hasattr(model_info.model, "layers"):
        return model_info.model.layers
    else:
        raise ValueError(f"Could not find decoder layers in model. Available attributes: {dir(model_info.model)}")
