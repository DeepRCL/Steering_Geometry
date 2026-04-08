from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from ..data_loader import ContrastivePair
from ..model_loader import ModelInfo

class SteeringMethod(ABC):
    @abstractmethod
    def compute_vectors(self, 
                        contrastive_pairs: List[ContrastivePair], 
                        model_info: ModelInfo, 
                        layers: List[int],
                        value_name: str) -> Dict[int, torch.Tensor]:
        """
        Given contrastive pairs for a value, a model, and target layers,
        return {layer_idx: steering_vector_tensor}.
        Also stores per-sample activations internally for later saving.
        """
        pass

    @abstractmethod
    def apply(self, model_info: ModelInfo, layer_idx: int, vector: torch.Tensor, alpha: float) -> Any:
        """
        Install hooks to add alpha * vector to the residual stream at layer_idx.
        Returns hook handles (or similar object) for cleanup via cleanup().
        """
        pass
        
    @abstractmethod
    def cleanup(self, handles: Any):
        """
        Remove hooks installed by apply().
        """
        pass

    @abstractmethod
    def save_activations(self, output_dir: str, value_name: str):
        """Save stored per-sample activations to disk."""
        pass
        
    @abstractmethod
    def clear_activations(self):
        """Clear memory of stored activations."""
        pass
