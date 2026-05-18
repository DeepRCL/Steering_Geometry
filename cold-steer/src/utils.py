import torch.nn as nn
import torch
import json
from typing import Union, Dict, List, Optional
import torch.nn.functional as F

def partial_cross_entropy(logits: torch.Tensor, partial_targets: Dict[int, float], 
                            vocab_size: Optional[int] = None) -> torch.Tensor:
    """
    Args:
        logits: (batch_size, vocab_size) - model predictions
        partial_targets: dict mapping class_idx -> probability for known classes
        vocab_size: total vocabulary size (inferred from logits if not provided)
    Returns:
        loss: scalar tensor
    """
    if vocab_size is None:
        vocab_size = logits.size(-1)
        
    batch_size = logits.size(0)
    device = logits.device
    
    # Convert partial targets to full probability vectors
    losses = []
    
    for batch_idx in range(batch_size):
        keys = [k for k, prob in partial_targets.items() if prob[batch_idx] != -1]
        known_targets = torch.tensor([partial_targets[k] for k in keys]).to(device)
        known_logits = logits[batch_idx, -1, keys]  # (K,)
        # Compute cross-entropy over known classes
        log_probs = F.log_softmax(known_logits, dim=-1)
        loss = -(known_targets * log_probs).sum()
        losses.append(loss)

    losses = torch.stack(losses)
    
    return losses.mean()

def lora_style_update(grads_dict, rank=16):
    low_rank_updates = {}
    for k, grad in grads_dict.items():
        if grad.dim() == 2 and min(grad.shape) > rank:
            # Factorize gradient as A @ B
            d_in, d_out = grad.shape
            A = torch.randn(d_in, rank, device=grad.device) * 0.01
            B = torch.linalg.lstsq(A, grad).solution  # Solve A @ B ≈ grad
            low_rank_updates[k] = (A, B)
        else:
            low_rank_updates[k] = grad
    return low_rank_updates


class ConstantIntervention(nn.Module):
    def __init__(self, dim, device) -> None:
        super().__init__()
        self.intervention = torch.nn.Parameter(torch.zeros(dim), requires_grad=True)
        
        self.to(device)
        
        
    def forward(self, *args, **kwargs):
        return self.intervention
    

def subset_mask(input_ids, subset_ids, nprev=0):
    mask = torch.zeros_like(input_ids).bool()
    for i in range(0, len(input_ids)):
        if torch.equal(input_ids[i: i+len(subset_ids)], subset_ids): 
            mask[i: i+len(subset_ids)] = True
            mask[i-nprev:i] = True
            return mask
    # remove starting
    return subset_mask(input_ids, subset_ids[1:], nprev=nprev+1)
        
def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    with open(f"{metadata_path}/metadata.jsonl", 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref = data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": data["concept_id"]
            }
            metadata += [flatten_data]  # Return the metadata as is
    return metadata
