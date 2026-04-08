import os
import json
import torch
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from .config import SCHWARTZ_CIRCUMPLEX_ORDER

def compute_cosine_consistency(activations_pos: Dict[str, Dict[str, torch.Tensor]], 
                               activations_neg: Dict[str, Dict[str, torch.Tensor]]) -> float:
    """
    Computes the mean pairwise cosine similarity of per-sample difference vectors
    for a single value category at a single layer.
    """
    # dict: sample_id -> tensor
    sample_ids = list(activations_pos.keys())
    if len(sample_ids) < 2:
        return 0.0
        
    diff_vectors = []
    for sid in sample_ids:
        if sid in activations_neg:
            d = activations_pos[sid] - activations_neg[sid]
            diff_vectors.append(d)
            
    if len(diff_vectors) < 2:
        return 0.0
        
    # Stack and normalize
    diffs = torch.stack(diff_vectors)
    norms = diffs.norm(dim=1, keepdim=True)
    norms[norms == 0] = 1.0
    normalized_diffs = diffs / norms
    
    # Compute dot products
    sim_matrix = torch.mm(normalized_diffs, normalized_diffs.t())
    
    # Get upper triangle excluding diagonal
    upper_tri = torch.triu(sim_matrix, diagonal=1)
    num_pairs = (len(diffs) * (len(diffs) - 1)) / 2
    
    return float(upper_tri.sum() / num_pairs)

def select_layer(config, 
                 vectors: Dict[str, Dict[int, torch.Tensor]], 
                 activations: Dict[str, Dict[str, Dict[int, Dict[str, torch.Tensor]]]]) -> int:
    """
    vectors: {value_name: {layer_idx: tensor}}
    activations: {value_name: {'pos': {layer_idx: {sample_id: tensor}}, 'neg': ...}}
    """
    print("Computing layer selection metrics...")
    
    # Load relations
    with open(config.relations_path, 'r') as f:
        relations_data = json.load(f)
        
    rel_matrix = relations_data['basic_value_relationship_matrix']
    
    # List of layers
    layers = list(vectors[SCHWARTZ_CIRCUMPLEX_ORDER[0]].keys())
    layers.sort()
    
    consistency_scores = {}
    discrimination_scores = {}
    
    for l_idx in layers:
        # 1. Cosine Consistency
        layer_const = []
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            acts_pos = activations[val]['pos'][l_idx]
            acts_neg = activations[val]['neg'][l_idx]
            c = compute_cosine_consistency(acts_pos, acts_neg)
            layer_const.append(c)
        consistency_scores[l_idx] = float(np.mean(layer_const))
        
        # 2. Cross-Value Discrimination
        d_scores = []
        # Get all vectors for this layer
        layer_vecs = {v: vectors[v][l_idx] for v in SCHWARTZ_CIRCUMPLEX_ORDER}
        
        checked_pairs = 0
        for v1, v2 in combinations(SCHWARTZ_CIRCUMPLEX_ORDER, 2):
            if v1 in rel_matrix and v2 in rel_matrix[v1]:
                if rel_matrix[v1][v2] == -1: # Opposing values
                    vec1 = layer_vecs[v1]
                    vec2 = layer_vecs[v2]
                    
                    if vec1.norm() > 0 and vec2.norm() > 0:
                        cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()
                        # Cosine distance = 1 - cos_sim. Higher handles separation better.
                        cos_dist = 1.0 - cos_sim
                        d_scores.append(cos_dist)
                        checked_pairs += 1
                        
        discrimination_scores[l_idx] = float(np.mean(d_scores)) if d_scores else 0.0

    # Normalize scores [0, 1]
    const_vals = np.array([consistency_scores[l] for l in layers])
    disc_vals = np.array([discrimination_scores[l] for l in layers])
    
    norm_const = (const_vals - const_vals.min()) / (const_vals.max() - const_vals.min() + 1e-9)
    norm_disc = (disc_vals - disc_vals.min()) / (disc_vals.max() - disc_vals.min() + 1e-9)
    
    combined_scores = norm_const + norm_disc
    
    # Argmax
    best_idx = int(np.argmax(combined_scores))
    selected_layer = layers[best_idx]
    
    print(f"Selected layer based on metrics: {selected_layer}")
    
    # Save results
    out_dir = config.subdir("layer_selection")
    
    scores_dict = {
        l: {
            "consistency": consistency_scores[l],
            "discrimination": discrimination_scores[l],
            "combined_normalized": float(combined_scores[i])
        }
        for i, l in enumerate(layers)
    }
    
    with open(os.path.join(out_dir, "layer_scores.json"), "w") as f:
        json.dump(scores_dict, f, indent=2)
        
    with open(os.path.join(out_dir, "selected_layer.json"), "w") as f:
        json.dump({"selected_layer": selected_layer}, f, indent=2)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(layers, const_vals, marker='o', label='Cosine Consistency')
    plt.xlabel('Layer')
    plt.ylabel('Score')
    plt.title('Cosine Consistency by Layer')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "cosine_consistency.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, disc_vals, marker='s', color='orange', label='Cross-Value Discrimination')
    plt.xlabel('Layer')
    plt.ylabel('Score')
    plt.title('Cross-Value Discrimination by Layer')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "cross_value_discrimination.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, norm_const, marker='o', label='Normalized Consistency')
    plt.plot(layers, norm_disc, marker='s', label='Normalized Discrimination')
    plt.plot(layers, combined_scores, marker='^', label='Combined Score', linewidth=2, color='green')
    plt.axvline(x=selected_layer, color='red', linestyle='--', label=f'Selected ({selected_layer})')
    plt.xlabel('Layer')
    plt.ylabel('Normalized Score')
    plt.title('Layer Selection Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "combined_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return selected_layer
