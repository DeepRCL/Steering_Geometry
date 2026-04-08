import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.manifold import MDS
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

from .config import PipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, HIGHER_ORDER_GROUPS, value_to_group, GROUP_COLORS, safe_name

def analyze_geometry(config: PipelineConfig, vectors: Dict[str, torch.Tensor]):
    print("Running geometry analysis...")
    out_dir = config.subdir("geometry")
    
    # Ensure vectors are normalized
    unit_vectors = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        vec = vectors[val].detach().cpu().float()
        norm = vec.norm()
        if norm > 0:
            unit_vectors[val] = vec / norm
        else:
            unit_vectors[val] = vec
            
    num_values = len(SCHWARTZ_CIRCUMPLEX_ORDER)
    
    # 1. Empirical Similarity Matrix
    empirical_sim = np.zeros((num_values, num_values))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            cos_sim = F.cosine_similarity(unit_vectors[v1], unit_vectors[v2], dim=0).item()
            empirical_sim[i, j] = cos_sim
            
    # 2. Theoretical Matrix
    with open(config.relations_path, 'r') as f:
        rel_data = json.load(f)
    rel_matrix = rel_data['basic_value_relationship_matrix']
    
    theoretical_sim = np.zeros((num_values, num_values))
    for i, v1 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        for j, v2 in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
            if v1 in rel_matrix and v2 in rel_matrix[v1]:
                theoretical_sim[i, j] = rel_matrix[v1][v2]
                
    # 3. Correlation
    # Get upper triangles without diagonal
    triu_indices = np.triu_indices(num_values, k=1)
    emp_flat = empirical_sim[triu_indices]
    theo_flat = theoretical_sim[triu_indices]
    
    rho, p_val = spearmanr(emp_flat, theo_flat)
    
    with open(os.path.join(out_dir, "spearman_report.json"), "w") as f:
        json.dump({
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "num_pairs": len(emp_flat)
        }, f, indent=2)
        
    print(f"Spearman correlation between theoretical and empirical similarities: rho={rho:.4f}, p={p_val:.4g}")
    
    # 4. Visualizations
    
    # Heatmaps
    plt.figure(figsize=(12, 10))
    sns.heatmap(empirical_sim, xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Empirical Cosine Similarities')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "empirical_similarity_heatmap.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(theoretical_sim, xticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, yticklabels=SCHWARTZ_CIRCUMPLEX_ORDER, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Theoretical Relationships')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "theoretical_similarity_heatmap.png"), dpi=300)
    plt.close()
    
    # UMAP 2D
    X = np.stack([unit_vectors[v].numpy() for v in SCHWARTZ_CIRCUMPLEX_ORDER])
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=config.seed)
    X_umap = reducer.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        group = value_to_group(val)
        color = GROUP_COLORS.get(group, "black")
        plt.scatter(X_umap[i, 0], X_umap[i, 1], c=color, s=100)
        plt.annotate(val.split(':')[-1].strip(), (X_umap[i, 0], X_umap[i, 1]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    # Legend
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=g) 
                  for g, c in GROUP_COLORS.items()]
    plt.legend(handles=legend_els, loc='best')
    plt.title('UMAP 2D Projection of Steering Vectors')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_2d.png"), dpi=300)
    plt.close()
    
    # MDS with Circumplex Overlay
    # Distance matrix = 1 - cosine similarity
    dist_matrix = 1 - empirical_sim
    # Replace negative distances with 0 just in case
    dist_matrix[dist_matrix < 0] = 0
    
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=config.seed, normalized_stress='auto')
    X_mds = mds.fit_transform(dist_matrix)
    
    # Theoretical points on a circle based on order
    angles = np.linspace(0, 2*np.pi, num_values, endpoint=False)
    # We want to optimally align X_mds (empirical) to the circle (theoretical) using Procrustes
    # but for a simple plot, we just plot both
    
    # We can calculate optimal rotation
    X_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    
    from scipy.linalg import orthogonal_procrustes
    R, sca = orthogonal_procrustes(X_mds, X_circle)
    X_mds_aligned = X_mds.dot(R)
    
    plt.figure(figsize=(12, 12))
    # Draw theoretical circle
    circle = plt.Circle((0, 0), 1, color='lightgray', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    
    for i, val in enumerate(SCHWARTZ_CIRCUMPLEX_ORDER):
        # Theoretical pos
        tx, ty = X_circle[i]
        plt.plot(tx, ty, 'x', color='gray', markersize=8)
        
        # Empirical pos
        ex, ey = X_mds_aligned[i]
        group = value_to_group(val)
        color = GROUP_COLORS.get(group, "black")
        
        plt.plot(ex, ey, 'o', color=color, markersize=8)
        
        # Draw line connecting theoretical to empirical
        plt.plot([tx, ex], [ty, ey], color='gray', alpha=0.3, linestyle=':')
        
        label = val.split(':')[-1].strip()
        plt.annotate(label, (ex, ey), xytext=(5, 5), textcoords='offset points', fontsize=9, color=color)
        
    plt.title('2D MDS Aligned to Theoretical Circumplex')
    plt.axis('equal')
    # Set limits clearly showing unit circle
    scale = np.max(np.abs(X_mds_aligned))
    lim = max(1.2, scale * 1.2)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mds_circumplex.png"), dpi=300)
    plt.close()
    
    print("Geometry analysis complete!")
