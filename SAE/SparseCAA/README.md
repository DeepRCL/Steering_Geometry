# SparseCAA — Value Persona Vectors in SAE Sparse Latent Space

This pipeline extracts per-value Schwartz persona vectors **in the SAE's 16384-d sparse feature space** rather than the dense residual stream, and steers Qwen 3.5 9B through that sparse space during evaluation.

---

## The Core Mechanic

```
activations (4096-d)
    ↓  hook on model.model.layers[16].mlp
    ↓  sae.encode  →  sparse codes (16384-d)
    ↓  CAA:  persona_vec = mean(z_pos) − mean(z_neg)
    ↓  steer: z_steered = z + α · persona_vec
    ↓  sae.decode  →  modified activations (4096-d)
    ↓  transformer residual connection adds result normally
```

The SAE (`sae_base_best.pt`) was trained on the **Layer 16 MLP output** of Qwen 3.5 9B — a 4096-d tensor — so every encode/decode call operates in the model's native activation space at that layer.

---

## Dataset

Combined from two sources:

| Source | Rows | Per value |
|---|---|---|
| `CAA/value_data/final_dataset_200.csv` (base) | 4,335 | 181–276 |
| `SAE/touche_gemma4-v2_remaining-validated-v3.csv` (supplement, `caa_suitable=True`, ≤50/value) | ~984 | up to 50 |
| **Combined** | **~5,319** | **231–326** |

All 20 Schwartz values are covered with no low-confidence values.

---

## Pipeline Modules

| Module | GPU needed | What it does |
|---|---|---|
| `finetune` | Yes (Qwen) | Collect all-token MLP activations from dataset; fine-tune SAE on them (MSE + L1); save `sae_finetuned.pt` |
| `extract` | Yes (Qwen) | For each value's training pairs: hook last-token MLP → SAE encode → sparse code; `persona_vec = mean(z_pos) − mean(z_neg)` |
| `evaluate` | Yes (Qwen) | Sparse steering hook during A/B inference; measure logit accuracy per alpha |
| `geometry` | No | Geometry metrics (`Spearman ρ`, `Pearson r`, circular/hierarchical alignment, lower-minus-opposite cosine), UMAP, t-SNE, MDS circumplex, heatmaps for **raw** and **mean-centred** vectors |

---

## Quick Start

Run from the **project root** (`Steering_Geometry/`):

```bash
# Full pipeline (all four modules)
python -m SAE.SparseCAA.run_pipeline \
  --base_dataset_path   CAA/value_data/final_dataset_200.csv \
  --touche_dataset_path SAE/touche_gemma4-v2_remaining-validated-v3.csv \
  --relations_path      schwartz_relations.json \
  --sae_checkpoint      SAE/sae_base_best.pt \
  --modules all
```

### Step-by-step (recommended for large models)

```bash
# Step 1 — fine-tune SAE (GPU)
python -m SAE.SparseCAA.run_pipeline \
  --base_dataset_path CAA/value_data/final_dataset_200.csv \
  --touche_dataset_path SAE/touche_gemma4-v2_remaining-validated-v3.csv \
  --relations_path schwartz_relations.json \
  --sae_checkpoint SAE/sae_base_best.pt \
  --modules finetune

# Step 2 — extract sparse vectors (GPU)
python -m SAE.SparseCAA.run_pipeline ... --modules extract

# Step 3 — evaluate steering (GPU)
python -m SAE.SparseCAA.run_pipeline ... --modules evaluate

# Step 4 — geometry analysis (CPU-only)
python -m SAE.SparseCAA.run_pipeline ... --modules geometry
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--base_dataset_path` | `CAA/value_data/final_dataset_200.csv` | Primary value dataset |
| `--touche_dataset_path` | `SAE/touche_gemma4-v2_remaining-validated-v3.csv` | Supplement (filtered) |
| `--touche_samples_per_value` | `50` | Max supplement rows per value |
| `--equal_samples_per_value` | `False` | Cap all values at minimum count (231) for strict balance |
| `--mlp_layer` | `16` | Transformer layer to hook — must match SAE training layer |
| `--sae_checkpoint` | `SAE/sae_base_best.pt` | Starting checkpoint for fine-tuning |
| `--finetune_epochs` | `3` | SAE fine-tuning epochs |
| `--finetune_lr` | `1e-5` | Learning rate (lower than pre-training's 5e-5) |
| `--finetune_batch_size` | `4096` | Activation vectors per SAE training step |
| `--l1_coefficient` | `0.005` | Sparsity penalty (matches pre-training) |
| `--alpha` | `0.5,1.0,2.0,4.0` | Steering magnitudes for evaluation |
| `--output_dir` | `SAE/SparseCAA/outputs` | Root output directory |
| `--modules` | `all` | Which modules to run |

---

## Output Structure

```
SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B/
├── pipeline_config.json
├── sae_finetuned.pt              ← fine-tuned SAE checkpoint
├── activation_cache.h5           ← cached MLP activations (auto-reused)
│
├── sparse_vectors/               ← (d_sae,) float32 per value
│   ├── Achievement.pt
│   ├── ...
│   └── value_metadata.json       ← sample counts, norms, feature stats
│
├── evaluation/
│   ├── eval_results.json         ← same format as CAA/Geometry eval
│   ├── baseline_vs_steered_accuracy.png
│   ├── accuracy_gain_vs_baseline.png
│   └── accuracy_gain_heatmap.png
│
├── geometry_raw/                 ← raw sparse persona vectors
│   ├── spearman_report.json
│   ├── empirical_similarity_heatmap.png
│   ├── theoretical_similarity_heatmap.png
│   ├── mds_circumplex.png
│   ├── umap_2d.png
│   └── tsne_2d.png
│
├── geometry_centered/            ← mean-centred vectors (visualisation only)
│   └── (same six files)
│
├── geometry_comparison.json      ← Δρ raw vs mean-centred
└── rho_comparison.png
```

---

## Interpreting Results

### Steering evaluation (`eval_results.json`)

- `baseline.accuracy`: fraction correct without steering (should be ~0.5 for a balanced MC test)
- `steered.<alpha>.accuracy`: fraction correct with sparse-SAE steering
- `accuracy_gain_vs_baseline`: positive = steering improved A/B selection toward the value

### Geometry

- **Spearman ρ** (`spearman_report.json`): correlation between empirical cosine similarity matrix and the theoretical Schwartz circumplex matrix. Higher (closer to +1) means the geometry reflects Schwartz theory better.
- **`geometry_metrics.json`**: richer quantitative report including `spearman_rho`, `pearson_r`, `circular_distance_spearman`, `hierarchical_distance_spearman`, and `lower_minus_opposite_cosine`.
- **`lower_minus_opposite_cosine`**: mean cosine for same lower-order families minus mean cosine for opposite higher-order pairs. Larger positive values mean closely related values are substantially nearer than theoretical opposites.
- **MDS circumplex**: empirical dots (coloured) vs. theoretical positions (grey ×). If the SAE space captures value structure, dots should spread around the circle in the correct group order.
- **Mean-centred vs raw**: subtracting the mean sparse vector removes the shared "baseline activation" direction. If mean-centred ρ > raw ρ, the common component was masking value-specific geometry.

---

## SAE Architecture Reference

| Property | Value |
|---|---|
| Base model | Qwen 3.5 9B |
| Activation source | Layer 16 MLP output |
| Input dimension | 4,096 |
| Feature dimension | 16,384 |
| Architecture | `Input → [−bias] → Linear → ReLU → [features] → Linear → [+bias] → Reconstruction` |
| Pre-training loss | MSE + L1 (λ=0.005), ~50M tokens |
