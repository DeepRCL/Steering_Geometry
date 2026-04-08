# SAE Analysis Pipeline for Schwartz Value Steering Vectors

This pipeline applies a **Sparse Autoencoder (SAE)** to the CAA steering vectors
extracted in `CAA/Geometry/`, with the goal of decomposing them into interpretable
features and testing whether the resulting geometry aligns better with Schwartz's
theory of basic human values.

---

## Background and Motivation

### The Problem

After running the `CAA/Geometry` pipeline on Qwen 3.5 9B, we got a weak
Spearman correlation (ρ ≈ 0.20) between the empirical cosine similarity of
steering vectors and the theoretical Schwartz circumplex.  The MDS and UMAP
plots show vectors clustered near the origin with poor separation — a classic
sign of **polysemanticity collapse**: each difference vector captures a
mixture of value-specific signal *and* generic language patterns (argumentation
style, topic words, etc.) that appear across all values.

### The Solution

A Sparse Autoencoder (SAE) trained on the same model's activations can
decompose each dense 4096-d vector into a sparse set of monosemantic features
(16 384 features, most inactive for any given input).  This lets us:

1. **Identify common features** — features highly active for *every* Schwartz
   value.  These encode generic content unrelated to any specific value.

2. **Purify the vectors** — zero out the common features in the sparse code and
   reconstruct.  The result is a vector that retains only value-specific signal.

3. **Re-run geometry analysis** — compare Spearman ρ before and after
   purification to see if the alignment with Schwartz theory improves.

4. **Disjointness test** — check whether opposing higher-order groups
   (Conservation ↔ Openness to Change; Self-Enhancement ↔ Self-Transcendence)
   activate *disjoint* feature sets, as Schwartz theory predicts.

### Why a separate extraction step?

The SAE was trained specifically on the **MLP output** of layer 16
(`model.model.layers[16].mlp`), not on the full residual stream.  The
`CAA/Geometry` pipeline saves residual-stream vectors (whole transformer block
output) which are correlated but not identical to the MLP output.

To feed the SAE vectors it was actually trained on, this pipeline re-extracts
CAA difference vectors by hooking directly into `layers[16].mlp`.

---

## File Structure

```
SAE/
├── __init__.py               # package marker
├── config.py                 # SAEConfig dataclass + Schwartz constants
├── sae_model.py              # SparseAutoencoder architecture + loader
├── extract_mlp_vectors.py    # GPU: extract CAA vectors from MLP layer
├── sae_analysis.py           # CPU: SAE projection, purification, geometry,
│                             #      disjointness test
├── run_sae_pipeline.py       # CLI entry point
├── sae_base_best.pt          # pre-trained SAE checkpoint (Qwen 3.5 9B)
└── README.md                 # this file
```

---

## Setup

Install dependencies (if not already present):

```bash
pip install torch transformers umap-learn scikit-learn matplotlib seaborn scipy h5py tqdm
```

The SAE checkpoint `sae_base_best.pt` is already in the `SAE/` directory.

---

## Usage

Run from the **project root** (`Steering_Geometry/`).

### Step 1 – Extract MLP vectors  *(requires GPU + Qwen 3.5 9B)*

```bash
python -m SAE.run_sae_pipeline \
  --model_name   Qwen/Qwen3.5-9B \
  --dataset_path CAA/value_data/final_dataset_200.csv \
  --relations_path schwartz_relations.json \
  --sae_checkpoint SAE/sae_base_best.pt \
  --modules extract
```

This creates:
```
SAE/outputs/Qwen__Qwen3.5-9B/mlp_vectors/<value>/layer_16.pt
```
one file per Schwartz value.  The step is **resumable** — already-computed
values are loaded from cache, so it is safe to interrupt and re-run.

### Step 2 – SAE analysis  *(CPU-only)*

```bash
python -m SAE.run_sae_pipeline \
  --model_name   Qwen/Qwen3.5-9B \
  --dataset_path CAA/value_data/final_dataset_200.csv \
  --relations_path schwartz_relations.json \
  --sae_checkpoint SAE/sae_base_best.pt \
  --modules analyze
```

### Run both steps at once

```bash
python -m SAE.run_sae_pipeline \
  --model_name   Qwen/Qwen3.5-9B \
  --dataset_path CAA/value_data/final_dataset_200.csv \
  --relations_path schwartz_relations.json \
  --sae_checkpoint SAE/sae_base_best.pt \
  --modules all
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--mlp_layer` | `16` | Transformer layer to hook (must match SAE training layer) |
| `--common_feature_top_k` | `128` | Features to zero out during purification.  Features are ranked by their **minimum** activation across all 20 value vectors — those with a high minimum are active for every value, suggesting generic (non-value-specific) content.  Increase if purification seems too weak; decrease if useful signal is removed. |
| `--top_features_per_value` | `64` | Top-K features per value used in the disjointness / Jaccard test |
| `--output_dir` | `SAE/outputs` | Root directory for all outputs |
| `--device` | `auto` | `auto` \| `cuda` \| `cpu` \| `mps` |
| `--seed` | `42` | Random seed (affects UMAP / t-SNE initialisation) |

---

## Output Files

After a complete run:

```
SAE/outputs/Qwen__Qwen3.5-9B/
├── sae_config.json                   ← run config
├── geometry_comparison.json          ← Spearman ρ before vs. after
├── rho_comparison.png                ← bar chart comparing ρ values
├── analysis_summary.json            ← high-level results
│
├── mlp_vectors/                      ← cached MLP CAA vectors
│   └── <value>/layer_16.pt
│
├── purified_vectors/                 ← SAE-purified CAA vectors
│   └── <value>.pt
│
├── features/
│   ├── feature_matrix.pt             ← (20, 16384) SAE activations
│   ├── common_features.json          ← IDs and activations of universal features
│   ├── value_feature_stats.json      ← per-value n_active, top features
│   └── common_feature_profile.png    ← heatmap: values × top-20 common features
│
├── geometry_raw_mlp/                 ← geometry on raw MLP vectors
│   ├── spearman_report.json
│   ├── empirical_heatmap.png
│   ├── theoretical_heatmap.png
│   ├── umap_2d.png
│   ├── pca_2d.png
│   └── mds_circumplex.png
│
├── geometry_purified/                ← same plots after purification
│   └── …
│
└── disjointness/
    ├── disjointness_results.json     ← pairwise Jaccard + raw counts
    ├── jaccard_heatmap.png           ← 4×4 group Jaccard heatmap
    └── opposing_overlap.png          ← exclusive vs. shared feature bars
```

---

## What to Look For

### 1. Does purification improve geometry alignment?

Check `geometry_comparison.json`:
```json
{
  "raw_spearman_rho":      0.196,
  "purified_spearman_rho": ???,
  "delta_rho":             ???
}
```
A positive Δρ means purification brought the empirical vectors closer to
Schwartz theory.  Compare the `mds_circumplex.png` plots — purified vectors
should fan out around the circle rather than clump in the centre.

### 2. Are common features generic?

Open `features/common_feature_profile.png`.  A uniform yellow band across all
20 rows for the top-K features confirms they encode value-neutral content.

### 3. Do opposing groups use disjoint features?

Open `disjointness/jaccard_heatmap.png`.  The Schwartz prediction is:
- **Diagonal**: Jaccard = 1.0 (trivially)
- **Adjacent groups** (e.g. Openness ↔ Self-Enhancement): moderate overlap
- **Opposing groups** (Conservation ↔ Openness-to-Change; Self-Enhancement ↔
  Self-Transcendence): **low Jaccard** (disjoint feature sets)

The `opposing_overlap.png` chart breaks down each opposing pair into exclusive
vs. shared features to make this concrete.

---

## Interpreting Results

| Result | Likely meaning |
|---|---|
| Δρ > 0 | Purification removed noise; value-specific geometry was obscured |
| Δρ ≈ 0 | Common features were not the bottleneck; try larger `--common_feature_top_k` or a different layer |
| Δρ < 0 | Common features carried meaningful geometry; the polysemanticity is inherent, not incidental |
| Low Jaccard for opposing pairs | Schwartz theory is reflected at the feature level ✓ |
| High Jaccard for opposing pairs | The model does not separate these value dimensions clearly |

---

## SAE Details

| Property | Value |
|---|---|
| Base model | Qwen 3.5 9B |
| Activation source | Layer 16 MLP output |
| Input dimension | 4 096 |
| Feature dimension | 16 384 (4× expansion) |
| Active features | 16 384 / 16 384 (0 dead features) |
| Training data | ~50 M tokens (pile-uncopyrighted) |

Architecture:
```
Input (4096) → [subtract bias] → Encoder (4096→16384) → ReLU → features
features → Decoder (16384→4096) → [add bias] → Reconstruction
```
