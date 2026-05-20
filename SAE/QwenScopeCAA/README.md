# QwenScopeCAA

SparseCAA pipeline using the [Qwen-Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50) pre-trained TopK Sparse Autoencoder on **Qwen3.5-9B-Base**.

Instead of hooking the MLP output (as in `SAE/SparseCAA/`), this pipeline hooks the **residual stream** after a chosen transformer layer and operates in the Qwen-Scope feature space (`d_sae = 65 536`, TopK k=50).

---

## Key Differences from `SAE/SparseCAA/`

| Property | SparseCAA | QwenScopeCAA |
|---|---|---|
| SAE source | Local `sae_base_best.pt` (custom-trained) | Downloaded from HuggingFace Hub per layer |
| SAE architecture | ReLU | TopK (k=50, always exactly 50 active features) |
| `d_sae` | 16 384 (4×) | 65 536 (16×) |
| Hook point | `model.model.layers[n].mlp` | `model.model.layers[n]` (full residual stream) |
| Checkpoint format | `{model_state_dict}` | `{W_enc, W_dec, b_enc, b_dec}` |
| Fine-tuning loss | MSE + L1 | MSE only (TopK enforces sparsity) |
| Touche samples/value | 50, from `-v3.csv` | 200, from `-final.csv` (Hedonism: 90) |
| Persona vector mean | Standard mean | τ-masked non-zero mean (default τ=0.7) |
| Common feature removal | No | Yes — shared features zeroed before subtraction |
| Steering correction | None | Δ reconstruction correction (default on) |

---

## Dataset

| Source | Usage |
|---|---|
| `CAA/value_data/final_dataset_200.csv` | All rows, all 20 Schwartz values |
| `SAE/touche_gemma4-v2_remaining-validated-final.csv` | Filtered to `caa_suitable=True`, up to **200 rows per value** added on top of the base dataset |

**Hedonism note**: the `-final` CSV contains only 90 suitable records for Hedonism. All 90 are used without padding; the minor count imbalance is accepted and handled naturally by the per-value mean in the CAA extraction step.

---

## Pipeline Modules

```
finetune  →  extract  →  evaluate  →  geometry
```

| Module | Description | GPU required? |
|---|---|---|
| `finetune` | Adapt the pre-trained Qwen-Scope SAE to value-specific residual activations. Collects all-token activations into an HDF5 cache, then fine-tunes with MSE loss. | Yes |
| `extract` | Compute per-value sparse CAA persona vectors using τ-masked non-zero mean and optional common-feature removal. | Yes |
| `evaluate` | Steer the model via the sparse SAE space (with optional Δ reconstruction correction) and measure A/B logit accuracy across alpha values. | Yes |
| `geometry` | Spearman ρ vs. Schwartz theory, UMAP, t-SNE, MDS circumplex, similarity heatmaps. | No (CPU) |

---

## Persona Vector Extraction Enhancements

### τ Frequency Threshold (`--tau`, default `0.7`)

Instead of a plain mean, persona vectors are computed with a **frequency-masked non-zero mean**:

- Stack all `pre_encode` activations for the positive side into matrix `S_pos` of shape `(N, 65536)`.
- For each feature `c`: `freq[c] = count(S_pos[:, c] != 0) / N`
- Feature `c` is included only if `freq[c] >= τ`; otherwise it is zeroed.
- For included features: `mean_vec[c] = sum(S_pos[:, c]) / count(S_pos[:, c] != 0)` (divide by non-zero rows, not N).

This eliminates features that fired for only 1–2 prompts, reducing noise in the persona vector. Set `--tau 0.0` to recover the original behaviour.

### Common Feature Removal (`--no_remove_common_features` to disable)

After computing `v_pos` and `v_neg` (both `(65536,)` after τ filtering), features that are non-zero in **both** sides are zeroed before the subtraction:

```
common_mask = (v_pos != 0) & (v_neg != 0)
v_pos[common_mask] = 0
v_neg[common_mask] = 0
persona_vec = v_pos - v_neg
```

Shared features are likely syntactic or positional artifacts that fire regardless of value polarity. Removing them sharpens the value-discriminative signal.

> **Note:** This step is automatically skipped in the default **pre-TopK (dense)** mode (`use_pre_topk_personas=True`). In that mode every dimension of `pre_encode` is non-zero, so the common mask would be all-True and wipe both vectors entirely, producing zero persona vectors. The dense subtraction already handles this correctly — if a feature fires equally on both sides, `mean_pos[c] − mean_neg[c] = 0` naturally. Common-feature removal is only meaningful for **post-TopK (sparse)** mode where the zero/non-zero boundary is semantically significant.

---

## Steering Mechanism

### Pre-TopK hook (default, `use_pre_topk_personas=True`)

```
residual  (batch, seq, 4096)
  ↓  sae.pre_encode  [dense, linear]
pre       (batch, seq, 65536)
  ↓  pre_steered = pre + α · persona_vec
  ↓  TopK(pre_steered, k=50)        ← α biases which 50 features are selected
z_steered (batch, seq, 65536)
  ↓  sae.decode
residual_steered  (batch, seq, 4096)   ← replaces layer output
```

The hook is registered on `model.model.layers[config.layer]`. Unlike `SparseCAA` (which patches the MLP output), this approach patches the full residual stream, so all downstream layers see the steered representation.

### Δ Reconstruction Correction (`--no_delta_correction` to disable)

By default, the steering hook also corrects for the SAE's inherent reconstruction error:

```
pre       (batch, seq, 65536)            ← from sae.pre_encode
z_u     = TopK(pre, k=50)               ← unsteered sparse encoding
act_recon = sae.decode(z_u)             ← unsteered reconstruction
delta   = residual - act_recon          ← reconstruction error

[steered path as above → recon]

residual_steered = recon + delta        ← error added back
```

Without correction, the SAE reconstruction error is injected into the residual stream on every steered forward pass, which can cause erratic behaviour especially in earlier layers. Adding `delta` back preserves the information the SAE cannot reconstruct and keeps the steered output faithful to the original residual.

---

## Usage

Run from the **project root** (`Steering_Geometry/`):

```bash
# Full pipeline — fine-tune SAE, then extract / evaluate / geometry:
python -m SAE.QwenScopeCAA.run_pipeline --layer 16 --modules all

# Skip fine-tuning — use the pre-trained Qwen-Scope SAE directly:
python -m SAE.QwenScopeCAA.run_pipeline --layer 16 --skip_finetune

# Different layer (any of 0–31), no fine-tuning:
python -m SAE.QwenScopeCAA.run_pipeline --layer 24 --skip_finetune

# GPU steps now, geometry later on CPU:
python -m SAE.QwenScopeCAA.run_pipeline --layer 16 --modules finetune,extract,evaluate
python -m SAE.QwenScopeCAA.run_pipeline --layer 16 --modules geometry

# Reproduce original behaviour (no tau filtering, no common removal, no delta correction):
python -m SAE.QwenScopeCAA.run_pipeline --layer 16 --skip_finetune \
    --tau 0.0 --no_remove_common_features --no_delta_correction
```

### Key CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--layer` | `16` | Transformer layer to hook (0–31) |
| `--skip_finetune` | off | Use the pre-trained SAE directly (skips `finetune` module) |
| `--modules` | `all` | Comma-separated subset: `finetune,extract,evaluate,geometry` |
| `--sae_repo` | `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` | HuggingFace repo for the Qwen-Scope SAE |
| `--model_name` | `Qwen/Qwen3.5-9B-Base` | HuggingFace model ID |
| `--touche_samples_per_value` | `200` | Max Touche rows added per value |
| `--alpha` | `0.5,1.0,2.0,4.0` | Steering strengths for evaluation |
| `--tau` | `0.7` | Frequency threshold τ for feature inclusion in persona mean |
| `--no_remove_common_features` | off | Disable zeroing of features active in both v_pos and v_neg |
| `--no_delta_correction` | off | Disable SAE reconstruction-error correction in the steering hook |
| `--output_dir` | `SAE/QwenScopeCAA/outputs` | Root output directory |

---

## Outputs

All outputs are written under `SAE/QwenScopeCAA/outputs/<model>_layer<n>_k<k>/`:

```
pipeline_config.json          # Full run configuration (includes tau, remove_common_features,
                              #   use_delta_correction)
activation_cache.h5           # HDF5 activation cache (finetune module)
sae_finetuned_layer16.pt      # Fine-tuned SAE checkpoint (Qwen-Scope format)
sparse_vectors/
  *.pt                        # One (65536,) persona vector per Schwartz value
  value_metadata.json         # Per-value stats: n_train, norm, feature counts,
                              #   tau, n_pos/neg_features_above_tau,
                              #   n_common_features_removed
evaluation/
  eval_results.json
  baseline_vs_steered_accuracy.png
  accuracy_gain_vs_baseline.png
  accuracy_gain_heatmap.png
geometry_raw/
  spearman_report.json
  geometry_metrics.json
  empirical_similarity_heatmap.png
  theoretical_similarity_heatmap.png
  mds_circumplex.png
  umap_2d.png
  tsne_2d.png
geometry_centered/            # Same plots, mean-centred vectors
  ...
geometry_comparison.json      # Δρ: raw vs. mean-centred
rho_comparison.png
```

---

## File Structure

```
SAE/QwenScopeCAA/
├── __init__.py
├── config.py                   # QwenScopePipelineConfig dataclass
│                               #   (tau, remove_common_features, use_delta_correction)
├── topk_sae_model.py           # TopKSparseAutoencoder, load/save/download helpers
├── data_loader.py              # load_combined() + re-exports from SparseCAA
├── finetune_sae.py             # Residual-stream activation collection + MSE fine-tuning
├── extract_sparse_vectors.py   # τ-masked mean + common removal → persona vecs
├── evaluate.py                 # Pre-TopK steering hook with Δ correction + A/B evaluation
├── geometry.py                 # Thin wrapper over SAE.SparseCAA.geometry
└── run_pipeline.py             # CLI entry point
```

---

## Citation

If you use the Qwen-Scope SAE in your work:

```bibtex
@misc{qwen_scope,
  title={{Qwen-Scope}: Turning Sparse Features into Development Tools for Large Language Models},
  author={Boyi Deng and Xu Wang and Yaoning Wang and Yu Wan and Yubo Ma and Baosong Yang
          and Haoran Wei and Jialong Tang and Huan Lin and Ruize Gao and Tianhao Li
          and Qian Cao and Xuancheng Ren and Xiaodong Deng and An Yang and Fei Huang
          and Dayiheng Liu and Jingren Zhou},
  year={2026},
  eprint={2605.11887},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2605.11887},
}
```
