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
| `extract` | Compute per-value sparse CAA persona vectors (`persona_vec = mean(z_pos) − mean(z_neg)`) in the 65 536-d SAE feature space. | Yes |
| `evaluate` | Steer the model via the sparse SAE space and measure A/B logit accuracy across alpha values. | Yes |
| `geometry` | Spearman ρ vs. Schwartz theory, UMAP, t-SNE, MDS circumplex, similarity heatmaps. | No (CPU) |

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
| `--output_dir` | `SAE/QwenScopeCAA/outputs` | Root output directory |

---

## Outputs

All outputs are written under `SAE/QwenScopeCAA/outputs/<model>_layer<n>/`:

```
pipeline_config.json          # Full run configuration
activation_cache.h5           # HDF5 activation cache (finetune module)
sae_finetuned_layer16.pt      # Fine-tuned SAE checkpoint (Qwen-Scope format)
sparse_vectors/
  *.pt                        # One (65536,) persona vector per Schwartz value
  value_metadata.json         # Per-value stats (n_train, norm, feature counts)
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

## Steering Mechanism

```
residual  (batch, seq, 4096)
  ↓  sae.encode  [TopK, k=50]
z         (batch, seq, 65536)   — sparse feature activations
  ↓  z_steered = z + α · persona_vec
  ↓  sae.decode
residual_steered  (batch, seq, 4096)   ← replaces layer output
```

The hook is registered on `model.model.layers[config.layer]`. Unlike `SparseCAA` (which patches the MLP output), this approach patches the full residual stream, so all downstream layers see the steered representation.

---

## File Structure

```
SAE/QwenScopeCAA/
├── __init__.py
├── config.py                   # QwenScopePipelineConfig dataclass
├── topk_sae_model.py           # TopKSparseAutoencoder, load/save/download helpers
├── data_loader.py              # load_combined() + re-exports from SparseCAA
├── finetune_sae.py             # Residual-stream activation collection + MSE fine-tuning
├── extract_sparse_vectors.py   # Last-token residual hook + TopK encode → persona vecs
├── evaluate.py                 # Steering hook + A/B accuracy evaluation
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
