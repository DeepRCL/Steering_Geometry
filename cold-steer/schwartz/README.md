# COLD-Steer × Schwartz value-steering pipeline

End-to-end pipeline that trains COLD-Steer on our 20-value Schwartz benchmark
and evaluates it with the *same* geometry and log-likelihood metrics as
`llm-steering-opt/pipeline` and `odesteer/scripts/schwartz`.

Two variants are supported via `--method`:

| Method | Steerer | Representative vector |
|--------|---------|----------------------|
| `cold_fd` (default) | `LossFDSteerer` | Mean `(z(θ′,x) − z(θ,x))/ε` at last token |
| `cold_kernel` | `KernelLossSteerer` | Mean `η·v_steer` at last token (matches hook) |

## What this pipeline does

```
CSV (final_dataset_v3.csv)
        │ per-value split: n_train rows → train, rest → val
        ▼
SteerableLLM (HuggingFace, Llama-shaped models incl. Qwen)
        │ layer sweep via mean normalized L2 separation of pos/neg activations
        ▼
For each of 20 Schwartz values:
    SchwartzValueDataset  →  steerer.train()  (cold_fd or cold_kernel)
                              │
                              ▼
        extract representative vector (method-specific; see table above)
            at the chosen layer's last response token
        │
        ▼
20 vectors → geometry.analyze_geometry  →  heatmaps + UMAP/PCA/t-SNE/MDS
                                            + cosine-vs-Schwartz metrics
Validation rows → evaluate.evaluate_steerer  →  baseline vs steered
                                                 log P(pos) − log P(neg)
```

## Design choices (locked in via the planning step)

| Decision | Choice | Why |
|----------|--------|-----|
| Steering method | `--method cold_fd` or `cold_kernel` | Select via CLI |
| Kernel (cold_kernel only) | `--kernel constant` (default) | YAML `none` → `constant` |
| Default model | `Qwen/Qwen3.5-9B-Base` | Matches our llm-steering-opt runs |
| Easily swappable model | `--model_name <any Llama-shaped HF id>` | We patched `SteerableLLM` to accept Qwen/Llama-3 |
| Representative vector | Mean activation displacement `(z(θ+ε·g) − z(θ))/ε` | Closest scalar summary of what the hook actually injects |
| Layer selection | Shared normalized L2 separation of pos/neg activations | Apples-to-apples with `llm-steering-opt` |
| Training samples per value | Configurable CLI flag (default 30) + sweep over `{1, 10, 30, 50}` | cold-steer is few-shot |
| Loss inside cold_fd | `--training_mode sft` by default; `dpo` available | Paper uses SFT; we expose DPO since our data has explicit pos/neg |

## Layout

```
cold-steer/schwartz/
├── __init__.py
├── config.py             # SchwartzColdConfig + Schwartz circumplex constants
├── data_utils.py         # CSV load, train/val split by n_training_samples, prompts
├── schwartz_dataset.py   # per-value torch Dataset (matching/not_matching tensors)
├── layer_selection.py    # normalized L2 separation across candidates
├── method_adapters.py    # Preloaded steerers + extract_representative_vector
├── evaluate.py           # log-prob steering eval + bar chart
├── geometry.py           # circumplex metrics + heatmaps + UMAP/PCA/t-SNE/MDS
├── pipeline.py           # SchwartzColdPipeline orchestrator
├── run.py                # CLI entry point
├── sbatch_schwartz.slurm # SLURM submission (handles single run + n_training_samples sweep)
└── README.md
```

We also applied one small patch to `cold-steer/src/llm.py`: the
`SteerableLLM` family check now also accepts `Qwen*` and `Llama-3` model
names (both use Llama-shaped modules).

## Quick start

From the repo root:

```bash
cd cold-steer
python -m schwartz.run \
    --dataset_path ../llm-steering-opt/final_dataset_v3.csv \
    --relations_path ../llm-steering-opt/schwartz_relations.json
```

Useful overrides:

```bash
# Few-shot sweep manually
python -m schwartz.run --n_training_samples 10

# Skip the layer sweep and use a fixed layer
python -m schwartz.run --no_layer_sweep --layer 22

# Use a different model
python -m schwartz.run --model_name meta-llama/Llama-2-7b-hf

# DPO-style gradient (uses both positive and negative answers)
python -m schwartz.run --training_mode dpo

# cold_kernel with constant kernel (default)
python -m schwartz.run --method cold_kernel --eta 40 --layer 22 --no_layer_sweep
```

## SLURM

```bash
# Single run with defaults
sbatch cold-steer/schwartz/sbatch_schwartz.slurm

# Pass extra CLI args (forwarded straight to schwartz.run)
sbatch cold-steer/schwartz/sbatch_schwartz.slurm --training_mode dpo --eta 2.0

# Sweep over n_training_samples ∈ {1, 10, 30, 50} (any value of N_TRAIN_LIST works)
N_TRAIN_LIST="1 10 30 50" sbatch cold-steer/schwartz/sbatch_schwartz.slurm

# Sweep + extra args together
N_TRAIN_LIST="1 10 30 50" \
    sbatch cold-steer/schwartz/sbatch_schwartz.slurm --model_name meta-llama/Llama-2-7b-hf
```

## Output layout

```
{output_dir}/{model_short}/cold_fd-{training_mode}-eta_{η}-eps_{ε}-layer_{L}-n_train_{N}-eval_{metric}/
# or for cold_kernel:
{output_dir}/{model_short}/cold_kernel-{training_mode}-eta_{η}-kernel_{κ}-layer_{L}-n_train_{N}-eval_{metric}/
├── config.json
├── training_info.json
├── layer_sweep.json                 (only when sweep was run)
├── steering_eval_metrics.json
├── steering_eval_accuracy.png
├── vectors/
│   ├── manifest.json
│   ├── {value}.pt                   ×20
│   └── {value}.json                 ×20
└── geometry/
    ├── geometry_metrics.json
    ├── spearman_report.json
    ├── empirical_similarity_heatmap.png
    ├── empirical_similarity_heatmap_enhanced.png
    ├── empirical_similarity_heatmap_ranked.png
    ├── theoretical_similarity_heatmap.png
    ├── empirical_minus_theoretical_heatmap.png
    ├── theory_vs_empirical_scatter.png
    ├── umap_2d.png
    ├── pca_2d.png
    ├── tsne_2d.png
    └── mds_circumplex.png
```

`geometry_metrics.json` exposes the same keys as `llm-steering-opt`:
`spearman_rho`, `pearson_r`, `circular_distance_spearman`,
`hierarchical_distance_spearman`, `lower_minus_opposite_cosine`,
`silhouette_by_higher_order_group`, `procrustes_rmse_after_alignment`,
etc., so the comparison table across methods reduces to reading the same
fields from each method's `geometry_metrics.json`.

## Method-specific notes

**Why we extract a "representative vector" instead of using the steerer
directly.** Neither COLD variant stores a single fixed direction. For
`cold_fd`, inference applies `z ← z − (η/ε) · (z(θ′, x) − z(θ, x))` with
`θ′ = θ + ε·mean_grad`; we average that displacement over training prompts.
For `cold_kernel`, the hook injects `η·v_steer` (kernel-weighted); we
average that delta at the last token. Geometry needs one vector per value,
so both methods use the same readout position but method-specific formulas.

**Few-shot training.** cold-steer's paper uses 50 samples per behavior.
Schwartz values have ~230 rows each, so any value in `{1, 10, 30, 50}`
is well-supported; pick `1` for a strict one-shot setting. All remaining
rows per value are used for validation (subject to `--n_eval_samples`).

**Layer selection is shared, not method-specific.** We deliberately use
the same activation-based L2 separation scoring as `llm-steering-opt` so
when methods choose different layers the comparison stays interpretable.
Each method *can* override with `--no_layer_sweep --layer L` if needed.

**Model swap.** `SteerableLLM` is layout-sensitive. We support every
Llama-shaped family used in our research (Qwen2/3, Llama-2/3, Mistral-v0.1,
Gemma-2). Adding another family requires extending the branch in
`cold-steer/src/llm.py`.

## Comparing against llm-steering-opt / ODESteer

All three pipelines produce a `geometry_metrics.json` with identical
keys. The simplest way to build a method-comparison table is to glob
across runs and diff the same keys:

```python
import json, glob, os, pandas as pd
rows = []
for run_dir in glob.glob("schwartz_results/*/*/geometry/geometry_metrics.json"):
    with open(run_dir) as f: m = json.load(f)
    rows.append({"run": run_dir, **m})
print(pd.DataFrame(rows)[["run", "spearman_rho", "lower_minus_opposite_cosine",
                          "silhouette_by_higher_order_group"]])
```
