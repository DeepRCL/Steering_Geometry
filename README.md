# Steering Geometry — Dataset Construction

## Current Steering Baseline Runs

These commands use `Qwen/Qwen3.5-9B-Base`, keep the same base dataset split for
CAA, SphericalSteer, and QwenScopeCAA steering/evaluation, and write to new
output folders so existing results are not overwritten.

Evaluation now reports both metrics:

- `ab_next_token`: multiple-choice prompt, compare next-token `P(A)` vs `P(B)`.
- `full_answer_mean_logprob`: OPT-style scoring, compare mean log-probability
  of the full positive answer vs the full negative answer.

Geometry is unchanged for linear CAA. For SphericalSteer and QwenScopeCAA,
geometry uses the empirical intervention vector:

```text
value_vector = mean(steered_activation - original_activation)
```

### CAA

Existing best result:

`CAA/Geometry/outputs/qwen3_5_9b_base_v2_centered_renorm/Qwen__Qwen3.5-9B-Base`

Rerun into a new folder with both accuracy metrics:

```bash
python -m CAA.Geometry.run_pipeline \
  --model_name Qwen/Qwen3.5-9B-Base \
  --dataset_path CAA/value_data/final_dataset_200.csv \
  --relations_path CAA/value_data/schwartz_relations-new.json \
  --output_dir CAA/Geometry/outputs/qwen3_5_9b_base_v2_dual_metrics \
  --steering_method caa \
  --alpha 0.25,0.5,1.0,2.0,4.0 \
  --geometry_transform centered_renorm \
  --modules all
```

Outputs:

- A/B metric: `.../evaluation/evaluation_summary.json`
- Full-answer logprob metric: `.../evaluation/evaluation_summary_full_logprob.json`
- Geometry: `.../geometry/`

### SphericalSteer

Existing best evaluation:

`SphericalSteer/focused_tuning/k2_bneg0p6/Qwen__Qwen3.5-9B-Base/evaluation/evaluation_summary.json`

Rerun into a new folder with the current best hyperparameters:

```bash
python -m CAA.Geometry.run_pipeline \
  --model_name Qwen/Qwen3.5-9B-Base \
  --dataset_path CAA/value_data/final_dataset_200.csv \
  --relations_path schwartz_relations.json \
  --output_dir SphericalSteer/focused_tuning/k2_bneg0p6_dual_metrics \
  --steering_method spherical \
  --spherical_kappa 2.0 \
  --spherical_beta -0.6 \
  --spherical_steer_position last \
  --spherical_geometry_alpha 0.9 \
  --spherical_geometry_source neg \
  --spherical_geometry_vector displacement \
  --geometry_transform centered_renorm \
  --layer_override 16 \
  --alpha 0.9 \
  --modules all
```

Outputs:

- A/B metric: `.../evaluation/evaluation_summary.json`
- Full-answer logprob metric: `.../evaluation/evaluation_summary_full_logprob.json`
- Displacement geometry: `.../geometry_vectors/` and `.../geometry/`

### QwenScopeCAA

QwenScopeCAA still uses the larger base+Touche dataset for SAE fine-tuning.
After fine-tuning, extraction, steering evaluation, and geometry use the saved
CAA-compatible base-only split:

`.../splits/caa_base_split.json`

Rerun aligned extraction/evaluation/geometry into a new root output folder:

```bash
python -m SAE.QwenScopeCAA.run_pipeline \
  --layer 16 \
  --k 50 \
  --modules extract,evaluate,geometry \
  --alpha 0.5,1.0,2.0,4.0 \
  --geometry_vector displacement \
  --geometry_source neg \
  --output_dir SAE/QwenScopeCAA/outputs_dual_metrics
```

If the fine-tuned SAE does not exist in that new output folder, run the full
pipeline once:

```bash
python -m SAE.QwenScopeCAA.run_pipeline \
  --layer 16 \
  --k 50 \
  --modules all \
  --alpha 0.5,1.0,2.0,4.0 \
  --geometry_vector displacement \
  --geometry_source neg \
  --output_dir SAE/QwenScopeCAA/outputs_dual_metrics
```

Outputs:

- A/B metric: `.../evaluation_caa_base/evaluation_summary.json`
- Full-answer logprob metric: `.../evaluation_caa_base/evaluation_summary_full_logprob.json`
- Saved split: `.../splits/caa_base_split.json`
- Displacement geometry: `.../geometry_vectors/`, `.../geometry_raw/`, and `.../geometry_centered/`

---

Builds augmented datasets for value-alignment research. Currently supports three pipelines:

1. **ValueBench** — generates positive/negative answers for the ValueBench dataset using a local Qwen model, then maps raw value labels to canonical Schwartz categories via Gemini.
2. **Touche23-ValueEval** — separate pipeline for the Touche23 dataset.
3. **Value Stability** — generates perturbed question variants (semantic paraphrase + adversarial) from the merged final dataset using a local Qwen model.

---

## Project structure

```
.
├── config.py                                  # loads .env and exposes typed config
├── .env                                       # your secrets/settings (git-ignored)
├── .env.example                               # template to copy from
├── environment.yml                            # conda environment
├── run_pipeline.py                            # entry point for value stability (CLI)
├── utils/
│   ├── __init__.py
│   └── utils.py                               # shared utilities (parse_json, load_pending_rows)
├── value_stability/
│   ├── __init__.py
│   ├── config.py                              # perturbation model/data config
│   ├── pipeline.py                            # PerturbationPipeline class
│   ├── prompts.py                             # paraphrase + adversarial prompt templates
│   └── parser.py                             # JSON output parser for perturbation results
└── dataset_construction/
    ├── data/
    │   └── perturb_result/                    # perturbation output CSVs
    ├── value_bench/
    │   ├── pipeline.py                        # model loading, generation, dataset building
    │   ├── prompt.py                          # system/user prompt templates
    │   ├── run_pipelines.py                   # entry point for answer generation (CLI)
    │   ├── preprocessing.ipynb
    │   └── mapping/
    │       ├── mapper_prompts.py              # Gemini prompt templates + VALUE_DEFINITIONS
    │       └── value_mapper.py                # maps raw value labels → canonical Schwartz categories
    └── Touche23-ValueEval/
        ├── pipeline.py
        ├── prompt.py
        ├── run_pipelines.py
        ├── preprocessing.ipynb
        ├── sample_dataset.py
        └── validate.py
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate persona
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

### Answer generation / value mapping

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face token ([get one here](https://huggingface.co/settings/tokens)) | — |
| `GEMINI_API_KEYS` | Comma-separated Gemini API keys for value mapping (rotates on rate limit) | — |
| `GEMINI_MODEL` | Gemini model name | `gemini-3.1-flash-lite-preview` |
| `MODEL_ID` | HuggingFace model ID for answer generation | `Qwen/Qwen3.5-2B` |
| `MAX_NEW_TOKENS` | Max tokens generated per row | `512` |
| `DEVICE_MAP` | `auto`, `cpu`, or `cuda` | `auto` |
| `INPUT_CSV` | Input filename inside `dataset_construction/data/` | `dataset_positive_only.csv` |
| `BATCH_SIZE` | Rows processed before saving a checkpoint | `10` |
| `DEBUG_ROWS` | Rows used in a debug run | `10` |

### Value stability (perturbation)

| Variable | Description | Default |
|---|---|---|
| `PERTURBATION_MODEL_ID` | HuggingFace model ID for perturbation | `Qwen/Qwen3.5-2B` |
| `PERTURBATION_MAX_NEW_TOKENS` | Max tokens per perturbation call | `2048` |
| `PERTURBATION_INPUT_CSV` | Input filename inside `dataset_construction/data/` | `final_dataset_v3.csv` |
| `PARAPHRASE_OUTPUT_CSV` | Output filename for paraphrases | `final_dataset_paraphrased.csv` |
| `ADVERSARIAL_OUTPUT_CSV` | Output filename for adversarials | `final_dataset_adversarial.csv` |
| `PERTURBATION_DEBUG_ROWS` | Rows used in a debug run | `10` |

---

## 1 — Answer generation (ValueBench)

Generate positive or negative answers for each row using a local Qwen model.

```bash
python dataset_construction/value_bench/run_pipelines.py [--direction DIRECTION] [--method METHOD]
```

### `--direction`

| Value | Reads from | Generates | Output file |
|---|---|---|---|
| `positive_to_negative` (default) | `positive_answer` | `negative_answer` | `data/dataset_negative_answer.csv` |
| `negative_to_positive` | `negative_answer` | `positive_answer` | `data/dataset_positive_answer.csv` |

### `--method`

| Value | How it calls the model | Best for |
|---|---|---|
| `single` (default) | one model call per row | Mac / CPU / debugging |
| `batch` | one model call per batch | GPU server |

### Examples

```bash
# generate negatives, one row at a time (default)
python dataset_construction/value_bench/run_pipelines.py

# generate negatives, batch mode
python dataset_construction/value_bench/run_pipelines.py --method batch

# generate positives, single mode
python dataset_construction/value_bench/run_pipelines.py --direction negative_to_positive
```

### Debug run

Run `pipeline.py` directly to test on a small sample (`DEBUG_ROWS` rows):

```bash
python dataset_construction/value_bench/pipeline.py
```

The debug sample is created once at `data/debug_input.csv` and reused on subsequent runs.

---

## 2 — Value mapping (ValueBench)

Maps the raw `value` column in the generated dataset to canonical Schwartz categories using Gemini.

```bash
# Full run
python dataset_construction/value_bench/mapping/value_mapper.py

# Debug run (50-row sample, resumable)
python dataset_construction/value_bench/mapping/value_mapper.py --debug
```

**How it works:**

- Rows whose `value` already matches a canonical category are written directly (no API call).
- Remaining rows are grouped by unique value. One Gemini call is made per unique value, with the question and answers as context.
- Results are saved incrementally — if interrupted, re-running the same command resumes from the last saved value.
- If a Gemini API key hits a rate limit, the script automatically rotates to the next key in `GEMINI_API_KEYS`.
- Values that cannot be mapped are saved as `NA`.

Output column: `mapped_value`

---

## 3 — Value stability (perturbation)

Generates perturbed question variants from the merged final dataset. Two perturbation types are supported:

- **Paraphrase** — rewrites the question with different vocabulary, keeping the exact meaning. Output columns: `paraphrased_question`, `paraphrased_positive_answer`, `paraphrased_negative_answer`.
- **Adversarial** — rewrites the question to bias toward the negative answer via framing. Output column: `adversarial_question`.

Results are saved incrementally row by row to `dataset_construction/data/perturb_result/`. The output CSVs contain all original columns plus the new perturbation columns.

### Run both types (default)

```bash
python run_pipeline.py
```

### Run a single type

```bash
python run_pipeline.py --type paraphrase
python run_pipeline.py --type adversarial
```

### Debug run (10 rows)

```bash
python value_stability/pipeline.py --debug
```

---

## Resumability

All pipelines support resumable runs. Re-run the exact same command after an interruption — already-processed rows are detected and skipped automatically.

---

## Server note

On a GPU server with `flash_attention_2` available, swap the attention implementation in `pipeline.py`:

```python
# model_kwargs={"attn_implementation": "flash_attention_2"}   # GPU server
model_kwargs={"attn_implementation": "sdpa"}                  # CPU / Mac (current)
```
