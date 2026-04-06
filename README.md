# Steering Geometry — Dataset Construction

Builds augmented datasets for value-alignment research. Currently supports two pipelines:

1. **ValueBench** — generates positive/negative answers for the ValueBench dataset using a local Qwen model, then maps raw value labels to canonical Schwartz categories via Gemini.
2. **Touche23-ValueEval** — separate pipeline for the Touche23 dataset.

---

## Project structure

```
.
├── config.py                                  # loads .env and exposes typed config
├── .env                                       # your secrets/settings (git-ignored)
├── .env.example                               # template to copy from
├── environment.yml                            # conda environment
├── utils/
│   ├── __init__.py
│   └── utils.py                               # shared utilities (parse_json, load_pending_rows)
└── dataset_construction/
    ├── data/                                  # shared intermediate data
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

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face token ([get one here](https://huggingface.co/settings/tokens)) | — |
| `GEMINI_API_KEYS` | Comma-separated Gemini API keys for value mapping (rotates on rate limit) | — |
| `GEMINI_MODEL` | Gemini model name | `gemini-3.1-flash-lite-preview` |
| `MODEL_ID` | HuggingFace model ID | `Qwen/Qwen3.5-2B` |
| `MAX_NEW_TOKENS` | Max tokens generated per row | `512` |
| `DEVICE_MAP` | `auto`, `cpu`, or `cuda` | `auto` |
| `INPUT_CSV` | Input filename inside `dataset_construction/data/` | `dataset_positive_only.csv` |
| `BATCH_SIZE` | Rows processed before saving a checkpoint | `10` |
| `DEBUG_ROWS` | Rows used in a debug run | `10` |

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

## Resumability

All pipelines support resumable runs. Re-run the exact same command after an interruption — already-processed rows are detected and skipped automatically.

---

## Server note

On a GPU server with `flash_attention_2` available, swap the attention implementation in `pipeline.py`:

```python
# model_kwargs={"attn_implementation": "flash_attention_2"}   # GPU server
model_kwargs={"attn_implementation": "sdpa"}                  # CPU / Mac (current)
```
