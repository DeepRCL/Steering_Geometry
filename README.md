# Steering Geometry — Dataset Construction

Generates negative or positive answers for the ValueBench dataset using a local Qwen language model.

---

## Project structure

```
.
├── config.py                          # loads .env and exposes typed config
├── .env                               # your secrets/settings (git-ignored)
├── .env.example                       # template to copy from
├── environment.yml                    # conda environment
├── utils/
│   ├── __init__.py
│   └── utils.py                       # parse_json, load_pending_rows
└── dataset_construction/
    ├── pipeline.py                    # model loading, generation, dataset building
    ├── prompt.py                      # system/user prompt templates and examples
    ├── run_pipelines.py               # main entry point (CLI)
    └── data/
        ├── dataset_positive_only.csv  # input: questions + positive answers
        ├── dataset_negative_only.csv  # input: questions + negative answers
        ├── debug_input.csv            # auto-created sample for debug runs
        └── debug_output.csv          # output of debug runs
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
| `MODEL_ID` | HuggingFace model ID | `Qwen/Qwen3.5-2B` |
| `MAX_NEW_TOKENS` | Max tokens the model generates per row | `512` |
| `DEVICE_MAP` | `auto`, `cpu`, or `cuda` | `auto` |
| `INPUT_CSV` | Input filename inside `dataset_construction/data/` | `dataset_positive_only.csv` |
| `BATCH_SIZE` | Rows processed before saving a checkpoint | `10` |
| `DEBUG_ROWS` | Rows used in a debug run | `10` |

---

## Running

All runs go through `run_pipelines.py`:

```bash
python dataset_construction/run_pipelines.py [--mode MODE] [--method METHOD]
```

### `--mode`

| Value | Reads from | Generates | Output file |
|---|---|---|---|
| `negative` (default) | `positive_answer` | `negative_answer` | `data/dataset_negative_answer.csv` |
| `positive` | `negative_answer` | `positive_answer` | `data/dataset_positive_answer.csv` |

### `--method`

| Value | How it calls the model | Best for |
|---|---|---|
| `single` (default) | one model call per row | Mac / CPU / debugging |
| `batch` | one model call per batch | GPU server |

### Examples

```bash
# generate negatives, one row at a time (default)
python dataset_construction/run_pipelines.py

# generate negatives, batch mode
python dataset_construction/run_pipelines.py --method batch

# generate positives, one row at a time
python dataset_construction/run_pipelines.py --mode positive

# generate positives, batch mode
python dataset_construction/run_pipelines.py --mode positive --method batch
```

### Debug run

Run `pipeline.py` directly to test on a small sample (`DEBUG_ROWS` rows from `INPUT_CSV`):

```bash
python dataset_construction/pipeline.py
```

The debug sample is created once at `data/debug_input.csv` and reused on subsequent runs.

---

## Resumability

If the run is interrupted, re-run the exact same command. The script detects the partially-filled output CSV and skips already-completed rows, picking up where it left off.

---

## Server note

On a GPU server with `flash_attention_2` available, swap the attention implementation in `pipeline.py`:

```python
# model_kwargs={"attn_implementation": "flash_attention_2"}   # GPU server
model_kwargs={"attn_implementation": "sdpa"}                  # CPU / Mac (current)
```
