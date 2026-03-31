# Steering Geometry — Dataset Construction

Generates negative (or positive) answers for the ValueBench dataset using a local Qwen language model.

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
| `MODEL_ID` | HuggingFace model to use | `Qwen/Qwen3.5-2B` |
| `MAX_NEW_TOKENS` | Max tokens the model can generate per row | `512` |
| `DEVICE_MAP` | `auto`, `cpu`, or `cuda` | `auto` |
| `INPUT_CSV` | Input filename inside `dataset_construction/data/` | `dataset_positive_only.csv` |
| `OUTPUT_CSV` | Output filename inside `dataset_construction/data/` | `dataset_with_negatives.csv` |
| `BATCH_SIZE` | Rows saved per checkpoint | `10` |
| `DEBUG_ROWS` | Rows used in debug run | `10` |

---

## Run

### Full dataset

```bash
python dataset_construction/main.py
```

### Debug run (first `DEBUG_ROWS` rows only)

```bash
python dataset_construction/debug_llm_call.py
```

---

## Single vs Batch mode

Edit `dataset_construction/main.py` and swap the method:

```python
# One model call per row — safe on CPU / Mac
pipe.build_dataset_single(input_csv=..., output_csv=..., target_col="negative_answer", batch_size=10)

# One model call per batch — faster on GPU
pipe.build_dataset_batch(input_csv=..., output_csv=..., target_col="negative_answer", batch_size=10)
```

---

## Resumability

If the run is interrupted, re-run the same command. It will load the partially-filled output CSV and skip already-completed rows.
