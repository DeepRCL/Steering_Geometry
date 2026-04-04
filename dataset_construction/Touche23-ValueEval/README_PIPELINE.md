# Touche23-ValueEval LLM Dataset Construction Pipeline

This pipeline transforms the Touche23 dataset into a format suitable for generating synthetic negatives via LLM: `question, value, positive_answer, negative_answer`.

## Files

### Stage 1: Preprocessing (Deterministic, No LLM)

**`preprocessing.ipynb`**

Transforms raw Touche TSV files into `data/touche_positive_only.csv`.

- Loads `arguments-{split}.tsv` and `labels-{split}.tsv` for training/validation/test
- Joins on `Argument ID`
- Melts 20 value columns → one row per active value label
- **Multi-label handling**: Arguments with multiple active values are automatically duplicated (one copy per value)
  - Example: argument with `Security: personal=1` AND `Universalism: concern=1` becomes 2 rows
  - Both rows share the same `question` and `positive_answer` (premise)
  - Each row has a different `value`
- Converts conclusions to questions:
  - `"We should ban X"` → `"Should we ban X?"`
  - `"X should Y"` → `"Should X Y?"`
  - Declarative (no "should") → append `"?"`
- Outputs columns: `argument_id, split, stance, question, value, positive_answer, negative_answer` (empty)

**Result:** 29,500 rows from 8,868 unique arguments across all splits

**Value Distribution:**
```
Universalism: concern       3,356  (11.4%)
Security: personal          3,296  (11.2%)
Security: societal          2,613  (8.9%)
Achievement                 2,499  (8.5%)
Benevolence: caring         2,301  (7.8%)
Self-direction: action      2,282  (7.7%)
Conformity: rules           1,919  (6.5%)
Universalism: objectivity   1,896  (6.4%)
[... 12 more values ...]
Mean: 1,475 rows/value; Median: 1,160; StdDev: 974
```

### Stage 2: LLM Generation

**`prompt.py`**

Touche-specific prompts and value definitions.

- `get_definition(value)` — combines two sources:
  - `VALUEBENCH_DEFINITIONS` (13 parent clusters): philosophical concept
  - `value-categories.json` (20 fine-grained labels): concrete sub-values + example effects
  - 19/20 values get both sources; `Universalism: objectivity` falls back to JSON-only
- System prompt uses a **5-step chain-of-thought** process the LLM must follow before answering:
  1. Identify what the value means in the policy context
  2. Understand the positive argument's claim
  3. Select one rhetorical strategy (challenge, trade-off, competing value, counter-example, assumption attack)
  4. Draft the negative answer
  5. Handle minimal-sub-category values (Hedonism, Power: resources) by focusing on the core concept
- LLM responds with a two-key JSON: `"thinking"` (internal reasoning, discarded) and `"negative_answer"` (stored)
- 5 few-shot examples in Touche argumentative style, each with an explicit `Thinking:` block and a different rhetorical strategy; includes examples for single-sub-value labels (Hedonism, Power: resources)

**`pipeline.py`**

Subclasses `DatasetConstructionPipeline` from `value_bench/pipeline.py`, overriding only `_build_messages()`.

- Handles circular import (both files named `pipeline.py`) via `importlib.util`
- All generation logic inherited:
  - `_generate()` — one model call per row
  - `_generate_batch()` — one call per batch
  - `build_dataset_single()` — single mode + checkpointing
  - `build_dataset_batch()` — batch mode + checkpointing
  - Resumability — if interrupted, re-run same command to pick up where it left off

**`run_pipelines.py`**

CLI entry point. Usage:

```bash
# Generate all negatives, one row at a time (safe on CPU/Mac)
python dataset_construction/Touche23-ValueEval/run_pipelines.py

# Batch mode (faster on GPU)
python dataset_construction/Touche23-ValueEval/run_pipelines.py --method batch
```

**Input:** `data/touche_positive_only.csv` (29,500 rows)
**Output:** `data/touche_dataset_negative_answer.csv` (29,500 rows with filled `negative_answer` column)

## Configuration

Uses `.env` variables from the project root:
- `MODEL_ID` — HuggingFace model ID (default: `Qwen/Qwen3.5-2B`)
- `MAX_NEW_TOKENS` — max tokens per generation (default: 512)
- `DEVICE_MAP` — `auto`, `cpu`, or `cuda` (default: `auto`)
- `BATCH_SIZE` — rows per checkpoint (default: 10)
- `DEBUG_ROWS` — sample size for debug runs (default: 10)

## Data Format

### Input: `touche_positive_only.csv`
```csv
argument_id,split,stance,question,value,positive_answer,negative_answer
A01010,training,against,Should we prohibit school prayer?,Self-direction: thought,"it should be allowed if the student wants...",
A01020,training,in favor of,Should we subsidize journalism?,Self-direction: thought,"It is important for news organizations...",
```

### Output: `touche_dataset_negative_answer.csv`
```csv
argument_id,split,stance,question,value,positive_answer,negative_answer
A01010,training,against,Should we prohibit school prayer?,Self-direction: thought,"it should be allowed if the student wants...","Mandatory school prayer violates the very freedom of thought this argument claims to protect..."
A01020,training,in favor of,Should we subsidize journalism?,Self-direction: thought,"It is important for news organizations...","Market-driven journalism has historically produced equally rigorous reporting as subsidized systems..."
```

## Key Design Decisions

1. **Multi-label duplication** — Each value gets its own row, maximizing dataset utility while preserving argument fidelity
2. **Premise-as-positive** — Touche premises are already positive instantiations of the labeled values, so no LLM rewriting needed for the positive side
3. **Combined definitions** — VALUEBENCH provides abstract principles; `value-categories.json` provides concrete grounding; together they enable diverse rhetorical strategies
4. **Argumentative style** — Stays in policy-debate register (not first-person), matching Touche's original data
5. **Diversity instruction** — System prompt explicitly asks LLM to vary attack strategies to avoid monotonous negatives
6. **Chain-of-thought reasoning** — LLM outputs a `thinking` field before the final answer, improving reasoning quality; only `negative_answer` is stored in the dataset
7. **Single-sub-value coverage** — Few-shot examples include Hedonism and Power: resources (each with only one sub-value) so the LLM handles all 20 value types correctly

## Next Steps

1. Run `preprocessing.ipynb` to generate intermediate CSV (already done in this implementation)
2. Configure `.env` with your model/device preferences
3. Run `python dataset_construction/Touche23-ValueEval/run_pipelines.py` to generate final dataset
4. Output will be saved to `data/touche_dataset_negative_answer.csv`
