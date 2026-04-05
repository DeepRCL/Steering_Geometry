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
- System prompt uses a **6-step chain-of-thought** process:
  1. Identify what the value means in this specific policy context
  2. Understand what claim the positive argument makes and how it invokes the value
  3. Apply the assigned strategy to this specific question and positive argument
  4. Identify the specific counter-claim, evidence, or consequence the argument will make
  5. Write the negative using concrete policy language (no value label names in the output)
  6. Self-check: does the negative re-endorse the value or name a value label? If so, revise
- Two **critical rules** enforced throughout:
  - Value label names (e.g. "Security: personal") are banned from `negative_answer` — only allowed in `thinking`; this keeps positive and negative in the same implicit register to avoid activation asymmetry in CAA
  - The negative must not re-endorse or soften the target value from a different angle
- LLM responds with a two-key JSON: `"thinking"` (discarded) and `"negative_answer"` (stored)
- **6 few-shot examples**, one per strategy, each with an explicit `Thinking:` block; covers single-sub-value labels (Hedonism, Power: resources)

**`pipeline.py`**

Subclasses `DatasetConstructionPipeline` from `value_bench/pipeline.py`, overriding only `_build_messages()`.

- Handles circular import (both files named `pipeline.py`) via `importlib.util`
- **Strategy rotation** — each row is assigned one of 6 strategies deterministically via `MD5(argument_id + value) % 6`, ensuring near-uniform distribution (~16–17% per strategy) across all rows without exposing any Schwartz value name as the suggested alternative:
  - `pragmatic` — challenge feasibility, cost, or implementation
  - `empirical` — challenge factual/causal claims with evidence
  - `counter-example` — cite a case where the same policy failed
  - `side-effects` — argue for serious unintended consequences in a different domain
  - `institutional` — argue existing rules already address the concern
  - `contradict` — challenge whether the value-based premise is valid or accurately applied
- Dynamic **word-count guidance** — `len(positive_answer.split())` is computed per row and passed to the user prompt as a target range (`N – N+10 words`)
- All generation logic inherited: `_generate()`, `_generate_batch()`, `build_dataset_single()`, `build_dataset_batch()`, resumable checkpointing

**`validate.py`**

Post-generation validation script. Samples the output CSV and uses an LLM judge to check whether each `negative_answer` inappropriately invokes the target value.

```bash
python dataset_construction/Touche23-ValueEval/validate.py --sample 200
python dataset_construction/Touche23-ValueEval/validate.py --input data/touche_dataset_negative_answer.csv --sample 0  # all rows
python dataset_construction/Touche23-ValueEval/validate.py --value "Security: personal" --sample 50
```

Outputs a report CSV (`data/validation_report.csv`) with per-row judgments (`invokes_target_value`, `confidence`, `explanation`) and prints a summary showing overall contamination rate and a per-value breakdown.

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
4. **Strategy rotation for CAA diversity** — Six strategies assigned deterministically per `(argument_id, value)` via MD5 hashing. This enforces near-uniform strategy distribution without passing any Schwartz value name to the LLM, preventing systematic alternative-value bias that would distort steering vectors
5. **No value labels in `negative_answer`** — Keeps both positive and negative in the same implicit, concrete-language register. Naming a value label explicitly in the negative while the positive uses implicit policy language creates an activation asymmetry that contaminates CAA vectors
6. **Register matching, not fixed style** — The negative matches the register and directness of the positive answer rather than a fixed "policy-debate" style, ensuring stylistic symmetry between pairs so steering is driven by content, not form
7. **Chain-of-thought reasoning** — LLM outputs a `thinking` field before the final answer; only `negative_answer` is stored. The 6-step reasoning process includes an explicit self-check for value-label leakage
8. **Dynamic length targeting** — Positive answer word count is computed per row and passed as a target range (`N – N+10 words`), preventing systematic length asymmetry between positive and negative pairs
9. **Single-sub-value coverage** — Few-shot examples include Hedonism and Power: resources so the LLM handles all 20 value types correctly

## Next Steps

1. Run `preprocessing.ipynb` to generate `data/touche_positive_only.csv` (if not already done)
2. Configure `.env` with your model/device preferences
3. Run `python dataset_construction/Touche23-ValueEval/run_pipelines.py` to generate the full dataset
4. Run `python dataset_construction/Touche23-ValueEval/validate.py --sample 200` to check contamination rate before committing API budget to the full run
