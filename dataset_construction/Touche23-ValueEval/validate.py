"""
Post-generation validation for Touche23-ValueEval negative answers.

Checks whether generated negative_answers inappropriately invoke the target
value (Issues 1 & 2: value-invoking negatives, same-value opposite-policy).

For each sampled row an LLM judge decides:
  "Does the negative answer invoke or rely on the target value?"

Results are saved to a CSV report and a summary is printed to the console.

Usage
-----
    python dataset_construction/Touche23-ValueEval/validate.py [options]

Options
-------
--input    Path to generated CSV  (default: data/touche_dataset_negative_answer.csv)
--sample   Rows to validate; 0 = all rows  (default: 200)
--output   Path for the report CSV  (default: data/validation_report.csv)
--seed     Random seed for sampling  (default: 42)
--value    Validate only rows with this value label (optional filter)
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root and make it importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402 (project-level config)

# ---------------------------------------------------------------------------
# Import DatasetConstructionPipeline from value_bench by explicit file path
# (same approach as pipeline.py to avoid name collision between pipeline.py
# files and to ensure value_bench's internal imports resolve correctly).
# ---------------------------------------------------------------------------
_VB_DIR = _ROOT / "dataset_construction" / "value_bench"
_VB_PIPELINE_PATH = _VB_DIR / "pipeline.py"

sys.path.insert(0, str(_VB_DIR))
_spec = importlib.util.spec_from_file_location("value_bench_pipeline", _VB_PIPELINE_PATH)
_vb_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vb_module)
sys.path.remove(str(_VB_DIR))

for _m in ("prompt", "utils"):
    sys.modules.pop(_m, None)

DatasetConstructionPipeline = _vb_module.DatasetConstructionPipeline

# Now import Touche-local helpers
from prompt import get_concept_line  # noqa: E402
from utils import parse_json  # noqa: E402  (Touche23-ValueEval/utils.py is not present;
                               # falls back to project-level utils/utils.py via sys.path)

# ---------------------------------------------------------------------------
# Validation prompts
# ---------------------------------------------------------------------------

VALIDATION_SYSTEM_PROMPT = """\
You are a strict quality validator for an AI value-steering dataset.

Your task: determine whether a negative argument inappropriately invokes the
target value it is supposed to oppose.

A negative argument INVOKES the target value if:
- It appeals to the target value as a justification or reason.
- It argues that the policy violates or fails to properly serve the target value
  (i.e., it still treats the target value as important, even to criticise it).
- The core concept of the target value appears in the argument's reasoning.

A negative argument does NOT invoke the target value if:
- It approaches the question from a completely different perspective or value.
- The target value concept is entirely absent from its reasoning and conclusion.

Respond ONLY with a JSON object — no extra text:
{
  "invokes_target_value": true or false,
  "confidence": "high" | "medium" | "low",
  "explanation": "one sentence explaining the judgment"
}
"""

VALIDATION_USER_PROMPT = """\
Target Value: {value}
Value Concept: {concept_line}

Positive Answer (correctly invokes the target value):
  {positive_answer}

Negative Answer (should NOT invoke the target value):
  {negative_answer}

Does the negative answer invoke or rely on the target value "{value}"?
"""


# ---------------------------------------------------------------------------
# Validation pipeline subclass
# ---------------------------------------------------------------------------

class ValidationPipeline(DatasetConstructionPipeline):
    """
    Reuses the model-loading and generation infrastructure from
    DatasetConstructionPipeline to run the LLM-as-judge validation.

    Only _build_messages() is overridden; generation is inherited.
    """

    def _build_messages(self, row, direction=None) -> list[dict]:
        concept = get_concept_line(str(row["value"]))
        user = VALIDATION_USER_PROMPT.format(
            value=row["value"],
            concept_line=concept,
            positive_answer=row["positive_answer"],
            negative_answer=row["negative_answer"],
        )
        return [
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ]

    # ------------------------------------------------------------------
    def validate(
        self,
        df: pd.DataFrame,
        output_csv: Path,
    ) -> pd.DataFrame:
        """
        Run validation on every row in *df*.

        Adds columns: invokes_target_value, confidence, explanation, raw_response.
        Saves the full report to *output_csv* and returns it.
        """
        results: list[dict] = []
        total = len(df)

        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"  [{i:>4}/{total}]  value={row['value']!r}", end=" ... ", flush=True)

            messages = self._build_messages(row)
            raw = self._generate(messages)

            invokes  = parse_json(raw, "invokes_target_value")
            confidence = parse_json(raw, "confidence")
            explanation = parse_json(raw, "explanation")

            # Normalise the bool field (parse_json returns actual Python bool
            # when JSON contains true/false; only falls back to "..." on error)
            if isinstance(invokes, bool):
                invokes_bool = invokes
            elif isinstance(invokes, str) and invokes.lower() in ("true", "yes"):
                invokes_bool = True
            elif isinstance(invokes, str) and invokes.lower() in ("false", "no"):
                invokes_bool = False
            else:
                invokes_bool = None  # parse failure

            label = "INVOKES" if invokes_bool else ("OK" if invokes_bool is False else "ERR")
            print(label)

            results.append({
                **row.to_dict(),
                "invokes_target_value": invokes_bool,
                "confidence":           str(confidence) if confidence else "",
                "explanation":          str(explanation) if explanation else "",
                "raw_response":         raw,
            })

        report = pd.DataFrame(results)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_csv, index=False)
        print(f"\nReport saved → {output_csv}")
        return report


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(report: pd.DataFrame) -> None:
    valid = report[report["invokes_target_value"].notna()]
    if valid.empty:
        print("No valid judgments to summarise.")
        return

    total  = len(valid)
    n_bad  = int(valid["invokes_target_value"].sum())
    rate   = 100 * n_bad / total

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Rows validated : {total}")
    print(f"  Contaminated   : {n_bad}  ({rate:.1f}%)")
    print(f"  Clean          : {total - n_bad}  ({100 - rate:.1f}%)")

    if report["invokes_target_value"].isna().any():
        n_err = report["invokes_target_value"].isna().sum()
        print(f"  Parse errors   : {n_err}")

    # Per-value breakdown (only show values with at least 1 contaminated row)
    if n_bad > 0:
        print("\n  Contamination by value (worst first):")
        breakdown = (
            valid.groupby("value")["invokes_target_value"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "bad", "count": "n"})
        )
        breakdown["rate"] = 100 * breakdown["bad"] / breakdown["n"]
        breakdown = breakdown[breakdown["bad"] > 0].sort_values("rate", ascending=False)
        for value, row in breakdown.iterrows():
            print(f"    {value:<42s}  {int(row['bad'])}/{int(row['n'])}  ({row['rate']:.0f}%)")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _DATA_DIR = _HERE.parent / "data"

    parser = argparse.ArgumentParser(
        description="Validate Touche23-ValueEval negative answers for value invocation."
    )
    parser.add_argument(
        "--input",
        default=str(_DATA_DIR / "touche_dataset_negative_answer.csv"),
        help="Path to the generated CSV file (default: data/touche_dataset_negative_answer.csv)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help="Number of rows to validate; 0 = all rows (default: 200)",
    )
    parser.add_argument(
        "--output",
        default=str(_DATA_DIR / "validation_report.csv"),
        help="Path for the validation report CSV (default: data/validation_report.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for row sampling (default: 42)",
    )
    parser.add_argument(
        "--value",
        default=None,
        help="Optional: validate only rows with this exact value label",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    # ------------------------------------------------------------------
    # Load and optionally filter the dataset
    # ------------------------------------------------------------------
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = {"value", "positive_answer", "negative_answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    # Drop rows with empty negative_answer (not yet generated)
    before = len(df)
    df = df[df["negative_answer"].notna() & (df["negative_answer"].astype(str).str.strip() != "")]
    if len(df) < before:
        print(f"Skipped {before - len(df)} rows with empty negative_answer.")

    if args.value:
        df = df[df["value"] == args.value]
        print(f"Filtered to value={args.value!r}: {len(df)} rows.")

    if len(df) == 0:
        print("No rows to validate. Exiting.")
        return

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------
    if args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {len(df)} rows (seed={args.seed}).")
    else:
        df = df.reset_index(drop=True)
        print(f"Validating all {len(df)} rows.")

    # ------------------------------------------------------------------
    # Run validation
    # ------------------------------------------------------------------
    print("\nInitialising validation pipeline …")
    pipeline = ValidationPipeline()

    print(f"Running LLM-as-judge on {len(df)} rows …\n")
    report = pipeline.validate(df, output_path)

    print_summary(report)


if __name__ == "__main__":
    main()
