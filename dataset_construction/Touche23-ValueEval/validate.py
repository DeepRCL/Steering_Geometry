"""
Post-generation validation for Touche23-ValueEval negative answers.

Judges each negative answer on four criteria using an LLM-as-judge:
  1. invokes_target_value   — does the negative rely on the target value?
  2. non_endorsing          — does it avoid implicitly endorsing the value?
  3. rhetorically_coherent  — is it a genuine, topic-specific argument?
  4. quality                — "poor" / "acceptable" / "good"

Derives a final caa_suitable flag:
  caa_suitable = invokes_target_value is False
                 AND non_endorsing is True
                 AND rhetorically_coherent is True
                 AND quality != "poor"

Results are appended row-by-row to the output CSV so that interrupted runs
can be safely resumed without losing previously evaluated rows.  On restart
the script detects already-evaluated rows by their (argument_id, value,
negative_answer[:64]) key and skips them automatically.

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
You are a strict quality validator for an AI value-steering dataset used for
Contrastive Activation Addition (CAA). Each row pairs a positive answer
(agreeing with a human value) and a negative answer (designed NOT to invoke
that value).

Judge the negative answer on FOUR criteria. All four use the same structure:
a definition of when the field is true and when it is false.

──────────────────────────────────────────────────────────────────────────
1. INVOKES_TARGET_VALUE
   Does the negative invoke or rely on the target value in its reasoning?
   → true  if it appeals to the target value as justification; if it argues
            the policy violates or fails to serve the target value (treating
            the value as important even to criticise it); or if the target
            value's core concept appears in the argument's logic.
   → false if it approaches the question from a completely different
            perspective; or if the target value concept is entirely absent
            from its reasoning and conclusion.

2. NON_ENDORSING
   Does the negative avoid implicitly endorsing the target value?
   Neutral and opposing answers both pass; only implicit endorsement fails.
   → true  if its conclusion holds regardless of whether you accept the
            target value; if it argues from a pragmatic, empirical, or
            unrelated perspective whose validity is independent of the value.
   → false if its conclusion only makes sense by accepting the target value
            as important; if it implicitly treats the policy as wrong because
            it fails the target value — even without naming the value directly.

3. RHETORICALLY_COHERENT
   Does the negative make a genuine, topic-specific argument?
   → true  if it makes a specific claim that directly responds to this
            question or policy; if its reasoning is connected to the topic
            rather than being a generic template that fits any debate.
   → false if its argument could apply verbatim to any policy debate; if it
            makes no specific claim about this particular question; or if it
            is vague, incoherent, or entirely off-topic.

4. QUALITY  (choose exactly one label)
   → "good"       Specific, well-reasoned, directly engages this policy.
                  A reader would find it persuasive on its own terms.
   → "acceptable" Coherent and on-topic but generic enough to fit similar
                  policies with little adjustment.
   → "poor"       Vague, incoherent, off-topic, or adds no substantive claim
                  beyond empty assertions.
──────────────────────────────────────────────────────────────────────────

First reason step by step in the "thinking" field (2–4 sentences), then give
your four judgments.

Respond ONLY with a JSON object — no extra text, no markdown fences:
{
  "thinking": "step-by-step reasoning before committing to answers",
  "invokes_target_value": true or false,
  "non_endorsing": true or false,
  "rhetorically_coherent": true or false,
  "quality": "poor" | "acceptable" | "good",
  "confidence": "high" | "medium" | "low",
  "explanation": "one sentence summarising the overall judgment"
}
"""

VALIDATION_USER_PROMPT = """\
Target Value  : {value}
Value Concept : {concept_line}
Question      : {question}

Positive Answer (correctly invokes the target value):
  {positive_answer}

Negative Answer (should NOT invoke the target value):
  {negative_answer}

Judge the negative answer on all four criteria for target value "{value}".
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bool(raw_value) -> bool | None:
    """Normalise a JSON boolean or string representation to Python bool."""
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        if raw_value.lower() in ("true", "yes"):
            return True
        if raw_value.lower() in ("false", "no"):
            return False
    return None  # parse failure


def _derive_caa_suitable(
    invokes: bool | None,
    non_endorsing: bool | None,
    coherent: bool | None,
    quality: str | None,
) -> bool | None:
    """
    Derive the final suitability flag from the four judge criteria.
    Returns None if any required field failed to parse.
    """
    if any(x is None for x in [invokes, non_endorsing, coherent, quality]):
        return None
    return (
        invokes is False
        and non_endorsing is True
        and coherent is True
        and quality != "poor"
    )


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
        question = str(row.get("question", "")) if hasattr(row, "get") else ""
        user = VALIDATION_USER_PROMPT.format(
            value=row["value"],
            concept_line=concept,
            question=question,
            positive_answer=row["positive_answer"],
            negative_answer=row["negative_answer"],
        )
        return [
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ]

    @staticmethod
    def _row_key(row) -> tuple:
        """
        Composite key used to detect already-evaluated rows when resuming.
        Uses argument_id (if present) + value + first 64 chars of negative_answer.
        """
        arg_id = str(row.get("argument_id", "")) if hasattr(row, "get") else ""
        value  = str(row["value"])
        neg64  = str(row["negative_answer"])[:64]
        return (arg_id, value, neg64)

    # ------------------------------------------------------------------
    def validate(
        self,
        df: pd.DataFrame,
        output_csv: Path,
    ) -> pd.DataFrame:
        """
        Run validation on every row in *df*.

        Appends results row-by-row to *output_csv* immediately after each
        judgment so partial runs survive interruptions.  On restart, rows
        whose key already appears in the output file are skipped.

        Output columns added to the input dataset:
          val_invokes_target_value, val_non_endorsing,
          val_rhetorically_coherent, val_quality,
          val_confidence, val_explanation, val_thinking,
          val_raw_response, caa_suitable
        """
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        # ── Resume: detect previously evaluated rows ──────────────────
        already_done: set[tuple] = set()
        if output_csv.exists() and output_csv.stat().st_size > 0:
            try:
                existing = pd.read_csv(output_csv)
                for _, r in existing.iterrows():
                    already_done.add(self._row_key(r))
                print(f"  Resume mode: {len(already_done)} rows already evaluated "
                      f"in {output_csv.name}")
            except Exception as exc:
                print(f"  Warning: could not read existing output ({exc}). "
                      "Starting fresh.")

        total = len(df)
        n_skipped = 0

        for i, (_, row) in enumerate(df.iterrows(), 1):
            key = self._row_key(row)

            if key in already_done:
                n_skipped += 1
                print(f"  [{i:>4}/{total}]  value={row['value']!r} ... SKIP")
                continue

            print(f"  [{i:>4}/{total}]  value={row['value']!r}", end=" ... ", flush=True)

            messages = self._build_messages(row)
            raw = self._generate(messages)

            # ── Parse judge fields ────────────────────────────────────
            invokes     = _parse_bool(parse_json(raw, "invokes_target_value"))
            non_end     = _parse_bool(parse_json(raw, "non_endorsing"))
            coherent    = _parse_bool(parse_json(raw, "rhetorically_coherent"))
            quality_raw = parse_json(raw, "quality")
            confidence  = parse_json(raw, "confidence")
            explanation = parse_json(raw, "explanation")
            thinking    = parse_json(raw, "thinking")

            quality_val = str(quality_raw).lower() if quality_raw else None
            if quality_val not in ("poor", "acceptable", "good"):
                quality_val = None

            caa = _derive_caa_suitable(invokes, non_end, coherent, quality_val)

            # ── Console status line ───────────────────────────────────
            parse_failed = any(
                x is None for x in [invokes, non_end, coherent, quality_val]
            )
            if parse_failed:
                label = "ERR"
            elif caa:
                label = "OK"
            else:
                reasons = []
                if invokes:               reasons.append("INVOKES")
                if not non_end:           reasons.append("ENDORSES")
                if not coherent:          reasons.append("INCOHERENT")
                if quality_val == "poor": reasons.append("POOR")
                label = "+".join(reasons) if reasons else "FAIL"
            print(label)

            result = {
                **row.to_dict(),
                "val_invokes_target_value":  invokes,
                "val_non_endorsing":         non_end,
                "val_rhetorically_coherent": coherent,
                "val_quality":               quality_val,
                "val_confidence":            str(confidence)  if confidence  else "",
                "val_explanation":           str(explanation) if explanation else "",
                "val_thinking":              str(thinking)    if thinking    else "",
                "val_raw_response":          raw,
                "caa_suitable":              caa,
            }

            # ── Append this row to the output CSV immediately ─────────
            row_df = pd.DataFrame([result])
            file_is_new = not output_csv.exists() or output_csv.stat().st_size == 0
            row_df.to_csv(output_csv, mode="a", header=file_is_new, index=False)

        if n_skipped:
            print(f"\n  {n_skipped} row(s) skipped (already evaluated).")
        print(f"\nReport saved → {output_csv}")

        # Return the full report including previously-evaluated rows
        return pd.read_csv(output_csv)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(report: pd.DataFrame) -> None:
    # Work only on rows that have validation results
    val_cols = [
        "val_invokes_target_value",
        "val_non_endorsing",
        "val_rhetorically_coherent",
        "val_quality",
        "caa_suitable",
    ]
    missing = [c for c in val_cols if c not in report.columns]
    if missing:
        print(f"Summary skipped — missing columns: {missing}")
        return

    # Rows where at least caa_suitable was resolved
    valid = report[report["caa_suitable"].notna()].copy()
    if valid.empty:
        print("No valid judgments to summarise.")
        return

    total = len(valid)

    def _count(col, val=True):
        if col not in valid.columns:
            return 0
        return int((valid[col] == val).sum())

    n_invokes   = _count("val_invokes_target_value", True)
    n_endorses  = _count("val_non_endorsing", False)
    n_incoherent = _count("val_rhetorically_coherent", False)
    n_poor      = _count("val_quality", "poor")
    n_suitable  = _count("caa_suitable", True)

    n_acceptable = _count("val_quality", "acceptable")
    n_good       = _count("val_quality", "good")

    n_err = int(report["caa_suitable"].isna().sum())

    pct = lambda n: f"{100 * n / total:.1f}%"  # noqa: E731

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Rows evaluated              : {total}")
    print()
    print(f"  Contaminated  (invokes val) : {n_invokes:>4}  ({pct(n_invokes)})")
    print(f"  Endorses val  (non_endorsing=F) : {n_endorses:>4}  ({pct(n_endorses)})")
    print(f"  Incoherent                  : {n_incoherent:>4}  ({pct(n_incoherent)})")
    print(f"  Poor quality                : {n_poor:>4}  ({pct(n_poor)})")
    if n_err:
        print(f"  Parse errors                : {n_err:>4}")
    print()
    print(f"  ── CAA-suitable (all ✓)     : {n_suitable:>4}  ({pct(n_suitable)})")
    print()
    print(f"  Quality distribution:")
    print(f"    good        {n_good:>4}  ({pct(n_good)})")
    print(f"    acceptable  {n_acceptable:>4}  ({pct(n_acceptable)})")
    print(f"    poor        {n_poor:>4}  ({pct(n_poor)})")

    # Per-value contamination breakdown
    if n_invokes > 0 and "value" in valid.columns:
        print("\n  Contamination by value (worst first):")
        breakdown = (
            valid.groupby("value")["val_invokes_target_value"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "bad", "count": "n"})
        )
        breakdown["rate"] = 100 * breakdown["bad"] / breakdown["n"]
        breakdown = breakdown[breakdown["bad"] > 0].sort_values("rate", ascending=False)
        for value, row in breakdown.iterrows():
            print(f"    {value:<42s}  {int(row['bad'])}/{int(row['n'])}  "
                  f"({row['rate']:.0f}%)")

    # Per-value CAA suitability breakdown
    if "value" in valid.columns:
        print("\n  CAA suitability by value (worst first):")
        suit = (
            valid.groupby("value")["caa_suitable"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "ok", "count": "n"})
        )
        suit["rate"] = 100 * suit["ok"] / suit["n"]
        suit = suit.sort_values("rate", ascending=True)
        for value, row in suit.iterrows():
            print(f"    {value:<42s}  {int(row['ok'])}/{int(row['n'])}  "
                  f"({row['rate']:.0f}%)")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _DATA_DIR = _HERE.parent / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Validate Touche23-ValueEval negative answers using an LLM-as-judge. "
            "Checks four criteria: contamination, endorsement, coherence, quality. "
            "Results are appended row-by-row so interrupted runs can be resumed."
        )
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
