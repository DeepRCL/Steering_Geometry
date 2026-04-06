import sys
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dataset_construction" / "value_bench"))

from google import genai
import pandas as pd
from tqdm import tqdm

import config
from mapper_prompts import MAPPED_VALUE_COL, SYSTEM_PROMPT, USER_PROMPT, VALUE_DEFINITIONS

VALID_CATEGORIES: set[str] = set(VALUE_DEFINITIONS.keys())


def _build_definitions_text() -> str:
    lines = []
    for category, defn in VALUE_DEFINITIONS.items():
        sub_parts = "; ".join(
            f"{sv} → [{', '.join(descriptors)}]"
            for sv, descriptors in defn["sub_values"].items()
        )
        lines.append(f"- Category: {category} — Overview: {defn['overview']} — Sub-values: {sub_parts}.")
    return "\n".join(lines)


def _parse_response(raw: str) -> str | None:
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw).get("mapped_value")


def _call_gemini(
    client: genai.Client,
    value: str,
    question: str,
    positive_answer: str,
    negative_answer: str,
    definitions_text: str,
) -> str | None:
    prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT.format(
        definitions=definitions_text,
        value=value,
        question=question,
        positive_answer=positive_answer,
        negative_answer=negative_answer,
    )
    try:
        response = client.models.generate_content(model=config.GEMINI_MODEL, contents=prompt)
        return _parse_response(response.text.strip())
    except Exception as e:
        tqdm.write(f"[WARN] Gemini call failed for '{value}': {e}")
        return None


def _append_rows(rows: pd.DataFrame, output_csv: Path) -> None:
    write_header = not output_csv.exists()
    rows.to_csv(output_csv, mode="a", header=write_header, index=False)


def run(input_csv: Path, output_csv: Path) -> None:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in your .env file.")

    df = pd.read_csv(input_csv)

    # ── Phase 1: rows already matching a canonical Schwartz value ────────────
    mask_valid = df["value"].isin(VALID_CATEGORIES)
    df_valid = df[mask_valid].copy()
    df_needs = df[~mask_valid].copy()

    print(f"Phase 1 — {len(df_valid)} rows already have canonical values → saving immediately.")
    if len(df_valid):
        df_valid[MAPPED_VALUE_COL] = df_valid["value"]
        _append_rows(df_valid, output_csv)

    # ── Phase 2: map remaining unique values via Gemini ──────────────────────
    unique_remaining = df_needs["value"].dropna().unique().tolist()
    print(f"Phase 2 — {len(df_needs)} rows need mapping ({len(unique_remaining)} unique values).")

    if output_csv.exists():
        already_saved = set(pd.read_csv(output_csv)["value"].unique())
        unique_remaining = [v for v in unique_remaining if v not in already_saved]
        print(f"  Resuming: {len(unique_remaining)} unique values still to process.")

    if not unique_remaining:
        print("Nothing left to map.")
        return

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    definitions_text = _build_definitions_text()

    representatives = (
        df_needs[["value", "question", "positive_answer", "negative_answer"]]
        .drop_duplicates(subset=["value"])
        .set_index("value")
    )

    for value in tqdm(unique_remaining, desc="Mapping values", colour="cyan"):
        rep = representatives.loc[value]
        result = _call_gemini(
            client,
            value=value,
            question=rep["question"],
            positive_answer=rep["positive_answer"],
            negative_answer=rep["negative_answer"],
            definitions_text=definitions_text,
        )

        mapped = result if (result in VALID_CATEGORIES or result == "NA") else "NA"
        if mapped == "NA" and result not in VALID_CATEGORIES:
            tqdm.write(f"[WARN] '{value}' → '{result}' not in canonical list, saving as NA.")

        rows = df_needs[df_needs["value"] == value].copy()
        rows[MAPPED_VALUE_COL] = mapped
        _append_rows(rows, output_csv)
        tqdm.write(f"  {len(rows)} row(s): '{value}' → '{mapped}'")

    total = len(pd.read_csv(output_csv))
    print(f"\nDone. {total} rows saved → {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run on first 10 rows only")
    args = parser.parse_args()

    _DATA = Path("/Users/hamidrezaei/Workspace/Steering_Geometry/dataset_construction/data")
    input_csv = _DATA / "dataset_negative_answer.csv"

    if args.debug:
        debug_input = _DATA / "dataset_negative_answer_debug_input.csv"
        pd.read_csv(input_csv).iloc[110:120].to_csv(debug_input, index=False)
        output_csv = _DATA / "dataset_negative_answer_debug_mapped.csv"
        if output_csv.exists():
            output_csv.unlink()
        print("=== DEBUG MODE: 10 rows ===")
        run(debug_input, output_csv)
    else:
        output_csv = _DATA / "dataset_negative_answer_mapped.csv"
        run(input_csv, output_csv)
