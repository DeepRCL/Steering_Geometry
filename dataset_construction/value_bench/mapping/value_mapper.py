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
from mapper_prompts import MAPPED_VALUE_COL, SYSTEM_PROMPT, USER_PROMPT

_VALUE_CATEGORIES_JSON = (
    _ROOT / "dataset_construction" / "Touche23-ValueEval" / "data" / "value-categories.json"
)

with open(_VALUE_CATEGORIES_JSON) as f:
    VALUE_CATEGORIES: dict = json.load(f)

VALID_CATEGORIES: set[str] = set(VALUE_CATEGORIES.keys())


def _build_definitions_text() -> str:
    lines = []
    for category, sub_values in VALUE_CATEGORIES.items():
        sub_names = ", ".join(sub_values.keys())
        lines.append(f"- {category} (sub-values: {sub_names})")
    return "\n".join(lines)


def _parse_response(raw: str) -> str | None:
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw).get("mapped_value")


def _call_gemini(client: genai.Client, value: str, question: str, answer: str, definitions_text: str) -> str | None:
    prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT.format(
        definitions=definitions_text, value=value, question=question, answer=answer
    )
    try:
        response = client.models.generate_content(model=config.GEMINI_MODEL, contents=prompt)
        return _parse_response(response.text.strip())
    except Exception as e:
        print(f"[WARN] Gemini call failed for '{value}': {e}")
        return None


def build_value_mapping(df: pd.DataFrame) -> dict[str, str]:

    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in your .env file.")

    definitions_text = _build_definitions_text()
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    answer_col = "positive_answer" if "positive_answer" in df.columns else df.columns[-1]

    representatives = (
        df[["value", "question", answer_col]]
        .dropna(subset=["value"])
        .drop_duplicates(subset=["value"])
        .set_index("value")
    )

    already_valid = {v: v for v in representatives.index if v in VALID_CATEGORIES}
    needs_mapping = [v for v in representatives.index if v not in VALID_CATEGORIES]

    print(f"  {len(already_valid)} values already match canonical categories (skipping API).")
    print(f"  {len(needs_mapping)} values need mapping.")

    mapping = {**already_valid}

    for value in tqdm(needs_mapping, desc="Mapping values", colour="cyan"):
        question = representatives.loc[value, "question"]
        answer = representatives.loc[value, answer_col]
        result = _call_gemini(client, value, question, answer, definitions_text)
        if result and result in VALID_CATEGORIES:
            mapping[value] = result
        else:
            tqdm.write(f"[WARN] Could not map '{value}' → got '{result}'. Keeping original.")
            mapping[value] = value

    return mapping


def run(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    unique_count = df["value"].dropna().nunique()
    print(f"Found {unique_count} unique values across {len(df)} rows.")

    mapping = build_value_mapping(df)

    print("\nMapping result:")
    for raw, mapped in mapping.items():
        marker = "" if raw == mapped else f" (was: {raw!r})"
        print(f"  {mapped!r}{marker}")

    df[MAPPED_VALUE_COL] = df["value"].map(mapping)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows → {output_csv}")
    return df


if __name__ == "__main__":
    input_csv = config.INPUT_CSV.parent / "dataset_positive_only.csv"
    output_csv = config.INPUT_CSV.parent / "value_bench_mapped_debug.csv"

    df_full = pd.read_csv(input_csv)
    df_debug = df_full.iloc[90:100].copy()
    df_debug.to_csv(input_csv.parent / "value_bench_debug_input.csv", index=False)

    mapping = build_value_mapping(df_debug)
    df_debug[MAPPED_VALUE_COL] = df_debug["value"].map(mapping)
    df_debug.to_csv(output_csv, index=False)
    print(f"\nDebug output saved → {output_csv}")
