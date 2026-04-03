import sys
import json
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dataset_construction" / "value_bench"))

from google import genai
from openai import OpenAI
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

OPENROUTER_MODEL = "google/gemma-3-27b-it:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return _parse_response(response.text.strip())
    except Exception as e:
        print(f"[WARN] Gemini call failed for '{value}': {e}")
        return None


def _call_openrouter(client: OpenAI, value: str, question: str, answer: str, definitions_text: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(
                    definitions=definitions_text, value=value, question=question, answer=answer
                )},
            ],
        )
        return _parse_response(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"[WARN] OpenRouter call failed for '{value}': {e}")
        return None


def build_value_mapping(df: pd.DataFrame, backend: str = "gemini") -> dict[str, str]:
    """
    Build a {raw_value: mapped_value} dict.
    - Values already in VALID_CATEGORIES are passed through without an API call.
    - Others are mapped using one representative (question, answer) row as context.
    """
    definitions_text = _build_definitions_text()

    if backend == "openrouter":
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set in your .env file.")
        client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        call_fn = lambda value, question, answer: _call_openrouter(client, value, question, answer, definitions_text)
    else:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in your .env file.")
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        call_fn = lambda value, question, answer: _call_gemini(client, value, question, answer, definitions_text)

    answer_col = "positive_answer" if "positive_answer" in df.columns else df.columns[-1]

    # One representative row per unique value
    representatives = (
        df[["value", "question", answer_col]]
        .dropna(subset=["value"])
        .drop_duplicates(subset=["value"])
        .set_index("value")
    )

    already_valid = {v for v in representatives.index if v in VALID_CATEGORIES}
    needs_mapping = [v for v in representatives.index if v not in VALID_CATEGORIES]

    print(f"  {len(already_valid)} values already match canonical categories (skipping API).")
    print(f"  {len(needs_mapping)} values need mapping.")

    mapping = {v: v for v in already_valid}

    for value in tqdm(needs_mapping, desc=f"Mapping values [{backend}]", colour="cyan"):
        question = representatives.loc[value, "question"]
        answer = representatives.loc[value, answer_col]
        result = call_fn(value, question, answer)
        if result and result in VALID_CATEGORIES:
            mapping[value] = result
        else:
            print(f"[WARN] Could not map '{value}' → got '{result}'. Keeping original.")
            mapping[value] = value

    return mapping


def run(input_csv: Path, output_csv: Path, backend: str = "gemini") -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    unique_count = df["value"].dropna().nunique()
    print(f"Found {unique_count} unique values across {len(df)} rows.")

    mapping = build_value_mapping(df, backend=backend)

    print("\nMapping result:")
    for raw, mapped in mapping.items():
        marker = "" if raw == mapped else f" (was: {raw!r})"
        print(f"  {mapped!r}{marker}")

    df[MAPPED_VALUE_COL] = df["value"].map(mapping)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows → {output_csv}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map dataset values to canonical categories.")
    parser.add_argument(
        "--backend", choices=["gemini", "openrouter"], default="gemini",
        help="gemini: Google Gemini 2.0 Flash | openrouter: Gemma 3 27B via OpenRouter (default: gemini)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_csv = config.INPUT_CSV.parent / "value_bench.csv"
    output_csv = config.INPUT_CSV.parent / "value_bench_mapped.csv"
    run(input_csv, output_csv, backend=args.backend)
