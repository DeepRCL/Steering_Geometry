import sys
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dataset_construction" / "value_bench"))

import google.generativeai as genai
import pandas as pd
from tqdm import tqdm

import config
from prompt import VALUEBENCH_DEFINITIONS
from mapper_prompts import SYSTEM_PROMPT, USER_PROMPT


MAPPED_VALUE_COL = "mapped_value"
def _build_definitions_text() -> str:
    return "\n".join(
        f"- {name}: {desc}" for name, desc in VALUEBENCH_DEFINITIONS.items()
    )


def _call_gemini(model: genai.GenerativeModel, value: str, definitions_text: str) -> str | None:
    prompt = USER_PROMPT.format(definitions=definitions_text, value=value)
    try:
        response = model.generate_content(
            [{"role": "user", "parts": [SYSTEM_PROMPT + "\n\n" + prompt]}]
        )
        raw = response.text.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw).get("mapped_value")
    except Exception as e:
        print(f"[WARN] Gemini call failed for '{value}': {e}")
        return None


def build_value_mapping(unique_values: list[str]) -> dict[str, str]:
    """Call Gemini once per unique value and return a {raw_value: mapped_value} dict."""
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in your .env file.")

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    definitions_text = _build_definitions_text()

    mapping = {}
    for value in tqdm(unique_values, desc="Mapping values", colour="cyan"):
        result = _call_gemini(model, value, definitions_text)
        if result and result in VALUEBENCH_DEFINITIONS:
            mapping[value] = result
        else:
            print(f"[WARN] Could not map '{value}' → got '{result}'. Keeping original.")
            mapping[value] = value

    return mapping


def run(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    unique_values = df["value"].dropna().unique().tolist()
    print(f"Found {len(unique_values)} unique values to map.")

    mapping = build_value_mapping(unique_values)

    print("\nMapping result:")
    for raw, mapped in mapping.items():
        print(f"  {raw!r:45s} → {mapped!r}")

    df[MAPPED_VALUE_COL] = df["value"].map(mapping)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows → {output_csv}")
    return df


if __name__ == "__main__":
    input_csv = config.INPUT_CSV.parent / "value_bench.csv"
    output_csv = config.INPUT_CSV.parent / "value_bench_mapped.csv"
    run(input_csv, output_csv)
