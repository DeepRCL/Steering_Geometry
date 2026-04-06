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
        lines.append(f"- {category} — {defn['overview']} Sub-values: {sub_parts}.")
    return "\n".join(lines)


def _parse_response(raw: str) -> str | None:
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw).get("mapped_value")


class KeyRotatingClient:

    def __init__(self, api_keys: list[str]) -> None:
        if not api_keys:
            raise ValueError("No GEMINI_API_KEYS set in your .env file.")
        self._keys = api_keys
        self._idx = 0
        self._client = genai.Client(api_key=self._keys[0])
        tqdm.write(f"[KEY] Using key index 0 ({self._masked()}).")

    def _masked(self) -> str:
        key = self._keys[self._idx]
        return f"{key[:6]}...{key[-4:]}"

    def _rotate(self) -> bool:
        next_idx = self._idx + 1
        if next_idx >= len(self._keys):
            return False
        self._idx = next_idx
        self._client = genai.Client(api_key=self._keys[self._idx])
        tqdm.write(f"[KEY] Rotated to key index {self._idx} ({self._masked()}).")
        return True

    def generate(self, prompt: str) -> str | None:
        while True:
            try:
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL, contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err.upper():
                    tqdm.write(f"[WARN] Rate limit hit on key index {self._idx}.")
                    if not self._rotate():
                        tqdm.write("[ERROR] All API keys exhausted.")
                        raise
                else:
                    raise


def _call_gemini(
    client: KeyRotatingClient,
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
    raw = client.generate(prompt) 
    return _parse_response(raw)
    

def _append_rows(rows: pd.DataFrame, output_csv: Path) -> None:
    write_header = not output_csv.exists()
    rows.to_csv(output_csv, mode="a", header=write_header, index=False)


def _resolve_mapped(value: str, result: str | None) -> str:
    if result in VALID_CATEGORIES or result == "NA":
        return result
    tqdm.write(f"[WARN] '{value}' → '{result}' not in canonical list, saving as NA.")
    return "NA"


def run_by_value(input_csv: Path, output_csv: Path) -> None:
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

    client = KeyRotatingClient(config.GEMINI_API_KEYS)
    definitions_text = _build_definitions_text()

    representatives = (
        df_needs[["value", "question", "positive_answer", "negative_answer"]]
        .drop_duplicates(subset=["value"])
        .set_index("value")
    )

    for value in tqdm(unique_remaining, desc="Mapping values", colour="cyan"):
        rep = representatives.loc[value]
        try:
            result = _call_gemini(
                client,
                value=value,
                question=rep["question"],
                positive_answer=rep["positive_answer"],
                negative_answer=rep["negative_answer"],
                definitions_text=definitions_text,
            )
        except Exception as e:
            tqdm.write(f"[ERROR] API call failed for '{value}', skipping (will retry on next run): {e}")
            continue

        mapped = _resolve_mapped(value, result)
        rows = df_needs[df_needs["value"] == value].copy()
        rows[MAPPED_VALUE_COL] = mapped
        _append_rows(rows, output_csv)
        tqdm.write(f"  {len(rows)} row(s): '{value}' → '{mapped}'")

    total = len(pd.read_csv(output_csv))
    print(f"\nDone. {total} rows saved → {output_csv}")


def run_by_row(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)

    # ── Phase 1: rows already matching a canonical Schwartz value ────────────
    mask_valid = df["value"].isin(VALID_CATEGORIES)
    df_valid = df[mask_valid].copy()
    df_needs = df[~mask_valid].copy()

    print(f"Phase 1 — {len(df_valid)} rows already have canonical values → saving immediately.")
    if len(df_valid):
        df_valid[MAPPED_VALUE_COL] = df_valid["value"]
        _append_rows(df_valid, output_csv)

    # ── Phase 2: map each row individually via Gemini ────────────────────────
    print(f"Phase 2 — {len(df_needs)} rows need mapping (row-by-row mode).")

    already_done = 0
    if output_csv.exists():
        already_done = len(pd.read_csv(output_csv))
        df_needs = df_needs.iloc[max(0, already_done - len(df_valid)):]
        print(f"  Resuming: {len(df_needs)} rows still to process.")

    if df_needs.empty:
        print("Nothing left to map.")
        return

    client = KeyRotatingClient(config.GEMINI_API_KEYS)
    definitions_text = _build_definitions_text()

    for _, row in tqdm(df_needs.iterrows(), total=len(df_needs), desc="Mapping rows", colour="cyan"):
        try:
            result = _call_gemini(
                client,
                value=row["value"],
                question=row["question"],
                positive_answer=row["positive_answer"],
                negative_answer=row["negative_answer"],
                definitions_text=definitions_text,
            )
        except Exception as e:
            tqdm.write(f"[ERROR] API call failed for row (value='{row['value']}'), skipping: {e}")
            continue

        mapped = _resolve_mapped(row["value"], result)
        out_row = row.to_frame().T.copy()
        out_row[MAPPED_VALUE_COL] = mapped
        _append_rows(out_row, output_csv)
        tqdm.write(f"  '{row['value']}' → '{mapped}'")

    total = len(pd.read_csv(output_csv))
    print(f"\nDone. {total} rows saved → {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run on a 50-row sample")
    parser.add_argument(
        "--mode",
        choices=["by_value", "by_row"],
        default="by_row",
        help="by_value: one API call per unique value (default); by_row: one call per row",
    )
    args = parser.parse_args()

    _DATA = Path("/Users/hamidrezaei/Workspace/Steering_Geometry/dataset_construction/data")
    input_csv = _DATA / "dataset_negative_answer.csv"

    runner = run_by_row if args.mode == "by_row" else run_by_value

    if args.debug:
        import time
        debug_dir = _DATA / "debug"
        debug_dir.mkdir(exist_ok=True)
        debug_input = debug_dir / f"negative_answer_input_{round(time.time())}.csv"
        if not debug_input.exists():
            pd.read_csv(input_csv).iloc[400:450].to_csv(debug_input, index=False)
        suffix = "row" if args.mode == "by_row" else "value"
        output_csv = debug_dir / f"negative_answer_mapped_{suffix}_{round(time.time())}.csv"
        print(f"=== DEBUG MODE ({args.mode}) ===")
        runner(debug_input, output_csv)
    else:
        output_csv = _DATA / "dataset_negative_answer_mapped.csv"
        runner(input_csv, output_csv)
