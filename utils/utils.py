import json
import re
from pathlib import Path

import pandas as pd


def _extract_jsons(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    results = []
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        results.append(json.loads(text[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    pos = i + 1
                    break
        else:
            break
    return results


def _first_value(d: dict) -> str | None:
    for v in d.values():
        return v
    return None


def parse_json(raw: str, key: str) -> str | None:
    if "</think>" in raw:
        before, after = raw.split("</think>", 1)
    else:
        before, after = "", raw

    after_jsons = _extract_jsons(after)
    if after_jsons:
        obj = after_jsons[0]
        val = obj.get(key, _first_value(obj))
        if val is not None:
            return val

    before_jsons = _extract_jsons(before)
    if before_jsons:
        obj = before_jsons[-1]
        val = obj.get(key, _first_value(obj))
        if val is not None:
            return val

    return "..."


def load_pending_rows(
    input_csv: Path,
    output_csv: Path,
    target_col: str,
) -> tuple[pd.DataFrame, list[int]]:
    if output_csv.exists():
        df = pd.read_csv(output_csv)
        print(f"Resuming from existing output: {output_csv} ({len(df)} rows)")
    else:
        df = pd.read_csv(input_csv)
        if target_col not in df.columns:
            df[target_col] = ""
        print(f"Starting fresh from: {input_csv} ({len(df)} rows)")

    df[target_col] = df[target_col].astype(object)
    pending_idx = df.index[
        df[target_col].isna() | (df[target_col].astype(str).str.strip() == "")
    ].tolist()
    print(f"Rows to process: {len(pending_idx)}")
    return df, pending_idx
