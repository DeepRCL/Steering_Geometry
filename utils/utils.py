import json
import re
from pathlib import Path

import pandas as pd


def parse_json(raw: str, key: str) -> str | None:
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    clean = re.sub(r"</?think>", "", clean).strip()
    clean = re.sub(r"```(?:json)?", "", clean).replace("```", "").strip()

    start = clean.find("{")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(clean[start:], start):
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
                        return json.loads(clean[start:i + 1]).get(key)
                    except json.JSONDecodeError:
                        break

    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)
    return m.group(1).strip() if m else None


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
