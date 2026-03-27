import os
import sys
from pathlib import Path
from transformers import pipeline
import json
import re
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from prompt import (
    VALUEBENCH_POSITIVE_SYSTEM,
    VALUEBENCH_POSITIVE_USER,
    VALUEBENCH_NEGATIVE_SYSTEM,
    VALUEBENCH_NEGATIVE_USER,
    EXAMPLES_POSITIVE,
    EXAMPLES_NEGATIVE,
    )


if config.HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", config.HF_TOKEN)
    try:
        from huggingface_hub import login
        login(token=config.HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass


class DatasetConstructionPipeline:

    def __init__(
        self,
        model_id:       str = config.MODEL_ID,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        device_map:     str = config.DEVICE_MAP,
    ):
        self.max_new_tokens = max_new_tokens
        print(f"Loading model: {model_id}")
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device_map=device_map,
            dtype="auto",
        )

    def _generate(self, messages):
        outputs = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        generated = outputs[0]["generated_text"].strip()
        return generated

    def _parse_json(self, raw: str, key: str) -> str | None:
        # strip markdown fences
        clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        # brace-counting extraction
        start = clean.find("{")
        if start != -1:
            depth, in_str, esc = 0, False, False
            for i, ch in enumerate(clean[start:], start):
                if esc:            esc = False;   continue
                if ch == "\\" and in_str: esc = True; continue
                if ch == '"':      in_str = not in_str; continue
                if in_str:         continue
                if ch == "{":      depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(clean[start : i + 1]).get(key)
                        except json.JSONDecodeError:
                            break
        # regex fallback
        m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)
        return m.group(1).strip() if m else None

    def create_answer(self, row, mode: str = "negative"):
        """
        mode='negative': given positive_answer → generate negative_answer
        mode='positive': given negative_answer → generate positive_answer
        """
        question = row["question"]
        value    = row["value"]

        if mode == "negative":
            system_prompt = VALUEBENCH_POSITIVE_SYSTEM
            user_prompt   = VALUEBENCH_POSITIVE_USER.format(
                examples=EXAMPLES_POSITIVE,
                question=question,
                value=value,
                provided_answer=row["positive_answer"],
            )
            json_key = "negative_answer"
        else:
            system_prompt = VALUEBENCH_NEGATIVE_SYSTEM
            user_prompt   = VALUEBENCH_NEGATIVE_USER.format(
                examples=EXAMPLES_NEGATIVE,
                question=question,
                value=value,
                provided_answer=row["negative_answer"],
            )
            json_key = "positive_answer"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
        ]
        raw = self._generate(messages)
        parsed = self._parse_json(raw, json_key)
        return parsed if parsed is not None else raw

    def build_dataset(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        target_col: str = "negative_answer",
        mode: str = "negative",
        batch_size: int = 10,
    ) -> pd.DataFrame:

        input_csv  = Path(input_csv)
        output_csv = Path(output_csv)

        # Load (or resume from) the output file
        if output_csv.exists():
            df = pd.read_csv(output_csv)
            print(f"Resuming from existing output: {output_csv} ({len(df)} rows)")
        else:
            df = pd.read_csv(input_csv)
            if target_col not in df.columns:
                df[target_col] = ""
            print(f"Starting fresh from: {input_csv} ({len(df)} rows)")

        df[target_col] = df[target_col].astype(object)

        pending_idx = df.index[df[target_col].isna() | (df[target_col].astype(str).str.strip() == "")].tolist()
        print(f"Rows to process: {len(pending_idx)}")

        for batch_start in range(0, len(pending_idx), batch_size):
            batch = pending_idx[batch_start : batch_start + batch_size]
            for idx in tqdm(batch, desc=f"Batch {batch_start // batch_size + 1}"):
                row = df.loc[idx]
                try:
                    df.at[idx, target_col] = self.create_answer(row, mode=mode)
                except Exception as e:
                    print(f"[WARN] Row {idx} failed: {e}")
                    df.at[idx, target_col] = f"ERROR: {e}"

            df.to_csv(output_csv, index=False)
            print(f"  Saved progress → {output_csv}")

        print(f"Done. Final dataset saved to {output_csv}")
        return df


