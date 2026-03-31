import os
import sys
import warnings
from pathlib import Path
from transformers import pipeline
import json
import re
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

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
            "text-generation",
            model=model_id,
            device_map=device_map,
            dtype="auto",
            # When run on server, use flash_attention_2
            # model_kwargs={"attn_implementation": "flash_attention_2"}
            model_kwargs={"attn_implementation": "sdpa"}
        )
        if self.pipe.tokenizer.pad_token_id is None:
            self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id

    def _generate(self, messages: list) -> str:
        outputs = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            max_length=None,
            do_sample=False,
            return_full_text=False,
        )
        generated = outputs[0]["generated_text"].strip()
        return generated

    def _generate_batch(self, batch_messages: list, batch_size: int) -> list[str]:
        outputs = self.pipe(
            batch_messages,
            batch_size=batch_size,
            max_new_tokens=self.max_new_tokens,
            max_length=None,
            do_sample=False,
            return_full_text=False,
        )
        return [out[0]["generated_text"].strip() for out in outputs]

    def _parse_json(self, raw: str, key: str) -> str | None:
        clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
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


    def _build_messages(self, row, mode: str = "negative"):
        question = row["question"]
        value    = row["value"]

        if mode == "negative":
            system_prompt = VALUEBENCH_POSITIVE_SYSTEM
            user_prompt   = VALUEBENCH_POSITIVE_USER.format(
                examples=EXAMPLES_POSITIVE, question=question,
                value=value, provided_answer=row["positive_answer"],
            )
        else:
            system_prompt = VALUEBENCH_NEGATIVE_SYSTEM
            user_prompt   = VALUEBENCH_NEGATIVE_USER.format(
                examples=EXAMPLES_NEGATIVE, question=question,
                value=value, provided_answer=row["negative_answer"],
            )

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
        ]


    def _create_answer(self, row, mode: str = "negative") -> str:
        json_key = "negative_answer" if mode == "negative" else "positive_answer"
        raw = self._generate(self._build_messages(row, mode))
        parsed = self._parse_json(raw, json_key)
        return parsed if parsed is not None else raw

    def _load_pending_rows(
        self,
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

    def build_dataset_single(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        target_col: str = "negative_answer",
        mode: str = "negative",
        batch_size: int = 10,
    ) -> pd.DataFrame:
        input_csv, output_csv = Path(input_csv), Path(output_csv)
        df, pending_idx = self._load_pending_rows(input_csv, output_csv, target_col)
        total_batches = (len(pending_idx) + batch_size - 1) // batch_size

        for batch_start in range(0, len(pending_idx), batch_size):
            batch_indices = pending_idx[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            for idx in tqdm(batch_indices, desc=f"Batch {batch_num}/{total_batches}", colour="green", leave=True):
                try:
                    df.at[idx, target_col] = self._create_answer(df.loc[idx], mode=mode)
                except Exception as e:
                    tqdm.write(f"[WARN] Row {idx} failed: {e}")
                    df.at[idx, target_col] = f"ERROR: {e}"
            df.to_csv(output_csv, index=False)
            tqdm.write(f"  ✓ Batch {batch_num}/{total_batches} saved → {output_csv}")

        print(f"Done. Final dataset saved to {output_csv}")
        return df

    def build_dataset_batch(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        target_col: str = "negative_answer",
        mode: str = "negative",
        batch_size: int = 10,
    ) -> pd.DataFrame:
        input_csv, output_csv = Path(input_csv), Path(output_csv)
        df, pending_idx = self._load_pending_rows(input_csv, output_csv, target_col)
        json_key = "negative_answer" if mode == "negative" else "positive_answer"
        total_batches = (len(pending_idx) + batch_size - 1) // batch_size

        pbar = tqdm(range(0, len(pending_idx), batch_size), total=total_batches, colour="green")
        for batch_start in pbar:
            batch_indices = pending_idx[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            pbar.set_description(f"Batch {batch_num}/{total_batches}")
            batch_messages = [self._build_messages(df.loc[idx], mode=mode) for idx in batch_indices]
            try:
                for idx, raw in zip(batch_indices, self._generate_batch(batch_messages, len(batch_indices))):
                    parsed = self._parse_json(raw, json_key)
                    df.at[idx, target_col] = parsed if parsed is not None else raw
            except Exception as e:
                tqdm.write(f"[WARN] Batch {batch_num} failed: {e}")
                for idx in batch_indices:
                    df.at[idx, target_col] = f"ERROR: {e}"
            df.to_csv(output_csv, index=False)

        print(f"Done. Final dataset saved to {output_csv}")
        return df


if __name__ == "__main__":

    TARGET_COL  = "negative_answer"
    DEBUG_INPUT  = config.INPUT_CSV.parent / "debug_input.csv"
    DEBUG_OUTPUT = config.INPUT_CSV.parent / "debug_output.csv"

    # Create debug_input.csv only once; reuse on subsequent runs
    if not DEBUG_INPUT.exists():
        print("Loading CSV and creating debug sample...")
        df = pd.read_csv(config.INPUT_CSV)
        sample = df.head(config.DEBUG_ROWS).copy()
        sample[TARGET_COL] = sample[TARGET_COL].astype(object)
        sample.to_csv(DEBUG_INPUT, index=False)
        print(f"Saved {config.DEBUG_ROWS}-row sample → {DEBUG_INPUT}")
    else:
        print(f"Reusing existing debug sample → {DEBUG_INPUT}")

    print(f"Initializing model: {config.MODEL_ID}")
    pipe = DatasetConstructionPipeline()
    pipe.build_dataset_batch(
        input_csv=DEBUG_INPUT,
        output_csv=DEBUG_OUTPUT,
        target_col=TARGET_COL,
        mode="negative",
        batch_size=config.BATCH_SIZE,
    )
