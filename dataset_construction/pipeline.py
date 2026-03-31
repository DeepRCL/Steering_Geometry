import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

import config
from utils import parse_json, load_pending_rows

warnings.filterwarnings("ignore")
from prompt import (
    VALUEBENCH_POSITIVE_SYSTEM,
    VALUEBENCH_POSITIVE_USER,
    VALUEBENCH_NEGATIVE_SYSTEM,
    VALUEBENCH_NEGATIVE_USER,
    EXAMPLES_POSITIVE,
    EXAMPLES_NEGATIVE,
)


def _hf_login() -> None:
    if not config.HF_TOKEN:
        return
    os.environ.setdefault("HF_TOKEN", config.HF_TOKEN)
    try:
        from huggingface_hub import login
        login(token=config.HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass


_hf_login()


class DatasetConstructionPipeline:

    def __init__(
        self,
        model_id: str = config.MODEL_ID,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        device_map: str = config.DEVICE_MAP,
    ):
        self.max_new_tokens = max_new_tokens
        print(f"Loading model: {model_id}")
        self.pipe = hf_pipeline(
            "text-generation",
            model=model_id,
            device_map=device_map,
            dtype="auto",
            # model_kwargs={"attn_implementation": "flash_attention_2"}  # GPU server
            model_kwargs={"attn_implementation": "sdpa"},
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
        return outputs[0]["generated_text"].strip()

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

    def _build_messages(self, row, mode: str = "negative") -> list:
        question, value = row["question"], row["value"]

        if mode == "negative":
            system = VALUEBENCH_POSITIVE_SYSTEM
            user = VALUEBENCH_POSITIVE_USER.format(
                examples=EXAMPLES_POSITIVE,
                question=question,
                value=value,
                provided_answer=row["positive_answer"],
            )
        else:
            system = VALUEBENCH_NEGATIVE_SYSTEM
            user = VALUEBENCH_NEGATIVE_USER.format(
                examples=EXAMPLES_NEGATIVE,
                question=question,
                value=value,
                provided_answer=row["negative_answer"],
            )

        return [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]

    def build_dataset_single(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        target_col: str = "negative_answer",
        mode: str = "negative",
        batch_size: int = 10,
    ) -> pd.DataFrame:
        input_csv, output_csv = Path(input_csv), Path(output_csv)
        df, pending_idx = load_pending_rows(input_csv, output_csv, target_col)
        json_key = "negative_answer" if mode == "negative" else "positive_answer"
        total_batches = (len(pending_idx) + batch_size - 1) // batch_size

        for batch_start in range(0, len(pending_idx), batch_size):
            batch_indices = pending_idx[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            for idx in tqdm(batch_indices, desc=f"Batch {batch_num}/{total_batches}", colour="green", leave=True):
                try:
                    raw = self._generate(self._build_messages(df.loc[idx], mode))
                    parsed = parse_json(raw, json_key)
                    df.at[idx, target_col] = parsed if parsed is not None else raw
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
        df, pending_idx = load_pending_rows(input_csv, output_csv, target_col)
        json_key = "negative_answer" if mode == "negative" else "positive_answer"
        total_batches = (len(pending_idx) + batch_size - 1) // batch_size

        pbar = tqdm(range(0, len(pending_idx), batch_size), total=total_batches, colour="green")
        for batch_start in pbar:
            batch_indices = pending_idx[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            pbar.set_description(f"Batch {batch_num}/{total_batches}")
            batch_messages = [self._build_messages(df.loc[idx], mode=mode) for idx in batch_indices]
            try:
                for idx, raw in zip(batch_indices, self._generate_batch(batch_messages, len(batch_indices))):
                    parsed = parse_json(raw, json_key)
                    df.at[idx, target_col] = parsed if parsed is not None else raw
            except Exception as e:
                tqdm.write(f"[WARN] Batch {batch_num} failed: {e}")
                for idx in batch_indices:
                    df.at[idx, target_col] = f"ERROR: {e}"
            df.to_csv(output_csv, index=False)

        print(f"Done. Final dataset saved to {output_csv}")
        return df


if __name__ == "__main__":
    TARGET_COL = "negative_answer"
    DEBUG_INPUT = config.INPUT_CSV.parent / "debug_input.csv"
    DEBUG_OUTPUT = config.INPUT_CSV.parent / "debug_output.csv"

    if not DEBUG_INPUT.exists():
        print("Loading CSV and creating debug sample...")
        df_full = pd.read_csv(config.INPUT_CSV)
        sample = df_full.head(config.DEBUG_ROWS).copy()
        sample[TARGET_COL] = sample[TARGET_COL].astype(object)
        sample.to_csv(DEBUG_INPUT, index=False)
        print(f"Saved {config.DEBUG_ROWS}-row sample → {DEBUG_INPUT}")
    else:
        print(f"Reusing existing debug sample → {DEBUG_INPUT}")

    pipe = DatasetConstructionPipeline()
    pipe.build_dataset_single(
        input_csv=DEBUG_INPUT,
        output_csv=DEBUG_OUTPUT,
        target_col=TARGET_COL,
        mode="negative",
        batch_size=config.BATCH_SIZE,
    )
