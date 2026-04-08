import os
import sys
import warnings
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

import config as root_config
from utils import parse_json
import value_stability.config as vs_config
from value_stability.prompts import PERTURBATION_SYSTEM_PROMPT, PERTURBATION_TYPES

warnings.filterwarnings("ignore")


def _hf_login() -> None:
    if not root_config.HF_TOKEN:
        return
    os.environ.setdefault("HF_TOKEN", root_config.HF_TOKEN)
    try:
        from huggingface_hub import login
        login(token=root_config.HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass


_hf_login()


class PerturbationPipeline:
    def __init__(
        self,
        model_id: str = vs_config.MODEL_ID,
        max_new_tokens: int = vs_config.MAX_NEW_TOKENS,
        device_map: str = vs_config.DEVICE_MAP,
    ):
        self.max_new_tokens = max_new_tokens
        print(f"Loading model: {model_id}")
        
        self.pipe = hf_pipeline(
            "text-generation",
            model=model_id,
            device_map=device_map,
            dtype=torch.bfloat16,
            model_kwargs={
                "attn_implementation": "sdpa",
            },
        )
        if self.pipe.tokenizer.pad_token_id is None:
            self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
        print(f"pipe device: {self.pipe.device}")

    def _generate(self, messages: list[dict]) -> str:
        outputs = self.pipe(
            text=messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()

    def _build_messages(self, row: pd.Series, perturbation_type: str) -> list[dict]:
        cfg = PERTURBATION_TYPES[perturbation_type]
        user = cfg["user_prompt"].format(
            value=row["value"],
            question=row["question"],
            positive_answer=row["positive_answer"],
            negative_answer=row["negative_answer"],
        )
        return [
            {"role": "system", "content": PERTURBATION_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _load_pending(
        self, input_csv: Path, output_csv: Path, output_col: str
    ) -> tuple[pd.DataFrame, list[int]]:
        """Return df with output_col added, and indices of rows still missing that column."""
        df = pd.read_csv(input_csv)
        if output_col not in df.columns:
            df[output_col] = None
        df[output_col] = df[output_col].astype(object)

        if output_csv.exists():
            saved = pd.read_csv(output_csv)
            if output_col in saved.columns:
                df[output_col] = saved[output_col]

        pending = df.index[df[output_col].isnull()].tolist()
        return df, pending

    def run_single_perturbation(
        self,
        perturbation_type: str,
        input_csv: str | Path,
        output_csv: str | Path,
    ) -> pd.DataFrame:
        if perturbation_type not in PERTURBATION_TYPES:
            raise ValueError(f"Unknown perturbation type '{perturbation_type}'. Choose from {list(PERTURBATION_TYPES)}")

        input_csv, output_csv = Path(input_csv), Path(output_csv)
        output_col = PERTURBATION_TYPES[perturbation_type]["output_col"]
        df, pending_idx = self._load_pending(input_csv, output_csv, output_col)

        print(f"[{perturbation_type}] Rows to process: {len(pending_idx)} → {output_csv.name}")

        for idx in tqdm(pending_idx, desc=perturbation_type, colour="green"):
            row = df.loc[idx]
            try:
                raw = self._generate(self._build_messages(row, perturbation_type))
                parsed = parse_json(raw, output_col)
                df.at[idx, output_col] = parsed if parsed is not None else raw
            except Exception as e:
                tqdm.write(f"[WARN] Row {idx} failed: {e}")
                df.at[idx, output_col] = f"ERROR: {e}"

            df.to_csv(output_csv, index=False)

        print(f"[{perturbation_type}] Done → {output_csv}")
        return df

    def build_perturbation_dataset(
        self,
        input_csv: str | Path,
        paraphrase_output_csv: str | Path,
        adversarial_output_csv: str | Path,
    ) -> None:
        self.run_single_perturbation("paraphrase",  input_csv, paraphrase_output_csv)
        self.run_single_perturbation("adversarial", input_csv, adversarial_output_csv)


if __name__ == "__main__":
    if not vs_config.DEBUG_INPUT_CSV.exists():
        pd.read_csv(vs_config.INPUT_CSV).head(vs_config.DEBUG_ROWS).to_csv(
            vs_config.DEBUG_INPUT_CSV, index=False
        )
        print(f"Created debug sample → {vs_config.DEBUG_INPUT_CSV}")

    pipe = PerturbationPipeline()
    pipe.build_perturbation_dataset(
        input_csv=vs_config.DEBUG_INPUT_CSV,
        paraphrase_output_csv=vs_config.DEBUG_PARAPHRASE_OUTPUT_CSV,
        adversarial_output_csv=vs_config.DEBUG_ADVERSARIAL_OUTPUT_CSV,
    )
