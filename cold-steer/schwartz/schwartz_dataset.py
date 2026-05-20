"""Per-value Dataset wrapper that yields cold-steer-compatible tensors.

The keys we expose match what ``src/steerer.py::LossFDSteerer.train`` reads
(``matching_input_ids``, ``not_matching_input_ids``, ``matching_labels``,
etc.). For each row we tokenize:

    prompt:    Schwartz question (formatted via tokenizer chat template
               when available, otherwise via ``prompt_template``)
    matching:  prompt + " " + positive_answer
    not_match: prompt + " " + negative_answer

``matching_labels`` mask out the prompt portion (set to ``-100``) so the
NLL loss is computed only over the answer tokens — matching cold-steer's
SFT contract.
"""

from typing import List, Optional

import torch
from torch.utils.data import Dataset

from . import data_utils


class SchwartzValueDataset(Dataset):
    """Single-value training subset built from Schwartz CSV rows.

    Args:
        rows: rows for one Schwartz value (already filtered).
        tokenizer: HuggingFace tokenizer used by cold-steer's SteerableLLM.
        device: target device for the tokenized tensors.
        use_chat_template: whether to use the tokenizer's chat template
            for the prompt. Falls back to ``prompt_template`` otherwise.
        prompt_template: fallback template with a ``{question}`` placeholder.
        n_samples: cap on the number of rows used; ``None`` = use all.
        seed: deterministic sub-sampling seed.
    """

    def __init__(
        self,
        rows: List[dict],
        tokenizer,
        device: str = "cuda:0",
        use_chat_template: bool = True,
        prompt_template: str = "",
        n_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.device = device
        self.rows = data_utils.sample_rows(rows, n_samples, seed)

        if not self.rows:
            self.inputs = {}
            self.labels = {}
            return

        prompts = [
            data_utils.format_prompt(
                r["question"], tokenizer, use_chat_template, prompt_template
            )
            for r in self.rows
        ]
        # We tokenize prompts on their own to know where each answer starts.
        prompt_tok = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        prompt_input_ids = prompt_tok["input_ids"].to(device)
        prompt_attention_mask = prompt_tok["attention_mask"].to(device)

        matching_texts = [
            f"{p} {r['positive_answer']}" for p, r in zip(prompts, self.rows)
        ]
        not_matching_texts = [
            f"{p} {r['negative_answer']}" for p, r in zip(prompts, self.rows)
        ]

        match_tok = tokenizer(
            matching_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        not_match_tok = tokenizer(
            not_matching_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        matching_input_ids = match_tok["input_ids"].to(device)
        matching_attention_mask = match_tok["attention_mask"].to(device)
        not_matching_input_ids = not_match_tok["input_ids"].to(device)
        not_matching_attention_mask = not_match_tok["attention_mask"].to(device)

        prompt_len = prompt_input_ids.shape[1]
        matching_labels = matching_input_ids.clone()
        matching_labels[:, :prompt_len] = -100
        matching_labels[matching_attention_mask == 0] = -100

        not_matching_labels = not_matching_input_ids.clone()
        not_matching_labels[:, :prompt_len] = -100
        not_matching_labels[not_matching_attention_mask == 0] = -100

        self.inputs = {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "matching_input_ids": matching_input_ids,
            "matching_attention_mask": matching_attention_mask,
            "not_matching_input_ids": not_matching_input_ids,
            "not_matching_attention_mask": not_matching_attention_mask,
        }
        self.labels = {
            "matching_labels": matching_labels,
            "not_matching_labels": not_matching_labels,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in {**self.inputs, **self.labels}.items()}

    @property
    def prompt_input_ids(self) -> torch.Tensor:
        return self.inputs["prompt_input_ids"]

    @property
    def prompt_attention_mask(self) -> torch.Tensor:
        return self.inputs["prompt_attention_mask"]
