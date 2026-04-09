"""
Data loading for the SparseCAA pipeline.

Combines two sources:
  1. base_dataset_path  (final_dataset_200.csv) — all 20 values, ~181–276 per value
  2. touche_dataset_path (touche_gemma4-v2_remaining-validated-v3.csv)
     filtered to caa_suitable=True, up to touche_samples_per_value per value

Result: ~231–326 samples per value (all 20 values covered).

The combined dataset is split per-value into training and evaluation sets
(90/10 by default).  If equal_samples_per_value is True, all values are capped
at the minimum per-value count so every value contributes exactly the same
number of samples — useful when downstream steps need stacked tensors of equal
depth.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SparseCAAPipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ContrastivePair:
    sample_id: str
    value: str
    question: str
    positive_answer: str
    negative_answer: str
    # Which MC option letter (A or B) holds the positive answer
    pos_is_a: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading & merging
# ──────────────────────────────────────────────────────────────────────────────
def _load_base(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"id": "sample_id"})
    return df[["sample_id", "question", "value", "positive_answer", "negative_answer"]]


def _load_touche(path: str, max_per_value: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["caa_suitable"] == True].copy()
    df = df.rename(columns={"argument_id": "sample_id"})
    df = df[["sample_id", "question", "value", "positive_answer", "negative_answer"]]

    # Take up to max_per_value rows per value (keep first N after filtering)
    parts = []
    for _, grp in df.groupby("value"):
        parts.append(grp.head(max_per_value))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def load_combined(config: SparseCAAPipelineConfig) -> pd.DataFrame:
    """Return the merged dataframe (base + Touche supplement), deduplicated."""
    base = _load_base(config.base_dataset_path)
    touche = _load_touche(config.touche_dataset_path, config.touche_samples_per_value)

    combined = pd.concat([base, touche], ignore_index=True)
    # Deduplicate by sample_id (base takes precedence, being first)
    combined = combined.drop_duplicates(subset=["sample_id"], keep="first")
    # Keep only recognised Schwartz values
    combined = combined[combined["value"].isin(SCHWARTZ_CIRCUMPLEX_ORDER)]
    combined = combined.reset_index(drop=True)
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Train / eval split
# ──────────────────────────────────────────────────────────────────────────────
def split_dataset(
    df: pd.DataFrame,
    config: SparseCAAPipelineConfig,
) -> Tuple[Dict[str, List[ContrastivePair]], Dict[str, List[ContrastivePair]]]:
    """
    Per-value 90/10 (train/eval) stratified split.

    If config.equal_samples_per_value is True, caps every value at the
    minimum per-value count before splitting, ensuring perfectly balanced
    training batches.

    Returns:
        train_data : {value -> list of ContrastivePair}
        eval_data  : {value -> list of ContrastivePair}
    """
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    grouped: Dict[str, List[ContrastivePair]] = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        rows = df[df["value"] == val]
        pairs = [
            ContrastivePair(
                sample_id=str(row["sample_id"]),
                value=val,
                question=row["question"],
                positive_answer=row["positive_answer"],
                negative_answer=row["negative_answer"],
            )
            for _, row in rows.iterrows()
        ]
        grouped[val] = pairs

    if config.equal_samples_per_value:
        min_count = min(len(p) for p in grouped.values() if p)
        for val in grouped:
            grouped[val] = grouped[val][:min_count]

    train_data: Dict[str, List[ContrastivePair]] = {}
    eval_data: Dict[str, List[ContrastivePair]] = {}

    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        pairs = list(grouped[val])
        rng.shuffle(pairs)
        n_eval = max(1, int(len(pairs) * config.eval_split))
        eval_pairs = pairs[:n_eval]
        train_pairs = pairs[n_eval:]

        # Assign random A/B polarity to each training pair
        for p in train_pairs:
            p.pos_is_a = bool(np_rng.integers(0, 2))
        # Same for eval
        for p in eval_pairs:
            p.pos_is_a = bool(np_rng.integers(0, 2))

        train_data[val] = train_pairs
        eval_data[val] = eval_pairs

    return train_data, eval_data


# ──────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ──────────────────────────────────────────────────────────────────────────────
def format_prompts(
    pair: ContrastivePair,
    tokenizer,
    is_instruct: bool,
) -> Tuple[List[int], List[int]]:
    """
    Format a contrastive pair as a multiple-choice prompt.

    Returns (pos_tokens, neg_tokens) — each a list of token IDs ending with
    the answer letter that corresponds to the positive / negative answer.
    This matches the format used throughout CAA/Geometry and SAE/extract_mlp_vectors.py.
    """
    if pair.pos_is_a:
        a_text, b_text = pair.positive_answer, pair.negative_answer
        pos_letter, neg_letter = "A", "B"
    else:
        a_text, b_text = pair.negative_answer, pair.positive_answer
        pos_letter, neg_letter = "B", "A"

    body = (
        f"Question: {pair.question}\n"
        f"(A) {a_text}\n"
        f"(B) {b_text}"
    )

    if is_instruct:
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        base = body + "\nResponse:"

    pos_prompt = base + f" ({pos_letter}"
    neg_prompt = base + f" ({neg_letter}"

    pos_tokens = tokenizer.encode(pos_prompt, add_special_tokens=True)
    neg_tokens = tokenizer.encode(neg_prompt, add_special_tokens=True)
    return pos_tokens, neg_tokens


def format_eval_prompt(
    pair: ContrastivePair,
    tokenizer,
    is_instruct: bool,
) -> Tuple[List[int], int, int]:
    """
    Format a pair for A/B logit evaluation (no answer appended).

    Returns (tokens, a_token_id, b_token_id).
    """
    if pair.pos_is_a:
        a_text, b_text = pair.positive_answer, pair.negative_answer
    else:
        a_text, b_text = pair.negative_answer, pair.positive_answer

    body = (
        f"Question: {pair.question}\n"
        f"(A) {a_text}\n"
        f"(B) {b_text}"
    )

    if is_instruct:
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        base = body + "\nResponse:"

    eval_prompt = base + " ("
    tokens = tokenizer.encode(eval_prompt, add_special_tokens=True)

    a_token_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    b_token_id = tokenizer.encode("B", add_special_tokens=False)[-1]
    return tokens, a_token_id, b_token_id


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: print dataset summary
# ──────────────────────────────────────────────────────────────────────────────
def print_dataset_summary(
    train_data: Dict[str, List[ContrastivePair]],
    eval_data: Dict[str, List[ContrastivePair]],
) -> None:
    print(f"{'Value':<40} {'Train':>6} {'Eval':>6} {'Total':>6}")
    print("-" * 60)
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        n_train = len(train_data.get(val, []))
        n_eval = len(eval_data.get(val, []))
        print(f"{val:<40} {n_train:>6} {n_eval:>6} {n_train + n_eval:>6}")
    total_train = sum(len(v) for v in train_data.values())
    total_eval = sum(len(v) for v in eval_data.values())
    print("-" * 60)
    print(f"{'TOTAL':<40} {total_train:>6} {total_eval:>6} {total_train + total_eval:>6}")
