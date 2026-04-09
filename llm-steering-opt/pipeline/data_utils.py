"""
Data loading, stratified splitting, and TrainingDatapoint creation.
"""

import csv
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import steering_opt


def load_dataset(path: str) -> List[dict]:
    """Load the CSV dataset and return a list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_unique_values(rows: List[dict]) -> List[str]:
    """Return sorted list of unique Schwartz values in the dataset."""
    return sorted(set(row["value"] for row in rows))


def stratified_split(
    rows: List[dict],
    train_ratio: float = 0.6,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    Split dataset so that each value has exactly `train_ratio` proportion
    in the training set and (1 - train_ratio) in the validation set.

    Args:
        rows: Full dataset as list of dicts.
        train_ratio: Fraction of each value's examples to put in train.
        seed: Random seed for reproducibility.

    Returns:
        (train_rows, val_rows)
    """
    rng = random.Random(seed)

    # Group rows by value
    by_value: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_value[row["value"]].append(row)

    train_rows, val_rows = [], []
    for value in sorted(by_value.keys()):
        group = by_value[value]
        rng.shuffle(group)
        n_train = max(1, int(len(group) * train_ratio))
        # Ensure at least 1 example in val
        if n_train >= len(group):
            n_train = len(group) - 1
        train_rows.extend(group[:n_train])
        val_rows.extend(group[n_train:])

    return train_rows, val_rows


def get_rows_for_value(rows: List[dict], value: str) -> List[dict]:
    """Filter rows to only those matching the given value."""
    return [r for r in rows if r["value"] == value]


def format_prompt(
    question: str,
    tokenizer=None,
    use_chat_template: bool = True,
    prompt_template: str = "",
) -> str:
    """
    Format a question into a model prompt.

    If use_chat_template is True and the tokenizer supports it, uses the
    tokenizer's chat template. Otherwise falls back to prompt_template.
    """
    if use_chat_template and tokenizer is not None:
        try:
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return prompt
        except Exception:
            pass  # Fall through to template

    return prompt_template.format(question=question)


def create_datapoints(
    rows: List[dict],
    tokenizer=None,
    use_chat_template: bool = True,
    prompt_template: str = "",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[steering_opt.TrainingDatapoint]:
    """
    Convert dataset rows into TrainingDatapoint objects for steering_opt.

    Each row becomes a datapoint where:
        - prompt = formatted question
        - dst_completions = [positive_answer]  (promote)
        - src_completions = [negative_answer]  (suppress)

    Args:
        rows: Dataset rows to convert.
        tokenizer: Model tokenizer for chat template formatting.
        use_chat_template: Whether to use tokenizer's chat template.
        prompt_template: Fallback template string with {question} placeholder.
        n_samples: If set, randomly sample this many rows. None = use all.
        seed: Random seed for sampling.

    Returns:
        List of TrainingDatapoint objects.
    """
    if n_samples is not None and n_samples < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, n_samples)

    datapoints = []
    for row in rows:
        prompt = format_prompt(
            row["question"], tokenizer, use_chat_template, prompt_template
        )
        dp = steering_opt.TrainingDatapoint(
            prompt=prompt,
            src_completions=[row["negative_answer"]],
            dst_completions=[row["positive_answer"]],
        )
        datapoints.append(dp)

    return datapoints


def print_split_summary(
    train_rows: List[dict], val_rows: List[dict], values: List[str]
):
    """Print a summary of the train/val split per value."""
    print(f"\n{'Value':<35} {'Train':>6} {'Val':>6} {'Total':>6} {'Train%':>7}")
    print("-" * 65)
    for v in values:
        n_train = sum(1 for r in train_rows if r["value"] == v)
        n_val = sum(1 for r in val_rows if r["value"] == v)
        total = n_train + n_val
        pct = 100 * n_train / total if total > 0 else 0
        print(f"  {v:<33} {n_train:>6} {n_val:>6} {total:>6} {pct:>6.1f}%")
    print("-" * 65)
    print(
        f"  {'TOTAL':<33} {len(train_rows):>6} {len(val_rows):>6} "
        f"{len(train_rows)+len(val_rows):>6} "
        f"{100*len(train_rows)/(len(train_rows)+len(val_rows)):.1f}%"
    )
    print()
