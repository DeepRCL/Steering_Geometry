"""Data loading and stratified splitting for the Schwartz dataset.

Mirrors ``llm-steering-opt/pipeline/data_utils.py`` so train/val splits and
prompt formatting match across methods.
"""

import csv
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def load_dataset(path: str) -> List[dict]:
    """Load the CSV dataset and return a list of row dicts."""
    print(path)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_unique_values(rows: List[dict]) -> List[str]:
    return sorted(set(row["value"] for row in rows))


def split_by_n_train(
    rows: List[dict],
    n_train: int,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Per-value split: up to ``n_train`` rows for training, the rest for validation.

    When a value has more than ``n_train`` rows, at least one row is kept in
    validation whenever possible.
    """
    rng = random.Random(seed)
    by_value: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_value[row["value"]].append(row)

    train_rows, val_rows = [], []
    for value in sorted(by_value.keys()):
        group = by_value[value]
        rng.shuffle(group)
        n = min(n_train, len(group))
        if n >= len(group) and len(group) > 1:
            n = len(group) - 1
        train_rows.extend(group[:n])
        val_rows.extend(group[n:])

    return train_rows, val_rows


def get_rows_for_value(rows: List[dict], value: str) -> List[dict]:
    return [r for r in rows if r["value"] == value]


def format_prompt(
    question: str,
    tokenizer=None,
    use_chat_template: bool = True,
    prompt_template: str = "",
) -> str:
    """Format a question into a model prompt.

    Falls back to ``prompt_template`` if the tokenizer has no chat template
    (e.g. Qwen-Base).
    """
    if use_chat_template and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            messages = [{"role": "user", "content": question}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return prompt_template.format(question=question)


def sample_rows(
    rows: List[dict],
    n: Optional[int],
    seed: int,
) -> List[dict]:
    """Deterministic subsample of rows; returns all if n is None or n>=len."""
    if n is None or n < 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, n)


def print_split_summary(
    train_rows: List[dict], val_rows: List[dict], values: List[str]
):
    print(f"\n{'Value':<35} {'Train':>6} {'Val':>6} {'Total':>6} {'Train%':>7}")
    print("-" * 65)
    for v in values:
        n_train = sum(1 for r in train_rows if r["value"] == v)
        n_val = sum(1 for r in val_rows if r["value"] == v)
        total = n_train + n_val
        pct = 100 * n_train / total if total > 0 else 0
        print(f"  {v:<33} {n_train:>6} {n_val:>6} {total:>6} {pct:>6.1f}%")
    print("-" * 65)
    total_train = len(train_rows)
    total_val = len(val_rows)
    total = total_train + total_val
    pct = 100 * total_train / total if total > 0 else 0
    print(
        f"  {'TOTAL':<33} {total_train:>6} {total_val:>6} {total:>6} {pct:>6.1f}%"
    )
    print()
