"""
Schwartz value dataset loading, stratified splitting, and prompt formatting.
"""

import csv
import random
from collections import defaultdict
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Schwartz (2012) Refined Theory — circumplex order (counter-clockwise)
# ─────────────────────────────────────────────────────────────────────────────
SCHWARTZ_CIRCUMPLEX_ORDER: List[str] = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: dependability",
    "Benevolence: caring",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
    "Universalism: objectivity",
]

HIGHER_ORDER_GROUPS = {
    "Openness to Change": [
        "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism",
    ],
    "Self-Enhancement": [
        "Achievement", "Power: dominance", "Power: resources", "Face",
    ],
    "Conservation": [
        "Security: personal", "Security: societal", "Tradition",
        "Conformity: rules", "Conformity: interpersonal", "Humility",
    ],
    "Self-Transcendence": [
        "Benevolence: dependability", "Benevolence: caring",
        "Universalism: concern", "Universalism: nature",
        "Universalism: tolerance", "Universalism: objectivity",
    ],
}

GROUP_COLORS = {
    "Openness to Change": "#2196F3",
    "Self-Enhancement":   "#F44336",
    "Conservation":       "#FF9800",
    "Self-Transcendence": "#4CAF50",
}


def value_to_group(value: str) -> str:
    for group, members in HIGHER_ORDER_GROUPS.items():
        if value in members:
            return group
    return "Unknown"


# ─── CSV Loading ─────────────────────────────────────────────────────────────

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
    """
    rng = random.Random(seed)
    by_value: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_value[row["value"]].append(row)

    train_rows, val_rows = [], []
    for value in sorted(by_value.keys()):
        group = by_value[value]
        rng.shuffle(group)
        n_train = max(1, int(len(group) * train_ratio))
        if n_train >= len(group):
            n_train = len(group) - 1
        train_rows.extend(group[:n_train])
        val_rows.extend(group[n_train:])

    return train_rows, val_rows


def get_rows_for_value(rows: List[dict], value: str) -> List[dict]:
    return [r for r in rows if r["value"] == value]


# ─── Prompt Formatting ──────────────────────────────────────────────────────

def format_qa_prompt(question: str, answer: str) -> str:
    """Format a question + answer into a full Q/A text for activation extraction.

    This matches ODESteer's `extract_base_activations` pattern:
        full_prompts = [f'Q: {q}\\nA: {a}' for q, a in zip(questions, answers)]
    """
    return f"Q: {question}\nA: {answer}"


def format_question_prompt(question: str) -> str:
    """Format just the question for compute_answer_prob.

    Matches ODESteer's prompt_template default: 'Q: {question}\\nA: '
    """
    return f"Q: {question}\nA: "


# ─── Pretty Printing ────────────────────────────────────────────────────────

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
    print(
        f"  {'TOTAL':<33} {len(train_rows):>6} {len(val_rows):>6} "
        f"{len(train_rows)+len(val_rows):>6} "
        f"{100*len(train_rows)/(len(train_rows)+len(val_rows)):.1f}%"
    )
    print()
