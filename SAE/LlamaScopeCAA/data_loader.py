"""
Data loading for the LlamaScopeCAA pipeline.

For SAE fine-tuning, combines two sources:
  1. base_dataset_path  (final_dataset_200.csv) — all rows, all 20 values
  2. touche_dataset_path (touche_gemma4-v2_remaining-validated-final.csv)
     filtered to caa_suitable=True, up to touche_samples_per_value per value.
     Hedonism has only 90 records in this file — all 90 are used without
     padding; the slight count imbalance is accepted and handled naturally by
     the per-value mean in the CAA extraction step.

After fine-tuning, persona-vector extraction, A/B steering evaluation, and
geometry use a saved CAA-compatible base-only split so LlamaScopeCAA is directly
comparable with CAA/SphericalSteer.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Re-export all shared primitives from SparseCAA ───────────────────────────
from SAE.SparseCAA.data_loader import (  # noqa: F401
    ContrastivePair,
    format_eval_prompt,
    format_prompts,
    print_dataset_summary,
    split_dataset,
)

from .config import SCHWARTZ_CIRCUMPLEX_ORDER


# ──────────────────────────────────────────────────────────────────────────────
# Internal loaders (duplicated from SparseCAA so this module is self-contained)
# ──────────────────────────────────────────────────────────────────────────────
def _load_base(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"id": "sample_id"})
    return df[["sample_id", "question", "value", "positive_answer", "negative_answer"]]


def _load_touche(path: str, max_per_value: int) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df = df[df["caa_suitable"] == True].copy()  # noqa: E712
    df = df.rename(columns={"argument_id": "sample_id"})
    df = df[["sample_id", "question", "value", "positive_answer", "negative_answer"]]

    parts = []
    for _, grp in df.groupby("value"):
        parts.append(grp.head(max_per_value))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


# ──────────────────────────────────────────────────────────────────────────────
# Public: load combined dataset
# ──────────────────────────────────────────────────────────────────────────────
def load_combined(config) -> pd.DataFrame:
    """
    Return the merged dataframe (base + Touche supplement), deduplicated.

    Accepts any config object with the following attributes:
        base_dataset_path         : str
        touche_dataset_path       : str
        touche_samples_per_value  : int

    This duck-typed signature lets the function work with both
    LlamaScopePipelineConfig and SparseCAAPipelineConfig.
    """
    base   = _load_base(config.base_dataset_path)
    touche = _load_touche(config.touche_dataset_path, config.touche_samples_per_value)

    combined = pd.concat([base, touche], ignore_index=True)
    # Base rows take precedence (listed first); duplicates from Touche are dropped
    combined = combined.drop_duplicates(subset=["sample_id"], keep="first")
    combined = combined[combined["value"].isin(SCHWARTZ_CIRCUMPLEX_ORDER)]
    combined = combined.reset_index(drop=True)
    return combined


def _pair_to_manifest(pair: ContrastivePair) -> dict:
    return {
        "sample_id": pair.sample_id,
        "value": pair.value,
        "question": pair.question,
        "positive_answer": pair.positive_answer,
        "negative_answer": pair.negative_answer,
        "pos_is_a": pair.pos_is_a,
    }


def _pair_from_manifest(row: dict) -> ContrastivePair:
    return ContrastivePair(
        sample_id=str(row["sample_id"]),
        value=row["value"],
        question=row["question"],
        positive_answer=row["positive_answer"],
        negative_answer=row["negative_answer"],
        pos_is_a=bool(row["pos_is_a"]),
    )


def _save_steering_split_manifest(
    config,
    train_data: Dict[str, List[ContrastivePair]],
    eval_data: Dict[str, List[ContrastivePair]],
) -> None:
    path = config.steering_split_manifest_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest = {
        "split_source": "caa_compatible_base_only",
        "split_algorithm": "CAA.Geometry.data_loader.DataLoader",
        "base_dataset_path": config.base_dataset_path,
        "eval_split": config.eval_split,
        "seed": config.seed,
        "values": {},
    }
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        manifest["values"][val] = {
            "train": [_pair_to_manifest(p) for p in train_data.get(val, [])],
            "eval": [_pair_to_manifest(p) for p in eval_data.get(val, [])],
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _load_steering_split_manifest(
    config,
) -> Tuple[Dict[str, List[ContrastivePair]], Dict[str, List[ContrastivePair]]]:
    path = config.steering_split_manifest_path
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)

    expected = {
        "base_dataset_path": config.base_dataset_path,
        "eval_split": config.eval_split,
        "seed": config.seed,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Existing split manifest {path} was created with "
                f"{key}={manifest.get(key)!r}, but the current config uses {value!r}. "
                "Use a different output_dir or remove the stale split manifest."
            )

    train_data: Dict[str, List[ContrastivePair]] = {}
    eval_data: Dict[str, List[ContrastivePair]] = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        entry = manifest["values"].get(val, {"train": [], "eval": []})
        train_data[val] = [_pair_from_manifest(p) for p in entry["train"]]
        eval_data[val] = [_pair_from_manifest(p) for p in entry["eval"]]
    return train_data, eval_data


def _create_caa_compatible_steering_split(
    config,
) -> Tuple[Dict[str, List[ContrastivePair]], Dict[str, List[ContrastivePair]]]:
    """
    Match CAA/Geometry/data_loader.py exactly for value-vector extraction and
    A/B evaluation: base dataset only, per-value shuffled 90/10 split, NumPy
    A/B polarity for training prompts, and Python-random A/B polarity for eval.
    """
    df = _load_base(config.base_dataset_path)
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    grouped: Dict[str, List[ContrastivePair]] = {}
    for _, row in df.iterrows():
        val = row["value"]
        if val not in SCHWARTZ_CIRCUMPLEX_ORDER:
            continue
        pair = ContrastivePair(
            sample_id=str(row["sample_id"]),
            value=val,
            question=row["question"],
            positive_answer=row["positive_answer"],
            negative_answer=row["negative_answer"],
        )
        grouped.setdefault(val, []).append(pair)

    train_data: Dict[str, List[ContrastivePair]] = {}
    eval_data: Dict[str, List[ContrastivePair]] = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        pairs = grouped.get(val, [])
        rng.shuffle(pairs)
        num_eval = int(len(pairs) * config.eval_split)
        train_pairs = pairs[num_eval:]
        eval_pairs = pairs[:num_eval]

        for pair in train_pairs:
            pair.pos_is_a = bool(np_rng.integers(0, 2))
        for pair in eval_pairs:
            pair.pos_is_a = rng.choice([True, False])

        train_data[val] = train_pairs
        eval_data[val] = eval_pairs

    return train_data, eval_data


def load_steering_split(
    config,
) -> Tuple[Dict[str, List[ContrastivePair]], Dict[str, List[ContrastivePair]]]:
    """
    Load or create the split used after SAE fine-tuning.

    Fine-tuning still uses load_combined(...)+split_dataset(...).  This split is
    only for extracting persona vectors, steering evaluation, and geometry, and
    intentionally mirrors CAA/SphericalSteer.
    """
    path = config.steering_split_manifest_path
    if os.path.exists(path):
        return _load_steering_split_manifest(config)

    train_data, eval_data = _create_caa_compatible_steering_split(config)
    _save_steering_split_manifest(config, train_data, eval_data)
    print(f"Saved CAA-compatible steering split manifest -> {path}")
    return train_data, eval_data
