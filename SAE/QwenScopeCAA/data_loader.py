"""
Data loading for the QwenScopeCAA pipeline.

Combines two sources:
  1. base_dataset_path  (final_dataset_200.csv) — all rows, all 20 values
  2. touche_dataset_path (touche_gemma4-v2_remaining-validated-final.csv)
     filtered to caa_suitable=True, up to touche_samples_per_value per value.
     Hedonism has only 90 records in this file — all 90 are used without
     padding; the slight count imbalance is accepted and handled naturally by
     the per-value mean in the CAA extraction step.

All primitive types (ContrastivePair, format_prompts, format_eval_prompt,
split_dataset, print_dataset_summary) are re-exported directly from
SAE.SparseCAA.data_loader — the logic is identical; only the config defaults
differ (100 samples/value, -final CSV).
"""
from __future__ import annotations

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
    df = pd.read_csv(path)
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
    QwenScopePipelineConfig and SparseCAAPipelineConfig.
    """
    base   = _load_base(config.base_dataset_path)
    touche = _load_touche(config.touche_dataset_path, config.touche_samples_per_value)

    combined = pd.concat([base, touche], ignore_index=True)
    # Base rows take precedence (listed first); duplicates from Touche are dropped
    combined = combined.drop_duplicates(subset=["sample_id"], keep="first")
    combined = combined[combined["value"].isin(SCHWARTZ_CIRCUMPLEX_ORDER)]
    combined = combined.reset_index(drop=True)
    return combined
