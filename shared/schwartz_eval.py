"""
Shared Schwartz steering evaluation utilities.

Provides CAA-aligned A/B multiple-choice scoring and metric aggregation so
llm-steering-opt, odesteer, and cold-steer can be compared on the same metric.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np

EVAL_METRIC_FULL_LOGPROB = "full_logprob"
EVAL_METRIC_AB_NEXT_TOKEN = "ab_next_token"
EVAL_METRICS = (EVAL_METRIC_FULL_LOGPROB, EVAL_METRIC_AB_NEXT_TOKEN)


def is_instruct_model(model_name: str) -> bool:
    """Match CAA Geometry ``ModelInfo.is_instruct`` (Base models → plain prompts)."""
    name_lower = model_name.lower()
    if "base" in name_lower:
        return False
    return True


def stable_pos_is_a(row: dict, seed: int) -> bool:
    """Deterministic A/B assignment per row (reproducible across runs)."""
    key = row.get("id") or row.get("sample_id") or row.get("question", "")
    return random.Random(f"{seed}:{key}").choice([True, False])


def format_ab_eval_prompt(
    question: str,
    positive_answer: str,
    negative_answer: str,
    pos_is_a: bool,
    tokenizer,
    model_name: str,
) -> str:
    """
    CAA-style MCQ prompt ending with ``" ("`` for next-token A vs B comparison.

    Uses ``question\\nResponse:`` for base models and chat template for instruct.
    """
    if pos_is_a:
        a_text, b_text = positive_answer, negative_answer
    else:
        a_text, b_text = negative_answer, positive_answer

    body = (
        f"Question: {question}\n"
        f"(A) {a_text}\n"
        f"(B) {b_text}"
    )

    if is_instruct_model(model_name) and getattr(tokenizer, "chat_template", None):
        try:
            base = tokenizer.apply_chat_template(
                [{"role": "user", "content": body}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            base = body + "\nResponse:"
    else:
        base = body + "\nResponse:"

    return base + " ("


def ab_token_ids(tokenizer) -> tuple:
    """Last token id for bare ``A`` and ``B`` (CAA convention)."""
    a_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    b_id = tokenizer.encode("B", add_special_tokens=False)[-1]
    return a_id, b_id


def score_ab_from_logits(
    logits_last,
    a_token_id: int,
    b_token_id: int,
    pos_is_a: bool,
) -> Dict[str, Any]:
    """Build per-sample A/B result dict from last-position logits (1D tensor)."""
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits_last.float(), dim=-1)
    prob_a = probs[a_token_id].item()
    prob_b = probs[b_token_id].item()
    prob_positive = prob_a if pos_is_a else prob_b
    prob_negative = prob_b if pos_is_a else prob_a
    chose_a = prob_a > prob_b

    return {
        "prob_a": prob_a,
        "prob_b": prob_b,
        "prob_positive": prob_positive,
        "prob_negative": prob_negative,
        "positive_margin": prob_positive - prob_negative,
        "chose_a": chose_a,
        "pos_is_a": pos_is_a,
        "is_correct": chose_a == pos_is_a,
    }


def aggregate_full_logprob_records(records: List[dict]) -> dict:
    """Aggregate logprob-preference records (lp_pos_* / lp_neg_*)."""
    n = len(records)
    if n == 0:
        return {"n_samples": 0}

    cb = sum(1 for r in records if r["lp_pos_base"] > r["lp_neg_base"])
    cs = sum(1 for r in records if r["lp_pos_steer"] > r["lp_neg_steer"])
    delta_lp = [
        (r["lp_pos_steer"] - r["lp_neg_steer"]) - (r["lp_pos_base"] - r["lp_neg_base"])
        for r in records
    ]
    return {
        "n_samples": n,
        "accuracy_baseline": round(cb / n, 4),
        "accuracy_steered": round(cs / n, 4),
        "delta_accuracy": round((cs - cb) / n, 4),
        "mean_delta_logprob": round(float(np.mean(delta_lp)), 6),
        "std_delta_logprob": round(float(np.std(delta_lp)), 6),
        "mean_lp_pos_baseline": round(
            float(np.mean([r["lp_pos_base"] for r in records])), 6
        ),
        "mean_lp_pos_steered": round(
            float(np.mean([r["lp_pos_steer"] for r in records])), 6
        ),
        "mean_lp_neg_baseline": round(
            float(np.mean([r["lp_neg_base"] for r in records])), 6
        ),
        "mean_lp_neg_steered": round(
            float(np.mean([r["lp_neg_steer"] for r in records])), 6
        ),
    }


def aggregate_ab_records(records: List[dict]) -> dict:
    """Aggregate A/B next-token records (ab_correct_* / margins)."""
    n = len(records)
    if n == 0:
        return {"n_samples": 0}

    cb = sum(1 for r in records if r["ab_correct_base"])
    cs = sum(1 for r in records if r["ab_correct_steer"])
    delta_margin = [r["ab_margin_steer"] - r["ab_margin_base"] for r in records]
    return {
        "n_samples": n,
        "accuracy_baseline": round(cb / n, 4),
        "accuracy_steered": round(cs / n, 4),
        "delta_accuracy": round((cs - cb) / n, 4),
        "mean_delta_positive_margin": round(float(np.mean(delta_margin)), 6),
        "mean_prob_positive_baseline": round(
            float(np.mean([r["ab_prob_positive_base"] for r in records])), 6
        ),
        "mean_prob_positive_steered": round(
            float(np.mean([r["ab_prob_positive_steer"] for r in records])), 6
        ),
        "mean_prob_negative_baseline": round(
            float(np.mean([r["ab_prob_negative_base"] for r in records])), 6
        ),
        "mean_prob_negative_steered": round(
            float(np.mean([r["ab_prob_negative_steer"] for r in records])), 6
        ),
    }


def build_eval_payload(
    eval_metric: str,
    records: List[dict],
    values: List[str],
    extra_fields: Optional[dict] = None,
) -> dict:
    """Per-value + overall metrics for the selected evaluation method."""
    if eval_metric == EVAL_METRIC_AB_NEXT_TOKEN:
        aggregate = aggregate_ab_records
    elif eval_metric == EVAL_METRIC_FULL_LOGPROB:
        aggregate = aggregate_full_logprob_records
    else:
        raise ValueError(
            f"Unknown eval_metric {eval_metric!r}; choose from {EVAL_METRICS}"
        )

    overall = aggregate(records)
    per_value: Dict[str, dict] = {}
    for value in values:
        vrecs = [r for r in records if r["value"] == value]
        if vrecs:
            per_value[value] = aggregate(vrecs)

    payload = {
        "eval_metric": eval_metric,
        "overall": overall,
        "per_value": per_value,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def eval_metric_label(eval_metric: str) -> str:
    if eval_metric == EVAL_METRIC_AB_NEXT_TOKEN:
        return "A/B next-token (CAA)"
    return "Full-answer mean logprob"
