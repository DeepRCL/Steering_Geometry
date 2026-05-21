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


def format_qa_eval_prompt(
    question: str,
    tokenizer=None,
    model_name: Optional[str] = None,
    use_chat_template: bool = True,
) -> str:
    """Full-answer logprob eval prompt.

    For instruct models (when a tokenizer with a ``chat_template`` is provided
    and ``use_chat_template`` is ``True``), the question is wrapped in the
    chat template with ``add_generation_prompt=True`` so the eval-time
    activation distribution matches training (which also uses the chat
    template on instruct models).

    Falls back to ODESteer's original plain ``"Q: {question}\\nA: "`` format
    when no chat template is available (e.g. Qwen3.5-9B-Base).
    """
    if use_chat_template and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        is_instruct = (
            resolve_is_instruct(model_name, tokenizer) if model_name is not None else True
        )
        if is_instruct:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
    return f"Q: {question}\nA: "


def prepare_qa_completion_inputs(
    tokenizer,
    prompt: str,
    completion: str,
    device,
):
    """
    Tokenize prompt + completion for ODESteer-style full-answer logprob scoring.

    Returns:
        (full_inputs, prompt_len, answer_ids) where ``answer_ids`` are completion
        tokens only (used with logits at positions ``prompt_len-1 : -1``).
    """
    import torch

    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    full_inputs = tokenizer(prompt + completion, return_tensors="pt").to(device)
    answer_ids = full_inputs.input_ids[0, prompt_len:]
    return full_inputs, prompt_len, answer_ids


def mean_completion_logprob_from_logits(
    logits,
    prompt_len: int,
    answer_ids,
) -> float:
    """Mean per-token logprob of ``answer_ids`` from a full-sequence forward pass."""
    import torch
    import torch.nn.functional as F

    if answer_ids.numel() == 0:
        return 0.0
    answer_logits = logits[prompt_len - 1:-1]
    if answer_logits.shape[0] == 0:
        return 0.0
    log_probs = F.log_softmax(answer_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item() / answer_ids.numel())


def compute_mean_completion_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device=None,
) -> float:
    """
    ODESteer Schwartz eval: one forward on ``prompt + completion``, mean token
    logprob over the completion span (no leading-space hack, no per-token reroll).
    """
    import torch

    if device is None:
        device = next(model.parameters()).device
    full_inputs, prompt_len, answer_ids = prepare_qa_completion_inputs(
        tokenizer, prompt, completion, device
    )
    outputs = model(
        input_ids=full_inputs.input_ids,
        attention_mask=full_inputs.get("attention_mask"),
    )
    return mean_completion_logprob_from_logits(
        outputs.logits[0], prompt_len, answer_ids
    )


def resolve_is_instruct(model_name: str, tokenizer) -> bool:
    """Match CAA Geometry ``model_loader.load_model`` instruct detection."""
    name_lower = model_name.lower()
    tokenizer_has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if "qwen" in name_lower:
        return "base" not in name_lower
    if "gemma" in name_lower:
        if "-it" in name_lower:
            return True
        if "-pt" in name_lower:
            return False
        return tokenizer_has_chat_template
    return "base" not in name_lower


def is_instruct_model(model_name: str, tokenizer=None) -> bool:
    """Backward-compatible wrapper around ``resolve_is_instruct``."""
    if tokenizer is not None:
        return resolve_is_instruct(model_name, tokenizer)
    name_lower = model_name.lower()
    return "base" not in name_lower


def assign_pos_is_a_caa(
    rows: List[dict],
    value_order: List[str],
    seed: int,
) -> None:
    """
    Assign ``pos_is_a`` on each row like CAA Geometry ``DataLoader._load_and_split``.

    Iterates ``value_order`` and draws ``rng.choice([True, False])`` per eval row
    in within-value list order (same sequential RNG as CAA eval instances).
    """
    rng = random.Random(seed)
    by_value: Dict[str, List[dict]] = {v: [] for v in value_order}
    for row in rows:
        v = row.get("value")
        if v in by_value:
            by_value[v].append(row)
    for value in value_order:
        for row in by_value.get(value, []):
            row["pos_is_a"] = rng.choice([True, False])


def stable_pos_is_a(row: dict, seed: int) -> bool:
    """Per-row hash assignment (legacy; prefer ``assign_pos_is_a_caa`` for CAA parity)."""
    key = row.get("id") or row.get("sample_id") or row.get("question", "")
    return random.Random(f"{seed}:{key}").choice([True, False])


def eval_pos_is_a(row: dict, seed: int) -> bool:
    """Return CAA-assigned ``pos_is_a`` if present, else legacy stable assignment."""
    if "pos_is_a" in row:
        return bool(row["pos_is_a"])
    return stable_pos_is_a(row, seed)


def format_ab_eval_prompt(
    question: str,
    positive_answer: str,
    negative_answer: str,
    pos_is_a: bool,
    tokenizer,
    model_name: str,
) -> str:
    """CAA Geometry ``PromptFormatter.format_eval_prompt`` prompt (pre-tokenize)."""
    if pos_is_a:
        a_text, b_text = positive_answer, negative_answer
    else:
        a_text, b_text = negative_answer, positive_answer

    prompt_text = (
        f"Question: {question}\n"
        f"(A) {a_text}\n"
        f"(B) {b_text}"
    )
    is_instruct = resolve_is_instruct(model_name, tokenizer)
    if is_instruct and getattr(tokenizer, "chat_template", None):
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        base = prompt_text + "\nResponse:"
    return base + " ("


def format_ab_eval_tokens(
    question: str,
    positive_answer: str,
    negative_answer: str,
    pos_is_a: bool,
    tokenizer,
    model_name: str,
) -> tuple:
    """
    CAA Geometry ``PromptFormatter.format_eval_prompt``.

    Returns:
        (token_ids, a_token_id, b_token_id) with ``add_special_tokens=True``.
    """
    eval_prompt = format_ab_eval_prompt(
        question,
        positive_answer,
        negative_answer,
        pos_is_a,
        tokenizer,
        model_name,
    )
    tokens = tokenizer.encode(eval_prompt, add_special_tokens=True)
    a_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    b_id = tokenizer.encode("B", add_special_tokens=False)[-1]
    return tokens, a_id, b_id


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
