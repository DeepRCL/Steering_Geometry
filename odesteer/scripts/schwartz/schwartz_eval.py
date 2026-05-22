"""
ODESteer Schwartz steering evaluation (self-contained).

This module is the single home for everything ODESteer needs at eval time:

* Prompt + token formatting helpers, mirroring CAA Geometry
  ``data_loader.PromptFormatter`` (``format_eval_prompt``,
  ``_format_base_prompt``) exactly.
* Per-row A/B next-token scoring: one forward pass per eval row, using the
  per-row ``pos_is_a`` assigned by ``assign_pos_is_a_caa`` — identical to
  CAA Geometry ``evaluate._score_instance`` + ``data_loader.DataLoader``.
* Full-answer mean log-probability scoring (ODESteer's original eval).
* The main ``evaluate_steering`` entrypoint plus per-value / overall
  aggregation, including the bar-plot helper used by the pipeline.

CAA is the gold-standard reference: the A/B path here mirrors it without
modification. The only Schwartz-specific behaviour layered on top is the
shuffle / RNG flow needed to assign ``pos_is_a`` to our val split (since
ODESteer does its own stratified split rather than reusing CAA's).
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


EVAL_METRIC_FULL_LOGPROB = "full_logprob"
EVAL_METRIC_AB_NEXT_TOKEN = "ab_next_token"
EVAL_METRICS = (EVAL_METRIC_FULL_LOGPROB, EVAL_METRIC_AB_NEXT_TOKEN)


# ─── Instruct / chat-template detection ─────────────────────────────────────


def resolve_is_instruct(model_name: Optional[str], tokenizer) -> bool:
    """Match CAA Geometry ``model_loader.load_model`` instruct detection."""
    name_lower = (model_name or "").lower()
    tokenizer_has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if "qwen" in name_lower:
        return "base" not in name_lower
    if "gemma" in name_lower:
        if "-it" in name_lower:
            return True
        if "-pt" in name_lower:
            return False
        return tokenizer_has_chat_template
    if name_lower:
        return "base" not in name_lower
    return tokenizer_has_chat_template


# ─── Prompt formatting (CAA-aligned) ────────────────────────────────────────


def format_qa_eval_prompt(
    question: str,
    tokenizer=None,
    model_name: Optional[str] = None,
    use_chat_template: bool = True,
) -> str:
    """Full-answer logprob eval prompt.

    For instruct models (chat-template available and selected), the question
    is rendered through ``apply_chat_template(add_generation_prompt=True)``
    so the eval-time activation distribution matches the chat-template
    training prompts. Falls back to ``"Q: {question}\\nA: "`` otherwise.
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


def format_ab_eval_prompt(
    question: str,
    positive_answer: str,
    negative_answer: str,
    pos_is_a: bool,
    tokenizer,
    model_name: Optional[str],
) -> str:
    """CAA Geometry ``PromptFormatter.format_eval_prompt`` (pre-tokenize)."""
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
    model_name: Optional[str],
) -> Tuple[List[int], int, int]:
    """Returns ``(token_ids, a_token_id, b_token_id)`` (CAA Geometry style)."""
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


# ─── pos_is_a assignment (CAA-aligned) ──────────────────────────────────────


def assign_pos_is_a_caa(
    rows: List[dict],
    value_order: List[str],
    seed: int,
) -> None:
    """
    Assign ``pos_is_a`` on each eval row in CAA Geometry order.

    Matches ``CAA/Geometry/data_loader.py::DataLoader._load_and_split`` for the
    eval portion: per value, sequentially draw ``rng.choice([True, False])``
    from a single seeded ``random.Random(seed)`` shared across values. Across
    the val set this yields ~50/50 positive=A vs positive=B, so a model with
    "always pick (A)" position bias scores ~50% baseline regardless of bias.
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


# ─── Logits → per-sample scoring primitives (CAA-aligned) ───────────────────


def score_ab_from_logits(
    logits_last,
    a_token_id: int,
    b_token_id: int,
    pos_is_a: bool,
) -> Dict[str, Any]:
    """Per-sample A/B result dict from last-position logits.

    Mirrors ``CAA/Geometry/evaluate.py::_score_instance``: a single forward
    pass, soft-max over the last-token logits, ``is_correct = chose_a ==
    pos_is_a``.
    """
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


def mean_completion_logprob_from_logits(
    logits,
    prompt_len: int,
    answer_ids,
) -> float:
    """Mean per-token logprob of ``answer_ids`` from a full-sequence forward pass."""
    if answer_ids.numel() == 0:
        return 0.0
    answer_logits = logits[prompt_len - 1:-1]
    if answer_logits.shape[0] == 0:
        return 0.0
    log_probs = F.log_softmax(answer_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item() / answer_ids.numel())


def prepare_qa_completion_inputs(
    tokenizer,
    prompt: str,
    completion: str,
    device,
):
    """Tokenize ``prompt + completion`` for single-forward logprob scoring."""
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    full_inputs = tokenizer(prompt + completion, return_tensors="pt").to(device)
    answer_ids = full_inputs.input_ids[0, prompt_len:]
    return full_inputs, prompt_len, answer_ids


# ─── Aggregation ────────────────────────────────────────────────────────────


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
    """Aggregate A/B next-token records."""
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


# ─── Per-row scoring tied to HuggingFaceLM hooks ────────────────────────────


@torch.no_grad()
def score_ab_next_token(
    hf_lm,
    row: dict,
    model_name: str,
    steer_T: float,
    use_steering: bool = False,
) -> Dict[str, Any]:
    """CAA Geometry ``_score_instance`` under optional ODESteer steering.

    One forward pass, using the ``pos_is_a`` field assigned by
    ``assign_pos_is_a_caa``. Steering (if enabled) is applied via
    ``hf_lm.register_steer_prob_hook`` on the last prompt position so the
    next-token A/B prediction is affected.
    """
    pos_is_a = bool(row["pos_is_a"])
    tokens, a_id, b_id = format_ab_eval_tokens(
        row["question"],
        row["positive_answer"],
        row["negative_answer"],
        pos_is_a,
        hf_lm.tokenizer,
        model_name,
    )
    input_ids = torch.tensor([tokens], device=hf_lm.model.device)

    if use_steering and hf_lm.steer_model is not None:
        hf_lm.register_steer_prob_hook(input_ids.shape[1] - 1, {"T": steer_T})
        try:
            logits = hf_lm.model(input_ids).logits
        finally:
            hf_lm.remove_steer_prob_hook()
    else:
        logits = hf_lm.model(input_ids).logits

    return score_ab_from_logits(logits[0, -1, :], a_id, b_id, pos_is_a)


@torch.no_grad()
def compute_full_logprob(
    hf_lm,
    prompt: str,
    completion: str,
    steer_T: float,
    use_steering: bool = False,
) -> float:
    """Single-forward mean per-token logprob of ``completion`` under prompt."""
    tokenizer = hf_lm.tokenizer
    device = hf_lm.model.device
    full_inputs, prompt_len, answer_ids = prepare_qa_completion_inputs(
        tokenizer, prompt, completion, device
    )

    if use_steering and hf_lm.steer_model is not None:
        hf_lm.register_steer_prob_hook(prompt_len - 1, {"T": steer_T})
        try:
            outputs = hf_lm.model(**full_inputs)
        finally:
            hf_lm.remove_steer_prob_hook()
    else:
        outputs = hf_lm.model(**full_inputs)

    return mean_completion_logprob_from_logits(
        outputs.logits[0], prompt_len, answer_ids
    )


# ─── Eval orchestration ─────────────────────────────────────────────────────


def _plot_eval_accuracy(
    per_val: Dict[str, dict],
    out_dir: str,
    value_order: List[str],
    eval_metric: str,
    verbose: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("  matplotlib not installed, skipping evaluation plot.")
        return

    labels = [v for v in value_order if v in per_val]
    if not labels:
        return

    base_accs = [per_val[v]["accuracy_baseline"] for v in labels]
    steer_accs = [per_val[v]["accuracy_steered"] for v in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 6))
    ax.bar(x - width / 2, base_accs, width, label="Baseline", color="#90CAF9", edgecolor="#1565C0")
    ax.bar(x + width / 2, steer_accs, width, label="Steered", color="#A5D6A7", edgecolor="#2E7D32")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
    ax.set_ylabel("Accuracy (positive preferred)")
    ax.set_title(f"ODESteer — {eval_metric_label(eval_metric)}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "steering_eval_accuracy.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  Saved evaluation plot → {plot_path}")


@torch.no_grad()
def evaluate_steering(
    hf_lm,
    steer_models: Dict[str, Any],
    val_rows: List[dict],
    values: List[str],
    args,
    *,
    get_rows_for_value: Callable[[List[dict], str], List[dict]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate steering on the validation set.

    The metric is chosen via ``args.eval_metric`` (``full_logprob`` or
    ``ab_next_token``). For ``ab_next_token`` we first assign ``pos_is_a``
    on each row using CAA Geometry's sequential per-value RNG flow, then
    score each row with a single forward pass — identical to CAA's
    ``_score_instance``.
    """
    eval_metric = getattr(args, "eval_metric", EVAL_METRIC_FULL_LOGPROB)
    use_ab = eval_metric == EVAL_METRIC_AB_NEXT_TOKEN

    if verbose:
        print("─" * 60)
        print(f"  Steering Evaluation ({eval_metric_label(eval_metric)})")
        print("─" * 60)

    if use_ab:
        assign_pos_is_a_caa(val_rows, values, args.seed)

    records: List[dict] = []
    eval_values = [v for v in values if v in steer_models]

    for value in eval_values:
        value_val_rows = get_rows_for_value(val_rows, value)
        if not value_val_rows:
            continue
        if args.n_eval_samples and args.n_eval_samples < len(value_val_rows):
            rng = random.Random(args.seed)
            value_val_rows = rng.sample(value_val_rows, args.n_eval_samples)

        hf_lm.steer_model = steer_models[value]

        if verbose:
            print(f"  {value} ({len(value_val_rows)} samples) ...")

        for row in tqdm(value_val_rows, desc=f"Eval: {value}", leave=False):
            rec = {"value": value}

            if use_ab:
                ab_base = score_ab_next_token(
                    hf_lm, row, args.model, args.T, use_steering=False
                )
                ab_steer = score_ab_next_token(
                    hf_lm, row, args.model, args.T, use_steering=True
                )
                rec.update({
                    "ab_prob_positive_base": ab_base["prob_positive"],
                    "ab_prob_negative_base": ab_base["prob_negative"],
                    "ab_margin_base": ab_base["positive_margin"],
                    "ab_correct_base": ab_base["is_correct"],
                    "ab_prob_positive_steer": ab_steer["prob_positive"],
                    "ab_prob_negative_steer": ab_steer["prob_negative"],
                    "ab_margin_steer": ab_steer["positive_margin"],
                    "ab_correct_steer": ab_steer["is_correct"],
                })
            else:
                prompt = format_qa_eval_prompt(
                    row["question"],
                    tokenizer=hf_lm.tokenizer,
                    model_name=args.model,
                )
                pos = row["positive_answer"]
                neg = row["negative_answer"]
                rec.update({
                    "lp_pos_base": compute_full_logprob(hf_lm, prompt, pos, args.T, use_steering=False),
                    "lp_neg_base": compute_full_logprob(hf_lm, prompt, neg, args.T, use_steering=False),
                    "lp_pos_steer": compute_full_logprob(hf_lm, prompt, pos, args.T, use_steering=True),
                    "lp_neg_steer": compute_full_logprob(hf_lm, prompt, neg, args.T, use_steering=True),
                })

            records.append(rec)

    hf_lm.steer_model = None

    if not records:
        if verbose:
            print("  WARNING: no evaluation records!")
        return {}

    eval_payload = build_eval_payload(
        eval_metric,
        records,
        values,
        extra_fields={"T": args.T, "steer_type": args.steer_type},
    )
    per_value = eval_payload["per_value"]
    overall = eval_payload["overall"]
    delta_key = (
        "mean_delta_positive_margin"
        if use_ab
        else "mean_delta_logprob"
    )

    if verbose:
        print(f"\n  {'Value':<35} {'Base Acc':>9} {'Steer Acc':>10} {'Δ Acc':>7} {'Δ':>9}")
        print("  " + "-" * 75)
        for value in values:
            if value not in per_value:
                continue
            m = per_value[value]
            print(
                f"  {value:<35} {m['accuracy_baseline']:>9.1%} "
                f"{m['accuracy_steered']:>10.1%} {m['delta_accuracy']:>+7.1%} "
                f"{m.get(delta_key, 0):>+9.4f}"
            )
        print("  " + "-" * 75)
        o = overall
        print(
            f"  {'OVERALL':<35} {o['accuracy_baseline']:>9.1%} "
            f"{o['accuracy_steered']:>10.1%} {o['delta_accuracy']:>+7.1%} "
            f"{o.get(delta_key, 0):>+9.4f}\n"
        )

    if getattr(args, "output_dir", None):
        _plot_eval_accuracy(
            per_value,
            args.output_dir,
            values,
            eval_metric,
            verbose=verbose,
        )

    return eval_payload
