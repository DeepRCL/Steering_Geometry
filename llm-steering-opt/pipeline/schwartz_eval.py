"""Self-contained Schwartz steering evaluation for ``llm-steering-opt``.

This file owns everything the steering pipeline needs to score a trained
steering vector against the held-out validation set:

* CAA-aligned prompt formatting (full-answer logprob + A/B next-token).
* CAA ``DataLoader._load_and_split`` style ``pos_is_a`` assignment for the
  A/B path (one shared ``random.Random(seed)`` walking values sequentially).
* Single-forward per-sample scoring primitives that mirror
  ``CAA/Geometry/evaluate._score_instance``.
* Per-value + overall aggregation and a small bar-chart plotter.

There is no dependency on a ``shared/`` package.
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ``steering_opt.py`` lives at the repo root next to ``pipeline/``.
_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
import steering_opt  # noqa: E402


# ─── Eval metric constants ──────────────────────────────────────────────────

EVAL_METRIC_FULL_LOGPROB = "full_logprob"
EVAL_METRIC_AB_NEXT_TOKEN = "ab_next_token"
EVAL_METRICS = (EVAL_METRIC_FULL_LOGPROB, EVAL_METRIC_AB_NEXT_TOKEN)


def eval_metric_label(eval_metric: str) -> str:
    if eval_metric == EVAL_METRIC_AB_NEXT_TOKEN:
        return "A/B next-token (CAA)"
    return "Full-answer mean logprob"


# ─── Instruct detection / prompt formatting (CAA-aligned) ───────────────────

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


def format_qa_eval_prompt(
    question: str,
    tokenizer=None,
    model_name: Optional[str] = None,
    use_chat_template: bool = True,
) -> str:
    """Full-answer logprob eval prompt.

    For instruct models the question is wrapped in the chat template (with
    ``add_generation_prompt=True``) so the eval activation distribution
    matches training. Falls back to ``"Q: {question}\\nA: "`` otherwise.
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
        question, positive_answer, negative_answer,
        pos_is_a, tokenizer, model_name,
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
    """Assign ``pos_is_a`` on each eval row in CAA Geometry order.

    Mirrors ``CAA/Geometry/data_loader.py::DataLoader._load_and_split`` for
    the eval portion: one ``random.Random(seed)`` shared across values,
    per-value sequential ``rng.choice([True, False])`` per row.
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


# ─── Logits → per-sample primitives (CAA-aligned) ───────────────────────────

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


def score_ab_from_logits(
    logits_last,
    a_token_id: int,
    b_token_id: int,
    pos_is_a: bool,
) -> Dict[str, Any]:
    """Per-sample A/B result dict, mirroring CAA ``_score_instance``."""
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


# ─── Per-row scoring (single forward, optional steering hooks) ──────────────

@torch.no_grad()
def compute_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device,
    hook_infos: Optional[list] = None,
) -> float:
    """Mean per-token logprob of ``completion`` given ``prompt``.

    Single forward pass on ``prompt + completion``; ``hook_infos`` (if any)
    are applied to the whole pass via ``steering_opt.hf_hooks_contextmanager``.
    """
    full_inputs, prompt_len, answer_ids = prepare_qa_completion_inputs(
        tokenizer, prompt, completion, device
    )
    if hook_infos:
        with steering_opt.hf_hooks_contextmanager(model, hook_infos):
            outputs = model(
                input_ids=full_inputs.input_ids,
                attention_mask=full_inputs.get("attention_mask"),
            )
    else:
        outputs = model(
            input_ids=full_inputs.input_ids,
            attention_mask=full_inputs.get("attention_mask"),
        )
    return mean_completion_logprob_from_logits(
        outputs.logits[0], prompt_len, answer_ids
    )


@torch.no_grad()
def score_ab_next_token(
    model,
    tokenizer,
    row: dict,
    model_name: str,
    device,
    hook_infos: Optional[list] = None,
) -> Dict[str, Any]:
    """CAA-style P(A) vs P(B) on the MCQ prompt ending in ``" ("``.

    ``row['pos_is_a']`` is expected to have been pre-assigned by
    :func:`assign_pos_is_a_caa`.
    """
    pos_is_a = bool(row["pos_is_a"])
    tokens, a_id, b_id = format_ab_eval_tokens(
        row["question"],
        row["positive_answer"],
        row["negative_answer"],
        pos_is_a,
        tokenizer,
        model_name,
    )
    input_ids = torch.tensor([tokens], device=device)

    if hook_infos:
        with steering_opt.hf_hooks_contextmanager(model, hook_infos):
            logits = model(input_ids).logits
    else:
        logits = model(input_ids).logits

    return score_ab_from_logits(logits[0, -1, :], a_id, b_id, pos_is_a)


# ─── Plotting ───────────────────────────────────────────────────────────────

def _plot_eval_accuracy(
    per_value: Dict[str, dict],
    overall: dict,
    values: List[str],
    alpha: float,
    eval_metric: str,
    out_path: str,
) -> None:
    """Grouped bar chart comparing baseline vs steered accuracy."""
    labels = [v for v in values if v in per_value]
    if not labels:
        return

    base_accs = [per_value[v]["accuracy_baseline"] for v in labels]
    steer_accs = [per_value[v]["accuracy_steered"] for v in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 6))
    ax.bar(x - width / 2, base_accs, width, label="Baseline",
           color="#90CAF9", edgecolor="#1565C0")
    ax.bar(x + width / 2, steer_accs, width, label="Steered",
           color="#A5D6A7", edgecolor="#2E7D32")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
    ax.set_ylabel("Accuracy (positive preferred)")
    ax.set_title(
        f"Steering Evaluation — {eval_metric_label(eval_metric)}\n"
        f"(α={alpha}, overall: "
        f"{overall['accuracy_baseline']:.1%} → {overall['accuracy_steered']:.1%})"
    )
    ax.set_xticks(x)
    short_labels = [v.split(":")[-1].strip() if ":" in v else v for v in labels]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# ─── Eval orchestration ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_steering(
    *,
    model,
    tokenizer,
    val_rows: List[dict],
    vectors: Dict[str, torch.Tensor],
    values: List[str],
    layer: Union[int, List[int]],
    alpha: float,
    eval_metric: str,
    model_name: str,
    device,
    output_dir: str,
    schwartz_value_order: List[str],
    random_seed: int,
    n_eval_samples: Optional[int],
    get_rows_for_value: Callable[[List[dict], str], List[dict]],
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Evaluate trained steering vectors on the held-out validation set.

    ``eval_metric`` selects ``full_logprob`` (mean per-token logprob of
    positive vs negative answers) or ``ab_next_token`` (CAA-style A/B prob
    on an MCQ prompt).
    """
    metric_label = eval_metric_label(eval_metric)
    log("\n" + "─" * 60)
    log(f"  Steering Evaluation ({metric_label})")
    log("─" * 60)

    layers = [layer] if isinstance(layer, int) else list(layer)
    use_ab = eval_metric == EVAL_METRIC_AB_NEXT_TOKEN

    records: List[dict] = []
    eval_values = [v for v in values if v in vectors]

    if use_ab:
        assign_pos_is_a_caa(val_rows, schwartz_value_order, random_seed)

    for value in eval_values:
        vec = vectors[value].detach().to(device)
        scaled_vec = alpha * vec
        hook_fn = steering_opt.make_steering_hook_hf(scaled_vec)
        hook_infos = [(l, hook_fn) for l in layers]

        value_rows = get_rows_for_value(val_rows, value)
        if not value_rows:
            log(f"  {value}: no validation rows – skipping")
            continue

        if n_eval_samples is not None and n_eval_samples < len(value_rows):
            rng = random.Random(random_seed)
            value_rows = rng.sample(value_rows, n_eval_samples)

        log(f"  Evaluating {value} ({len(value_rows)} samples) ...")

        pbar = tqdm(value_rows, desc="Eval steering", position=0, leave=True)
        for row in pbar:
            pbar.set_description(f"Eval: {value}")
            rec = {"value": value}

            if use_ab:
                ab_base = score_ab_next_token(
                    model, tokenizer, row, model_name, device
                )
                ab_steer = score_ab_next_token(
                    model, tokenizer, row, model_name, device, hook_infos
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
                    tokenizer=tokenizer,
                    model_name=model_name,
                )
                pos = row["positive_answer"]
                neg = row["negative_answer"]
                rec.update({
                    "lp_pos_base": compute_logprob(
                        model, tokenizer, prompt, pos, device
                    ),
                    "lp_neg_base": compute_logprob(
                        model, tokenizer, prompt, neg, device
                    ),
                    "lp_pos_steer": compute_logprob(
                        model, tokenizer, prompt, pos, device, hook_infos
                    ),
                    "lp_neg_steer": compute_logprob(
                        model, tokenizer, prompt, neg, device, hook_infos
                    ),
                })

            records.append(rec)

    if not records:
        log("  WARNING: no evaluation records collected!")
        return {}

    eval_payload = build_eval_payload(
        eval_metric,
        records,
        values,
        extra_fields={
            "alpha": alpha,
            "layer": layer if isinstance(layer, int) else layers,
        },
    )

    per_value = eval_payload["per_value"]
    overall = eval_payload["overall"]
    delta_key = (
        "mean_delta_positive_margin" if use_ab else "mean_delta_logprob"
    )

    log("")
    log(
        f"  {'Value':<35} {'Base Acc':>9} {'Steer Acc':>10} "
        f"{'Δ Acc':>7} {'Δ':>9}"
    )
    log("  " + "-" * 75)
    for value in values:
        if value not in per_value:
            continue
        m = per_value[value]
        log(
            f"  {value:<35} {m['accuracy_baseline']:>9.1%} "
            f"{m['accuracy_steered']:>10.1%} "
            f"{m['delta_accuracy']:>+7.1%} "
            f"{m.get(delta_key, 0):>+9.4f}"
        )
    log("  " + "-" * 75)
    o = overall
    log(
        f"  {'OVERALL':<35} {o['accuracy_baseline']:>9.1%} "
        f"{o['accuracy_steered']:>10.1%} "
        f"{o['delta_accuracy']:>+7.1%} "
        f"{o.get(delta_key, 0):>+9.4f}\n"
    )

    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, "steering_eval_metrics.json")
    with open(eval_path, "w") as f:
        json.dump(eval_payload, f, indent=2)
    log(f"  Saved evaluation metrics → {eval_path}")

    plot_path = os.path.join(output_dir, "steering_eval_accuracy.png")
    _plot_eval_accuracy(
        per_value,
        overall,
        values=values,
        alpha=alpha,
        eval_metric=eval_metric,
        out_path=plot_path,
    )
    log(f"  Saved accuracy plot   → {plot_path}")

    return eval_payload
