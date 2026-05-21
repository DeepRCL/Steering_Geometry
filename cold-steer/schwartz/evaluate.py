"""Log-likelihood steering evaluation under cold-steer hooks.

For each held-out (question, positive, negative) row we compute the mean
per-token log-probability of the positive and negative answers, both with
the steerer hooks active and with steering bypassed. Reports per-value
and overall accuracy / Δ-logprob, and saves a bar chart that mirrors the
``llm-steering-opt`` and ``odesteer`` plots.
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from shared import schwartz_eval

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data_utils
from . import method_adapters


@torch.no_grad()
def _logprob_of_completion(
    steerable_llm,
    prompt_text: str,
    completion_text: str,
    steerer=None,
) -> float:
    """Mean per-token log-probability of ``completion_text`` given ``prompt_text``.

    Caller is responsible for any hooks they want active during the
    forward pass (e.g. ``register_steering_hooks`` or ``bypass_steering``).

    When ``steerer`` is set, we populate ``gen_input_ids`` /
    ``gen_attention_mask`` so ``LossFDSteerer.steer_output_hook`` can run
    ``get_intermediate_activations`` (cold-steer's expected eval path).
    """
    tokenizer = steerable_llm.tokenizer
    prompt_tok = tokenizer(prompt_text, return_tensors="pt")
    full_tok = tokenizer(f"{prompt_text} {completion_text}", return_tensors="pt")
    device = steerable_llm.model.device
    prompt_len = prompt_tok["input_ids"].shape[1]
    full_input_ids = full_tok["input_ids"].to(device)
    full_attention_mask = full_tok["attention_mask"].to(device)

    answer_ids = full_input_ids[0, prompt_len:]
    if answer_ids.numel() == 0:
        return 0.0

    if steerer is not None:
        steerer.gen_input_ids = full_input_ids
        steerer.gen_attention_mask = full_attention_mask

    out = steerable_llm(input_ids=full_input_ids, attention_mask=full_attention_mask)
    logits = out.logits[0, prompt_len - 1:-1, :]
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    token_lps = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_lps.sum().item() / answer_ids.numel())


@torch.no_grad()
def _score_ab_next_token(
    steerable_llm,
    row: dict,
    model_name: str,
    seed: int,
    steerer=None,
    use_steering: bool = False,
) -> dict:
    """CAA-style P(A) vs P(B) with optional cold-steer hooks on one forward pass."""
    pos_is_a = schwartz_eval.stable_pos_is_a(row, seed)
    prompt = schwartz_eval.format_ab_eval_prompt(
        row["question"],
        row["positive_answer"],
        row["negative_answer"],
        pos_is_a,
        steerable_llm.tokenizer,
        model_name,
    )
    input_ids = steerable_llm.tokenizer(
        prompt, return_tensors="pt"
    ).input_ids.to(steerable_llm.model.device)

    if use_steering and steerer is not None:
        steerer.reset_steering()
        handles = steerable_llm.register_steering_hooks(
            lambda lidx: lambda m, i, o: steerer.steer_output_hook(m, i, o, layer_idx=lidx)
        )
        try:
            logits = steerable_llm(input_ids=input_ids).logits
        finally:
            for h in handles:
                h.remove()
    else:
        with steerer.bypass_steering():
            logits = steerable_llm(input_ids=input_ids).logits

    a_id, b_id = schwartz_eval.ab_token_ids(steerable_llm.tokenizer)
    return schwartz_eval.score_ab_from_logits(
        logits[0, -1, :], a_id, b_id, pos_is_a
    )


def _plot_eval_accuracy(
    per_value: Dict[str, dict],
    overall: dict,
    method: str,
    eta: float,
    eval_metric: str,
    values: List[str],
    out_path: str,
) -> None:
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
        f"{method} — {schwartz_eval.eval_metric_label(eval_metric)}\n"
        f"(η={eta}, overall: {overall['accuracy_baseline']:.1%} → "
        f"{overall['accuracy_steered']:.1%})"
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


def evaluate_steerer(
    steerable_llm,
    steerers_by_value: Dict[str, Any],
    val_rows: List[dict],
    values: List[str],
    layer_idx: int,
    method: str,
    eta: float,
    eval_metric: str,
    model_name: str,
    n_eval_samples: Optional[int],
    seed: int,
    use_chat_template: bool,
    prompt_template: str,
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate one trained steerer per value (metric via ``eval_metric``)."""
    use_ab = eval_metric == schwartz_eval.EVAL_METRIC_AB_NEXT_TOKEN

    if verbose:
        print("─" * 60)
        print(f"  Steering Evaluation ({schwartz_eval.eval_metric_label(eval_metric)})")
        print("─" * 60)

    records: List[dict] = []
    eval_values = [v for v in values if v in steerers_by_value]
    rng = random.Random(seed)
    device = steerable_llm.model.device

    for value in eval_values:
        steerer = steerers_by_value[value]
        method_adapters.load_steerer_state_to_device(steerer, device)
        value_rows = data_utils.get_rows_for_value(val_rows, value)
        if not value_rows:
            if verbose:
                print(f"  {value}: no validation rows, skipping")
            continue
        if (
            n_eval_samples is not None
            and n_eval_samples > 0
            and n_eval_samples < len(value_rows)
        ):
            value_rows = rng.sample(value_rows, n_eval_samples)

        if verbose:
            print(f"  Evaluating {value} ({len(value_rows)} samples) ...")

        for row in tqdm(value_rows, desc=f"Eval: {value}", leave=False):
            rec = {"value": value}

            if use_ab:
                ab_base = _score_ab_next_token(
                    steerable_llm, row, model_name, seed, steerer, use_steering=False
                )
                ab_steer = _score_ab_next_token(
                    steerable_llm, row, model_name, seed, steerer, use_steering=True
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
                prompt_text = data_utils.format_prompt(
                    row["question"],
                    steerable_llm.tokenizer,
                    use_chat_template,
                    prompt_template,
                )
                pos = row["positive_answer"]
                neg = row["negative_answer"]

                with steerer.bypass_steering():
                    lp_pos_base = _logprob_of_completion(steerable_llm, prompt_text, pos)
                    lp_neg_base = _logprob_of_completion(steerable_llm, prompt_text, neg)

                steerer.reset_steering()
                handles = steerable_llm.register_steering_hooks(
                    lambda lidx: lambda m, i, o: steerer.steer_output_hook(
                        m, i, o, layer_idx=lidx
                    )
                )
                try:
                    lp_pos_steer = _logprob_of_completion(
                        steerable_llm, prompt_text, pos, steerer=steerer
                    )
                    lp_neg_steer = _logprob_of_completion(
                        steerable_llm, prompt_text, neg, steerer=steerer
                    )
                finally:
                    for h in handles:
                        h.remove()

                rec.update({
                    "lp_pos_base": lp_pos_base,
                    "lp_neg_base": lp_neg_base,
                    "lp_pos_steer": lp_pos_steer,
                    "lp_neg_steer": lp_neg_steer,
                })

            records.append(rec)

        method_adapters.offload_steerer_state(steerer)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not records:
        if verbose:
            print("  WARNING: no evaluation records collected!")
        return {}

    payload = schwartz_eval.build_eval_payload(
        eval_metric,
        records,
        values,
        extra_fields={"method": method, "eta": eta, "layer": layer_idx},
    )
    per_value = payload["per_value"]
    overall = payload["overall"]
    delta_key = (
        "mean_delta_positive_margin"
        if use_ab
        else "mean_delta_logprob"
    )

    if verbose:
        print(
            f"\n  {'Value':<35} {'Base Acc':>9} {'Steer Acc':>10} "
            f"{'Δ Acc':>7} {'Δ':>9}"
        )
        print("  " + "-" * 75)
        for value in values:
            if value not in per_value:
                continue
            m = per_value[value]
            print(
                f"  {value:<35} {m['accuracy_baseline']:>9.1%} "
                f"{m['accuracy_steered']:>10.1%} "
                f"{m['delta_accuracy']:>+7.1%} "
                f"{m.get(delta_key, 0):>+9.4f}"
            )
        print("  " + "-" * 75)
        o = overall
        print(
            f"  {'OVERALL':<35} {o['accuracy_baseline']:>9.1%} "
            f"{o['accuracy_steered']:>10.1%} "
            f"{o['delta_accuracy']:>+7.1%} "
            f"{o.get(delta_key, 0):>+9.4f}\n"
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "steering_eval_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    _plot_eval_accuracy(
        per_value,
        overall,
        method=method,
        eta=eta,
        eval_metric=eval_metric,
        values=values,
        out_path=os.path.join(output_dir, "steering_eval_accuracy.png"),
    )
    if verbose:
        print(f"  Saved evaluation metrics → {output_dir}/steering_eval_metrics.json")

    return payload
