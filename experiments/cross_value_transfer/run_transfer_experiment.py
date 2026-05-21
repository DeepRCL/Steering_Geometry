"""
Core runner for the Cross-Value Steering Transfer experiment.

Public API
----------
run_experiment(config, methods)
    Load model once, evaluate baselines, build T matrix per method, save
    outputs, and generate the cross-method comparison table.

Internal helpers (also importable for testing)
-----------------------------------------------
load_eval_instances(config)
compute_baseline(model_info, eval_instances, formatter, output_dir, force)
build_T_matrix(method, model_info, eval_instances, formatter, baseline_accs, alpha)
save_method_outputs(method_name, T, metrics_dict, baseline_accs, output_dir, alpha)
generate_comparison_table(output_dir)
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm

# ── project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from CAA.Geometry.model_loader import load_model, ModelInfo
from CAA.Geometry.data_loader import PromptFormatter, EvalInstance
from CAA.Geometry.evaluate import _score_instance

from .circumplex_utils import (
    CIRCUMPLEX_ORDER,
    CIRCUMPLEX_IDX,
    HO_BLOCK_BOUNDARIES,
    circular_distance,
    load_R_matrix,
    flatten_off_diagonal,
)
from .metrics import compute_all_metrics
from .config import TransferExperimentConfig
from .steering_method import SteeringMethod


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Evaluation data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_instances(
    config: TransferExperimentConfig,
) -> Dict[str, List[EvalInstance]]:
    """Load and sample evaluation instances from the Touché CSV.

    Filters to rows where ``caa_suitable == True``, then for each Schwartz
    value samples ``min(config.n_eval_samples, available)`` rows in a
    reproducible order (seeded by ``config.seed``).  Each instance gets a
    randomly assigned ``pos_is_a`` flag.

    Returns
    -------
    dict mapping each value name → list of ``EvalInstance`` objects.
    """
    rng = random.Random(config.seed)
    grouped: Dict[str, List[dict]] = {v: [] for v in CIRCUMPLEX_ORDER}

    with open(config.eval_dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("value", "").strip()
            if value not in CIRCUMPLEX_ORDER:
                continue
            if row.get("caa_suitable", "").strip() != "True":
                continue
            grouped[value].append(row)

    eval_instances: Dict[str, List[EvalInstance]] = {}
    for value in CIRCUMPLEX_ORDER:
        rows = grouped[value]
        rng.shuffle(rows)
        rows = rows[: config.n_eval_samples]

        instances = []
        for row in rows:
            instances.append(
                EvalInstance(
                    sample_id=row.get("argument_id", row.get("id", "")),
                    value=value,
                    question=row["question"],
                    positive_answer=row["positive_answer"],
                    negative_answer=row["negative_answer"],
                    pos_is_a=rng.choice([True, False]),
                )
            )
        eval_instances[value] = instances

    n_total = sum(len(v) for v in eval_instances.values())
    print(f"Loaded eval instances: {n_total} total across {len(CIRCUMPLEX_ORDER)} values")
    for v in CIRCUMPLEX_ORDER:
        n = len(eval_instances[v])
        if n < config.n_eval_samples:
            print(f"  Warning: only {n} caa_suitable rows for '{v}' (< {config.n_eval_samples})")

    return eval_instances


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Baseline accuracy
# ─────────────────────────────────────────────────────────────────────────────

def compute_baseline(
    model_info: ModelInfo,
    eval_instances: Dict[str, List[EvalInstance]],
    formatter: PromptFormatter,
    output_dir: Path,
    force: bool = False,
) -> Dict[str, float]:
    """Compute per-value baseline accuracy (no steering hook).

    Results are cached to ``{output_dir}/baseline_accuracies.json``.  If the
    cache file already exists and ``force=False``, it is loaded directly.

    Returns
    -------
    dict mapping each value name → float accuracy in [0, 1].
    """
    cache_path = output_dir / "baseline_accuracies.json"

    if cache_path.exists() and not force:
        print(f"Loading cached baseline accuracies from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    print("Computing baseline accuracies (no steering)...")
    baseline: Dict[str, float] = {}

    for value in tqdm(CIRCUMPLEX_ORDER, desc="Baseline"):
        instances = eval_instances[value]
        if not instances:
            baseline[value] = 0.0
            continue
        results = [_score_instance(model_info, formatter, inst) for inst in instances]
        baseline[value] = float(np.mean([r["is_correct"] for r in results]))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline saved to {cache_path}")

    return baseline


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Build T matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_T_matrix(
    method: SteeringMethod,
    model_info: ModelInfo,
    eval_instances: Dict[str, List[EvalInstance]],
    formatter: PromptFormatter,
    baseline_accs: Dict[str, float],
    alpha: float,
) -> np.ndarray:
    """Build the 20×20 transfer matrix T for a single steering method.

    Outer loop: steering value A (20 iterations — one hook register/remove).
    Inner loop: eval value B (20 iterations per A).

    T[i, j] = steered_accuracy_on_B_j − baseline_accuracy_on_B_j
              when model is steered toward A_i.

    Returns
    -------
    np.ndarray of shape (20, 20), dtype float32.
    """
    print(f"\nLoading vectors for method '{method.name}' (layer {method.layer})...")
    vectors = method.load_vectors()

    T = np.zeros((len(CIRCUMPLEX_ORDER), len(CIRCUMPLEX_ORDER)), dtype=np.float32)

    for i, value_a in enumerate(tqdm(CIRCUMPLEX_ORDER, desc=f"[{method.name}] Steering A")):
        vector = vectors[value_a]
        handle = method.apply_hook(model_info, vector, alpha)

        try:
            for j, value_b in enumerate(CIRCUMPLEX_ORDER):
                instances = eval_instances[value_b]
                if not instances:
                    T[i, j] = 0.0
                    continue
                results = [
                    _score_instance(model_info, formatter, inst)
                    for inst in instances
                ]
                steered_acc = float(np.mean([r["is_correct"] for r in results]))
                T[i, j] = steered_acc - baseline_accs[value_b]
        finally:
            method.remove_hook(handle)

    return T


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2/3 — Output saving
# ─────────────────────────────────────────────────────────────────────────────

def save_method_outputs(
    method_name: str,
    T: np.ndarray,
    metrics_dict: dict,
    baseline_accs: Dict[str, float],
    output_dir: Path,
    alpha: float,
) -> Path:
    """Save all per-method artefacts under ``{output_dir}/{method_name}/``.

    Files written:
        T_matrix.npy
        T_matrix.json
        baseline_accuracies.json
        metrics.json
        T_heatmap.png
        bmd_bin_plot.png

    Returns
    -------
    Path to the method-specific output directory.
    """
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    # T_matrix.npy
    np.save(method_dir / "T_matrix.npy", T)

    # T_matrix.json
    T_json: Dict[str, Dict[str, float]] = {}
    for i, va in enumerate(CIRCUMPLEX_ORDER):
        T_json[va] = {}
        for j, vb in enumerate(CIRCUMPLEX_ORDER):
            T_json[va][vb] = float(T[i, j])
    with open(method_dir / "T_matrix.json", "w") as f:
        json.dump(T_json, f, indent=2)

    # baseline_accuracies.json (self-contained copy)
    with open(method_dir / "baseline_accuracies.json", "w") as f:
        json.dump(baseline_accs, f, indent=2)

    # metrics.json
    with open(method_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Visualisations
    plot_T_heatmap(T, method_name, alpha, method_dir / "T_heatmap.png")
    plot_bmd_bin(metrics_dict, method_name, method_dir / "bmd_bin_plot.png")

    print(f"Outputs for '{method_name}' saved to {method_dir}")
    return method_dir


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation 1 — T-matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

_HO_GROUP_NAMES = [
    "Openness to Change",
    "Self-Enhancement",
    "Conservation",
    "Self-Transcendence",
]
_HO_COLORS = ["#D4A017", "#F44336", "#1E88E5", "#4CAF50"]

# Short labels for the 20×20 axes (abbreviations to fit)
_SHORT_LABELS = [
    "SD-t", "SD-a", "Stim", "Hedo",
    "Ach", "Pw-d", "Pw-r", "Face",
    "Sec-p", "Sec-s", "Trad", "Con-r",
    "Con-i", "Hum", "Ben-d", "Ben-c",
    "Uni-c", "Uni-n", "Uni-t", "Uni-o",
]


def plot_T_heatmap(
    T: np.ndarray,
    method_name: str,
    alpha: float,
    save_path: Path,
) -> None:
    """Save a 20×20 heatmap of the T matrix.

    Features:
    - RdBu_r diverging colormap centred at 0.
    - Axes in CIRCUMPLEX_ORDER with short abbreviation labels.
    - White border rectangles around the four HO group blocks.
    - Cell value annotations (2dp) when figure is large enough.
    - Title: ``{method_name} | Transfer Matrix T[A→B] | α={alpha}``.
    """
    n = len(CIRCUMPLEX_ORDER)
    figsize = (16, 14)
    annotate = True  # always annotate; font size scales with figure

    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(T.min()), abs(T.max())) or 0.1
    im = ax.imshow(T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Δ Accuracy (steered − baseline)", fontsize=11)

    # Axis ticks and labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(_SHORT_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(_SHORT_LABELS, fontsize=8)
    ax.set_xlabel("Eval value B", fontsize=12)
    ax.set_ylabel("Steering value A", fontsize=12)
    ax.set_title(
        f"{method_name} | Transfer Matrix T[A→B] | α={alpha}",
        fontsize=13,
        pad=12,
    )

    # Cell annotations
    if annotate:
        font_size = max(4, int(figsize[0] * 0.45))
        for i in range(n):
            for j in range(n):
                val = T[i, j]
                color = "black" if abs(val) < vmax * 0.6 else "white"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=font_size, color=color,
                )

    # HO group block borders (white rectangles)
    for (start, end), color in zip(HO_BLOCK_BOUNDARIES, _HO_COLORS):
        size = end - start + 1
        rect_x = mpatches.FancyBboxPatch(
            (start - 0.5, start - 0.5), size, size,
            boxstyle="square,pad=0",
            linewidth=2.0,
            edgecolor="white",
            facecolor="none",
        )
        ax.add_patch(rect_x)

    # HO group legend
    legend_handles = [
        mpatches.Patch(color=c, label=n)
        for c, n in zip(_HO_COLORS, _HO_GROUP_NAMES)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.12, 1.0),
        fontsize=9,
        title="HO Group",
        title_fontsize=9,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  T_heatmap saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation 2 — BMD bin bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_bmd_bin(
    metrics_dict: dict,
    method_name: str,
    save_path: Path,
) -> None:
    """Save the Bin-wise Monotonic Decay bar chart.

    Features:
    - Bars coloured warm (red/orange) for k ≤ 4, grey for k = 5, cool
      (blue/teal) for k ≥ 6, consistent with theoretical R sign.
    - Dashed cosine reference line scaled to the bar height range.
    - Annotated bar values (2dp).
    - Horizontal dashed line at y = 0.
    - X labels: ``k=N\\n(N×18°)``.
    """
    bmd_rho = metrics_dict["BMD_rho"]
    raw_bin_means = metrics_dict["bin_means"]
    bin_means = {int(k): v for k, v in raw_bin_means.items()}

    ks = list(range(1, 11))
    mu_values = [bin_means[k] for k in ks]

    # Bar colours by theoretical sign
    warm = "#E53935"
    warm_light = "#FF7043"
    neutral = "#9E9E9E"
    cool_light = "#42A5F5"
    cool = "#1565C0"

    bar_colors = []
    for k in ks:
        if k <= 2:
            bar_colors.append(warm)
        elif k <= 4:
            bar_colors.append(warm_light)
        elif k == 5:
            bar_colors.append(neutral)
        elif k <= 7:
            bar_colors.append(cool_light)
        else:
            bar_colors.append(cool)

    fig, ax = plt.subplots(figsize=(11, 5))

    bars = ax.bar(ks, mu_values, color=bar_colors, edgecolor="white", linewidth=0.8, alpha=0.9)

    # Cosine reference line
    cos_vals = [np.cos(k * np.pi / 10) for k in ks]
    max_bar = max(abs(v) for v in mu_values) if any(mu_values) else 1.0
    max_cos = max(abs(c) for c in cos_vals)
    scale = max_bar / max_cos if max_cos > 0 else 1.0
    scaled_cos = [c * scale for c in cos_vals]
    ax.plot(
        ks, scaled_cos,
        linestyle="--", color="#212121", linewidth=1.5, alpha=0.7,
        label=f"Cosine reference (scaled ×{scale:.3f})",
    )

    # y = 0 reference
    ax.axhline(0, color="#424242", linewidth=0.8, linestyle="--", alpha=0.6)

    # Bar value annotations
    for bar, val in zip(bars, mu_values):
        offset = 0.002 if val >= 0 else -0.004
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:.2f}",
            ha="center", va=va, fontsize=9,
        )

    # X-axis labels
    ax.set_xticks(ks)
    ax.set_xticklabels([f"k={k}\n({k*18}°)" for k in ks], fontsize=9)
    ax.set_xlabel("Angular distance k", fontsize=11)
    ax.set_ylabel("Mean Δ Accuracy", fontsize=11)
    ax.set_title(
        f"{method_name} | Bin-wise Transfer by Angular Distance | "
        f"BMD-ρ={bmd_rho:.3f}",
        fontsize=12,
    )
    ax.legend(fontsize=9)

    # Colour legend for theoretical relationship
    legend_handles = [
        mpatches.Patch(color=warm, label="k=1–2 (strong positive, R>0.8)"),
        mpatches.Patch(color=warm_light, label="k=3–4 (positive, R>0)"),
        mpatches.Patch(color=neutral, label="k=5 (orthogonal, R≈0)"),
        mpatches.Patch(color=cool_light, label="k=6–7 (negative, R<0)"),
        mpatches.Patch(color=cool, label="k=8–10 (strong negative, R<−0.5)"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=8,
        loc="upper right",
        title="Theoretical R",
        title_fontsize=8,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  bmd_bin_plot saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-method comparison table
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_table(output_dir: Path) -> None:
    """Scan method sub-directories for metrics.json and generate a comparison.

    Outputs
    -------
    comparison_table.json  — list of dicts sorted by CFS descending.
    comparison_table.png   — formatted matplotlib table figure.
    """
    records = []
    for method_dir in sorted(output_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        metrics_path = method_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r") as f:
            m = json.load(f)
        records.append(
            {
                "method": m.get("method", method_dir.name),
                "CTC_rho": m.get("CTC_rho", float("nan")),
                "BMD_rho": m.get("BMD_rho", float("nan")),
                "AOG": m.get("AOG", float("nan")),
                "CFS": m.get("CFS", float("nan")),
            }
        )

    if not records:
        print("No metrics.json files found; skipping comparison table.")
        return

    records.sort(key=lambda r: r["CFS"] if not np.isnan(r["CFS"]) else -999, reverse=True)

    with open(output_dir / "comparison_table.json", "w") as f:
        json.dump(records, f, indent=2)

    _plot_comparison_table(records, output_dir / "comparison_table.png")
    print(f"Comparison table saved to {output_dir}")


def _diverging_cell_color(
    value: float,
    null: float,
    vmin: float,
    vmax: float,
) -> tuple:
    """Map a scalar to an RGBA colour on a red-white-green scale."""
    if np.isnan(value):
        return (0.9, 0.9, 0.9, 1.0)
    span = max(vmax - null, null - vmin, 1e-9)
    t = (value - null) / span  # -1 … +1
    t = max(-1.0, min(1.0, t))
    if t >= 0:
        # white → green
        r = 1.0 - 0.6 * t
        g = 1.0
        b = 1.0 - 0.6 * t
    else:
        # white → red
        r = 1.0
        g = 1.0 + 0.6 * t
        b = 1.0 + 0.6 * t
    return (r, g, b, 1.0)


def _plot_comparison_table(records: list, save_path: Path) -> None:
    """Render the comparison table as a matplotlib figure."""
    columns = ["Method", "CTC-ρ", "BMD-ρ", "AOG (pp)", "CFS"]
    null_vals = {"CTC-ρ": 0.0, "BMD-ρ": 0.0, "AOG (pp)": 0.0, "CFS": 0.5}

    # Find best value per numeric column for bolding
    keys = ["CTC_rho", "BMD_rho", "AOG", "CFS"]
    col_keys = dict(zip(columns[1:], keys))
    best = {}
    for col, key in col_keys.items():
        vals = [r[key] for r in records if not np.isnan(r[key])]
        best[col] = max(vals) if vals else None

    cell_texts = []
    cell_colors = []
    for r in records:
        row_texts = [r["method"]]
        row_colors = [(0.95, 0.95, 0.95, 1.0)]  # method name cell: light grey
        for col, key in col_keys.items():
            val = r[key]
            fmt = f"{val:.3f}" if not np.isnan(val) else "—"
            if best[col] is not None and not np.isnan(val) and abs(val - best[col]) < 1e-9:
                fmt = f"**{fmt}**"
            row_texts.append(fmt)

            null = null_vals[col]
            if col in ("CTC-ρ", "BMD-ρ"):
                vmin, vmax = -1.0, 1.0
            elif col == "AOG (pp)":
                vmin, vmax = -0.30, 0.30
            else:  # CFS
                vmin, vmax = 0.0, 1.0
            row_colors.append(_diverging_cell_color(val, null, vmin, vmax))

        cell_texts.append(row_texts)
        cell_colors.append(row_colors)

    n_rows = len(records)
    n_cols = len(columns)
    fig_h = max(2.0, 0.5 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_texts,
        colLabels=columns,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    ax.set_title(
        "Cross-Method Comparison (sorted by CFS ↓)",
        fontsize=13,
        pad=10,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  comparison_table.png saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    config: TransferExperimentConfig,
    methods: List[SteeringMethod],
) -> None:
    """Run the full cross-value transfer experiment.

    Steps:
    1. Resolve model_name from config or from the first method that provides it.
    2. Load model once (shared across all methods).
    3. Load eval instances from the Touché CSV.
    4. Compute/load baseline accuracies.
    5. For each method:
       a. Check if outputs already exist (skip if force_recompute=False).
       b. Build T matrix.
       c. Compute all metrics.
       d. Save outputs.
    6. Generate the cross-method comparison table.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve model name ────────────────────────────────────────────────────
    model_name = config.model_name
    if not model_name:
        # Try to read from the first method that exposes .model_name
        for m in methods:
            if hasattr(m, "model_name") and m.model_name != "unknown":
                model_name = m.model_name
                print(f"Using model_name from method '{m.name}': {model_name}")
                break
    if not model_name:
        raise ValueError(
            "model_name could not be determined. "
            "Pass --model_name or ensure the run_dir contains a valid config.json."
        )

    # ── Load model (once) ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    model_info = load_model(model_name, device=config.device)
    formatter = PromptFormatter(model_info.tokenizer, model_info.is_instruct)

    # ── Load eval instances ───────────────────────────────────────────────────
    eval_instances = load_eval_instances(config)

    # ── Baseline (cached, shared across methods) ──────────────────────────────
    baseline_accs = compute_baseline(
        model_info, eval_instances, formatter, output_dir, force=config.force_recompute
    )

    # ── Load R matrix for metrics ─────────────────────────────────────────────
    R_matrix = load_R_matrix(Path(config.relations_path))

    # ── Per-method evaluation ─────────────────────────────────────────────────
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method.name}  |  layer: {method.layer}  |  α: {config.alpha}")
        print(f"{'='*60}")

        method_dir = output_dir / method.name
        metrics_path = method_dir / "metrics.json"

        if metrics_path.exists() and not config.force_recompute:
            print(f"Outputs already exist at {method_dir}. Skipping (use --force_recompute to redo).")
            continue

        T = build_T_matrix(
            method, model_info, eval_instances, formatter, baseline_accs, config.alpha
        )

        metrics_dict = compute_all_metrics(
            T, R_matrix,
            method_name=method.name,
            alpha=config.alpha,
        )

        save_method_outputs(
            method.name, T, metrics_dict, baseline_accs, output_dir, config.alpha
        )

        print(f"\nMetrics summary for '{method.name}':")
        print(f"  CTC-ρ = {metrics_dict['CTC_rho']:.4f}  (p={metrics_dict['CTC_pvalue']:.4f})")
        print(f"  BMD-ρ = {metrics_dict['BMD_rho']:.4f}  (p={metrics_dict['BMD_pvalue']:.4f})")
        print(f"  AOG   = {metrics_dict['AOG']:.4f}")
        print(f"  CFS   = {metrics_dict['CFS']:.4f}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating cross-method comparison table...")
    generate_comparison_table(output_dir)

    print(f"\nExperiment complete. Results in: {output_dir}")
