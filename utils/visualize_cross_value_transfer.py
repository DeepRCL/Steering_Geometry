"""Visualize geometry fidelity against cross-value transfer magnitude.

This mirrors ``utils/visualize.py`` but uses cross-value-transfer metrics:

- y-axis: normalized centered TWTM, a relative magnitude-sensitive score.
- x-axis 1: representation geometry fidelity rho_T.
- x-axis 2: residualized BMD from the transfer experiment.

The script reads ``experiments-outputs_final200/comparison_table.json`` and
writes PNG/PDF figures into the same directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("experiments-outputs_final200")
COMPARISON_PATH = OUTPUT_DIR / "comparison_table.json"
ALL_METHODS_PATH = OUTPUT_DIR / "comparison_table_all_methods.json"


METHOD_INFO = {
    # Folder names stay as code keys. Plot labels follow paper naming:
    # qwenscope_finetuned_alpha_4.0 -> SAS, odesteer_vectors_alpha20 -> ODESteer.
    "caa": {"label": "CAA", "rhoT": 0.4606, "type": "distribution"},
    "spherical": {"label": "SphericalSteer", "rhoT": 0.3962, "type": "distribution"},
    "odesteer_vectors_alpha20": {"label": "ODESteer", "rhoT": 0.2730, "type": "distribution"},
    # "odesteer_vectors" is the alpha=1 run; omitted from the paper-style plot.
    "qwenscope_l15_k100_finetuned_alpha_4.0": {"label": "SAS", "rhoT": 0.5164, "type": "distribution"},
    "qwenscope_finetuned_alpha_4.0": {"label": "SAS256", "rhoT": 0.5069, "type": "distribution"},
    # "qwenscope-v2" is the older original QwenScopeCAA run; omitted in favor
    # of the finetuned alpha=4 SAS result.
    # "qwenscope" is another older QwenScopeCAA run; omitted from the paper-style plot.
    # "sparsecaa" is intentionally omitted; SAS is the plotted sparse method name.
    # "sparsecaa": {"label": "SparseCAA", "rhoT": 0.4584, "type": "distribution"},
    "bipo": {"label": "BiPO", "rhoT": 0.1188, "type": "behavior"},
    "cold_steer": {"label": "Cold-Steer", "rhoT": 0.0265, "type": "behavior"},
    "llm_steering_opt": {"label": "OPT", "rhoT": 0.0231, "type": "behavior"},
}


COLOR_MAP = {
    "distribution": "#185FA5",
    "behavior": "#D85A30",
}


LABEL_OFFSETS = {
    "rhoT": {
        "CAA": (14, 10),
        "SAS": (-56, -18),
        "SphericalSteer": (14, -18),
        "ODESteer": (14, 10),
        "BiPO": (14, 12),
        "Cold-Steer": (14, 12),
        "OPT": (14, -16),
    },
    "resid_bmd": {
        "CAA": (14, 12),
        "ODESteer": (14, -18),
        "SphericalSteer": (14, 10),
        "SAS": (14, 12),
        "BiPO": (14, -14),
        "Cold-Steer": (14, 12),
        "OPT": (14, -16),
    },
}


def _load_points() -> list[dict]:
    records = _load_metric_records()

    points = []
    for record in records:
        method = record["method"]
        if method not in METHOD_INFO:
            continue
        info = METHOD_INFO[method]
        twtm_c = record.get("resid_centered_TWTM", record.get("centered_TWTM", 0.0))
        points.append(
            {
                **info,
                "method": method,
                "twtm_c": twtm_c,
                "normalized_twtm_c": twtm_c,
                "ctc_rho": record.get("CTC_rho", np.nan),
                "aog": record.get("AOG", np.nan),
                "resid_bmd": record.get("resid_BMD_rho", np.nan),
                "resid_ctc_rho": record.get("resid_CTC_rho", np.nan),
                "resid_aog": record.get("resid_AOG", np.nan),
                "resid_beta": record.get("resid_theory_beta", np.nan),
                "stability_exp": record.get("stability_exp_neg_error", np.nan),
                "stability_inverse": record.get("stability_inverse_error", np.nan),
                "resid_stability_exp": record.get("resid_stability_exp_neg_error", np.nan),
                "resid_stability_inverse": record.get("resid_stability_inverse_error", np.nan),
                "resid_cfs": record.get("resid_CFS", np.nan),
            }
        )
    scale = sum(point["twtm_c"] for point in points)
    if abs(scale) < 1e-12:
        scale = 1.0
    for point in points:
        point["normalized_twtm_c"] /= scale
    return points


def _load_metric_records() -> list[dict]:
    """Load metrics-rich records; never depend on the reduced paper table."""
    records = []
    for method_dir in sorted(p for p in OUTPUT_DIR.iterdir() if p.is_dir()):
        metrics_path = method_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        residualized = metrics.get("residualized", {})
        raw_regression = metrics.get("theory_regression", {})
        resid_regression = residualized.get("theory_regression", {})
        raw_stability = metrics.get("self_scaled_transfer_stability", {})
        resid_stability = residualized.get("self_scaled_transfer_stability", {})
        raw_summary = metrics.get("transfer_summary", {})
        records.append(
            {
                "method": method_dir.name,
                "reported_method": metrics.get("method", method_dir.name),
                "alpha": metrics.get("alpha", np.nan),
                "mean_transfer": raw_summary.get("off_diagonal_mean_transfer", np.nan),
                "positive_fraction": raw_summary.get("off_diagonal_positive_fraction", np.nan),
                "CTC_rho": metrics.get("CTC_rho", np.nan),
                "BMD_rho": metrics.get("BMD_rho", np.nan),
                "AOG": metrics.get("AOG", np.nan),
                "TWTM": metrics.get("TWTM", np.nan),
                "centered_TWTM": metrics.get("centered_TWTM", np.nan),
                "theory_beta": raw_regression.get("beta", np.nan),
                "stability_mean_relative_error": raw_stability.get("mean_relative_error", np.nan),
                "stability_exp_neg_error": raw_stability.get("stability_exp_neg_error", np.nan),
                "stability_inverse_error": raw_stability.get("stability_inverse_error", np.nan),
                "CFS": metrics.get("CFS", np.nan),
                "resid_CTC_rho": residualized.get("CTC_rho", np.nan),
                "resid_BMD_rho": residualized.get("BMD_rho", np.nan),
                "resid_AOG": residualized.get("AOG", np.nan),
                "resid_TWTM": residualized.get("TWTM", np.nan),
                "resid_centered_TWTM": residualized.get(
                    "centered_TWTM",
                    metrics.get("centered_TWTM", 0.0),
                ),
                "resid_theory_beta": resid_regression.get("beta", np.nan),
                "resid_stability_mean_relative_error": resid_stability.get("mean_relative_error", np.nan),
                "resid_stability_exp_neg_error": resid_stability.get("stability_exp_neg_error", np.nan),
                "resid_stability_inverse_error": resid_stability.get("stability_inverse_error", np.nan),
                "resid_CFS": residualized.get("CFS", np.nan),
            }
        )
    with ALL_METHODS_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return records


def _write_paper_comparison_table(points: list[dict]) -> None:
    """Write a comparison table using exactly the plotted method set.

    This intentionally differs from an all-results audit table: SparseCAA and
    the ODESteer alpha=1 run are excluded here to match the paper-style figures.
    """
    records = sorted(points, key=lambda p: p["normalized_twtm_c"], reverse=True)
    with (OUTPUT_DIR / "comparison_table_paper.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with (OUTPUT_DIR / "comparison_table.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    columns = [
        "Method", "rhoT", "Norm TWTM-c", "TWTM-c",
        "CTC-rho", "AOG", "Stab exp", "Stab inv",
        "Resid CTC-rho", "Resid BMD", "Resid AOG",
        "Resid Stab exp", "Resid Stab inv", "Type",
    ]
    rows = [
        [
            p["label"],
            f"{p['rhoT']:.4f}",
            f"{p['normalized_twtm_c']:.4f}",
            f"{p['twtm_c']:.5f}",
            f"{p['ctc_rho']:.3f}",
            f"{p['aog']:.5f}",
            f"{p['stability_exp']:.3f}",
            f"{p['stability_inverse']:.3f}",
            f"{p['resid_ctc_rho']:.3f}",
            f"{p['resid_bmd']:.3f}",
            f"{p['resid_aog']:.5f}",
            f"{p['resid_stability_exp']:.3f}",
            f"{p['resid_stability_inverse']:.3f}",
            p["type"],
        ]
        for p in records
    ]

    fig, ax = plt.subplots(figsize=(18.5, max(2.2, 0.42 * len(rows) + 1.1)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.12, 1.35)
    ax.set_title("Paper-Style Cross-Value Transfer Comparison", fontsize=13, pad=10)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_table_paper.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "comparison_table.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _write_full_available_results_table() -> None:
    """Write an all-runs table from every result folder with metrics.json.

    This is an audit table, not the paper-style method subset. It includes
    alternate runs such as different alphas, QwenScope k values, and older
    baselines whenever their result folders are present under OUTPUT_DIR.
    """
    records = []
    for method_dir in sorted(p for p in OUTPUT_DIR.iterdir() if p.is_dir()):
        metrics_path = method_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        metadata_path = method_dir / "run_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

        residualized = metrics.get("residualized", {})
        raw_regression = metrics.get("theory_regression", {})
        resid_regression = residualized.get("theory_regression", {})
        raw_summary = metrics.get("transfer_summary", {})
        resid_summary = residualized.get("transfer_summary", {})
        cache_metadata = metadata.get("method_cache_metadata", {})

        records.append(
            {
                "folder": method_dir.name,
                "reported_method": metrics.get("method", method_dir.name),
                "model_name": metadata.get("model_name", ""),
                "alpha": metrics.get("alpha"),
                "layer": metadata.get("layer", cache_metadata.get("layer")),
                "k": cache_metadata.get("k"),
                "mean_delta": raw_summary.get("off_diagonal_mean_transfer"),
                "diag_delta": raw_summary.get("diagonal_mean_transfer"),
                "positive_fraction": raw_summary.get("off_diagonal_positive_fraction"),
                "CTC_rho": metrics.get("CTC_rho"),
                "BMD_rho": metrics.get("BMD_rho"),
                "AOG": metrics.get("AOG"),
                "TWTM": metrics.get("TWTM"),
                "centered_TWTM": metrics.get("centered_TWTM"),
                "theory_beta": raw_regression.get("beta"),
                "theory_r_squared": raw_regression.get("r_squared"),
                "stability_mean_relative_error": metrics.get(
                    "self_scaled_transfer_stability", {}
                ).get("mean_relative_error"),
                "stability_exp_neg_error": metrics.get(
                    "self_scaled_transfer_stability", {}
                ).get("stability_exp_neg_error"),
                "stability_inverse_error": metrics.get(
                    "self_scaled_transfer_stability", {}
                ).get("stability_inverse_error"),
                "CFS": metrics.get("CFS"),
                "CTC_norm": metrics.get("CTC_norm"),
                "BMD_norm": metrics.get("BMD_norm"),
                "AOG_norm": metrics.get("AOG_norm"),
                "resid_mean_delta": resid_summary.get("off_diagonal_mean_transfer"),
                "resid_positive_fraction": resid_summary.get("off_diagonal_positive_fraction"),
                "resid_CTC_rho": residualized.get("CTC_rho"),
                "resid_BMD_rho": residualized.get("BMD_rho"),
                "resid_AOG": residualized.get("AOG"),
                "resid_TWTM": residualized.get("TWTM"),
                "resid_centered_TWTM": residualized.get("centered_TWTM"),
                "resid_theory_beta": resid_regression.get("beta"),
                "resid_theory_r_squared": resid_regression.get("r_squared"),
                "resid_stability_mean_relative_error": residualized.get(
                    "self_scaled_transfer_stability", {}
                ).get("mean_relative_error"),
                "resid_stability_exp_neg_error": residualized.get(
                    "self_scaled_transfer_stability", {}
                ).get("stability_exp_neg_error"),
                "resid_stability_inverse_error": residualized.get(
                    "self_scaled_transfer_stability", {}
                ).get("stability_inverse_error"),
                "resid_CFS": residualized.get("CFS"),
                "resid_CTC_norm": residualized.get("CTC_norm"),
                "resid_BMD_norm": residualized.get("BMD_norm"),
                "resid_AOG_norm": residualized.get("AOG_norm"),
            }
        )

    scale = sum(
        record["resid_centered_TWTM"]
        for record in records
        if isinstance(record.get("resid_centered_TWTM"), (int, float))
    )
    if abs(scale) < 1e-12:
        scale = 1.0
    for record in records:
        value = record.get("resid_centered_TWTM")
        record["resid_normalized_TWTM_c"] = (
            float(value) / scale if isinstance(value, (int, float)) else np.nan
        )

    records.sort(
        key=lambda record: (
            record["resid_centered_TWTM"]
            if isinstance(record.get("resid_centered_TWTM"), (int, float))
            else -999.0
        ),
        reverse=True,
    )

    with (OUTPUT_DIR / "full_available_results_table.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    columns = [
        ("folder", "Folder"),
        ("reported_method", "Method"),
        ("alpha", "alpha"),
        ("layer", "Layer"),
        ("k", "k"),
        ("mean_delta", "Mean d"),
        ("diag_delta", "Diag d"),
        ("positive_fraction", "Pos frac"),
        ("CTC_rho", "CTC"),
        ("BMD_rho", "BMD"),
        ("AOG", "AOG"),
        ("centered_TWTM", "TWTM-c"),
        ("theory_beta", "beta"),
        ("stability_exp_neg_error", "Stab exp"),
        ("stability_inverse_error", "Stab inv"),
        ("CFS", "CFS"),
        ("resid_CTC_rho", "Resid CTC"),
        ("resid_BMD_rho", "Resid BMD"),
        ("resid_AOG", "Resid AOG"),
        ("resid_centered_TWTM", "Resid TWTM-c"),
        ("resid_normalized_TWTM_c", "Resid Norm"),
        ("resid_theory_beta", "Resid beta"),
        ("resid_stability_exp_neg_error", "Resid Stab exp"),
        ("resid_stability_inverse_error", "Resid Stab inv"),
        ("resid_CFS", "Resid CFS"),
    ]

    def format_cell(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if np.isnan(value):
                return ""
            return f"{value:.6f}"
        return str(value)

    markdown_lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in records:
        markdown_lines.append(
            "| "
            + " | ".join(format_cell(record.get(key)) for key, _ in columns)
            + " |"
        )
    with (OUTPUT_DIR / "full_available_results_table.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines) + "\n")

    compact_columns = [
        ("folder", "Folder"),
        ("alpha", "alpha"),
        ("mean_delta", "Mean d"),
        ("positive_fraction", "Pos"),
        ("CTC_rho", "CTC"),
        ("BMD_rho", "BMD"),
        ("AOG", "AOG"),
        ("centered_TWTM", "TWTM-c"),
        ("resid_CTC_rho", "Resid CTC"),
        ("resid_BMD_rho", "Resid BMD"),
        ("resid_AOG", "Resid AOG"),
        ("resid_centered_TWTM", "Resid TWTM-c"),
        ("resid_normalized_TWTM_c", "Norm"),
        ("resid_theory_beta", "Resid beta"),
        ("stability_exp_neg_error", "Stab exp"),
        ("stability_inverse_error", "Stab inv"),
        ("resid_stability_exp_neg_error", "Resid Stab exp"),
        ("resid_stability_inverse_error", "Resid Stab inv"),
        ("resid_CFS", "Resid CFS"),
    ]
    table_rows = [
        [format_cell(record.get(key)) for key, _ in compact_columns]
        for record in records
    ]

    fig, ax = plt.subplots(figsize=(15, max(4.0, 0.38 * len(table_rows) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=[label for _, label in compact_columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.24)
    ax.set_title("All Available Cross-Value Transfer Results", fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_available_results_table.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "full_available_results_table.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot(points: list[dict], x_key: str, x_label: str, output_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.3))

    for point in points:
        color = COLOR_MAP[point["type"]]
        ax.scatter(
            point[x_key],
            point["normalized_twtm_c"],
            color=color,
            edgecolors=color,
            facecolors="white" if point["type"] == "behavior" else color,
            s=120,
            linewidths=2.2,
            zorder=3,
        )
        dx, dy = LABEL_OFFSETS.get(x_key, {}).get(point["label"], (12, 10))
        ax.annotate(
            point["label"],
            xy=(point[x_key], point["normalized_twtm_c"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            color=color,
            va="center",
            ha="left" if dx >= 0 else "right",
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 1.0,
                "alpha": 0.85,
                "shrinkA": 2,
                "shrinkB": 5,
            },
            zorder=4,
        )

    ax.set_axisbelow(True)
    ax.grid(color="#e0e0e0", linewidth=0.7, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel("Normalized TWTM-c", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)

    xs = [p[x_key] for p in points if not np.isnan(p[x_key])]
    ys = [p["normalized_twtm_c"] for p in points if not np.isnan(p["normalized_twtm_c"])]
    x_pad = max(0.04, (max(xs) - min(xs)) * 0.12)
    y_pad = max(0.08, (max(ys) - min(ys)) * 0.18)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad * 2.5)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_MAP["distribution"], label="Distribution-driven"),
        mpatches.Patch(facecolor=COLOR_MAP["behavior"], label="Behavior-centric"),
    ]
    ax.legend(handles=legend_handles, fontsize=13, frameon=False, loc="upper left")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{output_stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / f"{output_stem}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_rhot_metric_chart(
    points: list[dict],
    metric_key: str,
    metric_label: str,
    output_stem: str,
    output_dir: Path,
) -> None:
    plot_points = [
        p for p in points
        if not np.isnan(p.get("rhoT", np.nan)) and not np.isnan(p.get(metric_key, np.nan))
    ]
    if not plot_points:
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.3))

    for point in plot_points:
        color = COLOR_MAP[point["type"]]
        ax.scatter(
            point["rhoT"],
            point[metric_key],
            color=color,
            edgecolors=color,
            facecolors="white" if point["type"] == "behavior" else color,
            s=120,
            linewidths=2.2,
            zorder=3,
        )
        dx, dy = LABEL_OFFSETS.get("rhoT", {}).get(point["label"], (12, 10))
        ax.annotate(
            point["label"],
            xy=(point["rhoT"], point[metric_key]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            color=color,
            va="center",
            ha="left" if dx >= 0 else "right",
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 1.0,
                "alpha": 0.85,
                "shrinkA": 2,
                "shrinkB": 5,
            },
            zorder=4,
        )

    ax.set_axisbelow(True)
    ax.grid(color="#e0e0e0", linewidth=0.7, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xlabel(r"Geometric fidelity  $\rho_T$", fontsize=15)
    ax.set_ylabel(metric_label, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)

    xs = [p["rhoT"] for p in plot_points]
    ys = [p[metric_key] for p in plot_points]
    x_pad = max(0.04, (max(xs) - min(xs)) * 0.12)
    y_range = max(ys) - min(ys)
    y_pad = max(0.02, y_range * 0.18)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad * 2.5)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_MAP["distribution"], label="Distribution-driven"),
        mpatches.Patch(facecolor=COLOR_MAP["behavior"], label="Behavior-centric"),
    ]
    ax.legend(handles=legend_handles, fontsize=13, frameon=False, loc="upper left")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_dir / f"{output_stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_rhot_metric_suite(points: list[dict]) -> None:
    output_dir = OUTPUT_DIR / "rhoT_metric_charts"
    specs = [
        ("ctc_rho", "CTC-rho", "rhoT_vs_CTC_rho"),
        ("resid_ctc_rho", "Residualized CTC-rho", "rhoT_vs_resid_CTC_rho"),
        ("resid_aog", "Residualized AOG", "rhoT_vs_resid_AOG"),
        ("resid_beta", r"Residualized theory slope $\beta$", "rhoT_vs_resid_beta"),
        ("stability_exp", "Self-scaled stability exp(-error)", "rhoT_vs_stability_exp"),
        ("stability_inverse", "Self-scaled stability 1/(1+error)", "rhoT_vs_stability_inverse"),
        (
            "resid_stability_exp",
            "Residualized self-scaled stability exp(-error)",
            "rhoT_vs_resid_stability_exp",
        ),
        (
            "resid_stability_inverse",
            "Residualized self-scaled stability 1/(1+error)",
            "rhoT_vs_resid_stability_inverse",
        ),
    ]
    for metric_key, metric_label, output_stem in specs:
        _plot_rhot_metric_chart(
            points,
            metric_key=metric_key,
            metric_label=metric_label,
            output_stem=output_stem,
            output_dir=output_dir,
        )


def _plot_resid_bmd_by_method(points: list[dict]) -> None:
    grouped = []
    for method_type in ("distribution", "behavior"):
        group_points = [
            p for p in points
            if p["type"] == method_type and not np.isnan(p["resid_bmd"])
        ]
        group_points.sort(key=lambda p: p["resid_bmd"], reverse=True)
        grouped.append((method_type, group_points))

    xs = []
    labels = []
    values = []
    colors = []
    x = 0.0
    group_centers = {}
    for method_type, group_points in grouped:
        start = x
        for point in group_points:
            xs.append(x)
            labels.append(point["label"])
            values.append(point["resid_bmd"])
            colors.append(COLOR_MAP[method_type])
            x += 1.0
        if group_points:
            group_centers[method_type] = (start + x - 1.0) / 2.0
            x += 0.85

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bars = ax.bar(
        xs,
        values,
        width=0.68,
        color=colors,
        edgecolor=colors,
        linewidth=1.5,
        alpha=0.92,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
        )

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.7, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_ylabel("Residualized BMD", fontsize=15)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)

    for method_type, center in group_centers.items():
        ax.text(
            center,
            -0.20,
            "Distribution-driven" if method_type == "distribution" else "Behavior-centric",
            ha="center",
            va="top",
            fontsize=13,
            color=COLOR_MAP[method_type],
            transform=ax.get_xaxis_transform(),
        )

    ax.set_ylim(0, max(values) + 0.18)
    ax.set_title("Residualized BMD by Method Type", fontsize=15, pad=10)

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_MAP["distribution"], label="Distribution-driven"),
        mpatches.Patch(facecolor=COLOR_MAP["behavior"], label="Behavior-centric"),
    ]
    ax.legend(handles=legend_handles, fontsize=12, frameon=False, loc="upper right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "resid_BMD_by_method_type.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "resid_BMD_by_method_type.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    points = _load_points()
    _write_paper_comparison_table(points)
    _write_full_available_results_table()
    _plot(
        points,
        "rhoT",
        r"Geometric fidelity  $\rho_T$",
        "rhoT_vs_normalized_TWTM_c",
    )
    _plot(
        points,
        "resid_bmd",
        "Residualized BMD",
        "resid_BMD_vs_normalized_TWTM_c",
    )
    _plot_resid_bmd_by_method(points)
    _plot_rhot_metric_suite(points)
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
