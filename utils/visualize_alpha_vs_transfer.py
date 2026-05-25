"""Table and scatter plots: steering alpha vs cross-value transfer metrics.

Reads ``experiments-outputs_final200/*/metrics.json`` and writes:
  - alpha_vs_transfer_table.json / .md / .csv
  - alpha_vs_transfer_metrics.png / .pdf
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUTPUT_DIR = Path("experiments-outputs_final200")

METHOD_LABELS = {
    "caa": "CAA",
    "spherical": "SphericalSteer",
    "bipo": "BiPO",
    "sparsecaa": "SparseCAA",
    "qwenscope": "QwenScope (v1)",
    "qwenscope-v2": "SAS",
    "odesteer_vectors": "ODESteer (α=1)",
    "odesteer_vectors_alpha20": "ODESteer (α=20)",
    "cold_steer": "Cold-Steer",
    "llm_steering_opt": "OPT",
}


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    for method_dir in sorted(p for p in OUTPUT_DIR.iterdir() if p.is_dir()):
        metrics_path = method_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        resid = metrics.get("residualized", {})
        folder = method_dir.name
        rows.append(
            {
                "folder": folder,
                "method": METHOD_LABELS.get(folder, folder),
                "alpha": float(metrics.get("alpha", float("nan"))),
                "twtm_c": float(
                    resid.get("centered_TWTM", metrics.get("centered_TWTM", float("nan")))
                ),
                "bmd_rho": float(resid.get("BMD_rho", metrics.get("BMD_rho", float("nan")))),
                "mean_transfer": float(
                    resid.get("transfer_summary", {}).get(
                        "off_diagonal_mean_transfer",
                        metrics.get("transfer_summary", {}).get(
                            "off_diagonal_mean_transfer", float("nan")
                        ),
                    )
                ),
            }
        )
    rows.sort(key=lambda r: r["alpha"])
    return rows


def _correlation_summary(rows: list[dict], x_key: str, y_key: str) -> dict:
    xs = np.array([r[x_key] for r in rows], dtype=float)
    ys = np.array([r[y_key] for r in rows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 3:
        return {"n": int(len(xs)), "pearson_r": float("nan"), "spearman_rho": float("nan")}
    pearson_r, pearson_p = stats.pearsonr(xs, ys)
    spearman_rho, spearman_p = stats.spearmanr(xs, ys)
    return {
        "n": int(len(xs)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def _write_table(rows: list[dict]) -> None:
    json_path = OUTPUT_DIR / "alpha_vs_transfer_table.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    md_path = OUTPUT_DIR / "alpha_vs_transfer_table.md"
    header = "| Method | Folder | α | Resid. centered TWTM | Resid. BMD ρ | Off-diag. mean transfer |"
    sep = "| --- | --- | ---: | ---: | ---: | ---: |"
    body = [
        f"| {r['method']} | `{r['folder']}` | {r['alpha']:.1f} | {r['twtm_c']:.5f} | {r['bmd_rho']:.3f} | {r['mean_transfer']:.4f} |"
        for r in rows
    ]
    corr_twtm = _correlation_summary(rows, "alpha", "twtm_c")
    corr_bmd = _correlation_summary(rows, "alpha", "bmd_rho")
    footer = [
        "",
        "**Cross-method correlations (n=10 methods; α not comparable across methods):**",
        f"- α vs resid. centered TWTM: Pearson r = {corr_twtm['pearson_r']:.3f} (p = {corr_twtm['pearson_p']:.3f}), "
        f"Spearman ρ = {corr_twtm['spearman_rho']:.3f} (p = {corr_twtm['spearman_p']:.3f})",
        f"- α vs resid. BMD ρ: Pearson r = {corr_bmd['pearson_r']:.3f} (p = {corr_bmd['pearson_p']:.3f}), "
        f"Spearman ρ = {corr_bmd['spearman_rho']:.3f} (p = {corr_bmd['spearman_p']:.3f})",
    ]
    md_path.write_text("\n".join([header, sep, *body, *footer]) + "\n", encoding="utf-8")

    csv_path = OUTPUT_DIR / "alpha_vs_transfer_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "folder", "alpha", "twtm_c", "bmd_rho", "mean_transfer"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict]) -> None:
    alphas = np.array([r["alpha"] for r in rows])
    twtm = np.array([r["twtm_c"] for r in rows])
    bmd = np.array([r["bmd_rho"] for r in rows])
    labels = [r["method"] for r in rows]

    corr_twtm = _correlation_summary(rows, "alpha", "twtm_c")
    corr_bmd = _correlation_summary(rows, "alpha", "bmd_rho")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    for ax, ys, ylabel, corr, color in [
        (axes[0], twtm, "Residualized centered TWTM", corr_twtm, "#185FA5"),
        (axes[1], bmd, "Residualized BMD ρ", corr_bmd, "#D85A30"),
    ]:
        ax.scatter(alphas, ys, s=90, c=color, edgecolors="white", linewidths=0.8, zorder=3)
        for x, y, label in zip(alphas, ys, labels):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                alpha=0.9,
            )
        if len(alphas) >= 2 and np.std(alphas) > 0:
            coef = np.polyfit(alphas, ys, 1)
            x_line = np.linspace(alphas.min(), alphas.max(), 50)
            ax.plot(x_line, np.polyval(coef, x_line), "--", color="#424242", alpha=0.45, linewidth=1.2)
        ax.set_xlabel("Steering strength α")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{ylabel}\n"
            f"Pearson r = {corr['pearson_r']:.2f}, Spearman ρ = {corr['spearman_rho']:.2f} (n={corr['n']})"
        )
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="#757575", linewidth=0.8, linestyle=":", alpha=0.7)

    fig.suptitle(
        "Cross-value transfer vs steering α (Qwen3.5-9B-Base)\n"
        "Note: α scales differ by method; treat cross-method trend as exploratory.",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    png_path = OUTPUT_DIR / "alpha_vs_transfer_metrics.png"
    pdf_path = OUTPUT_DIR / "alpha_vs_transfer_metrics.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = _load_rows()
    if not rows:
        raise SystemExit(f"No metrics.json files found under {OUTPUT_DIR}")
    _write_table(rows)
    _plot(rows)
    print(f"Wrote table and plots for {len(rows)} methods under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
