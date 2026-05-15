import json
from pathlib import Path


ROOT = Path("/home/mahdi.abootorabi/Steering_Geometry")
OUT_PATH = ROOT / "CAA" / "geometry_tables.tex"
OUT_JSON = ROOT / "CAA" / "geometry_table_data.json"

METRICS = [
    ("spearman_rho", "Spearman $\\rho$", "spearman_p_value"),
    ("pearson_r", "Pearson $r$", "pearson_p_value"),
    ("circular_distance_spearman", "Circ. Dist. $\\rho$", "circular_distance_p_value"),
    ("hierarchical_distance_spearman", "Hier. Dist. $\\rho$", "hierarchical_distance_p_value"),
    ("lower_minus_opposite_cosine", "Lower$-$Opp. Cos.", None),
]

OPT_METRICS = {
    "Qwen/Qwen3.5-9B-Base": {
        "spearman_rho": -0.004916224926200378,
        "spearman_p_value": 0.9463279831521348,
        "pearson_r": -0.016722656952075882,
        "pearson_p_value": 0.8188682661902891,
        "circular_distance_spearman": -0.004916224926200378,
        "circular_distance_p_value": 0.9463279831521348,
        "hierarchical_distance_spearman": 0.11802173797190742,
        "hierarchical_distance_p_value": 0.10485579109585051,
        "lower_minus_opposite_cosine": -0.010869252068964254,
    },
    "Qwen/Qwen3.5-9B": {
        "spearman_rho": 0.061545155695080934,
        "spearman_p_value": 0.3989246685531906,
        "pearson_r": 0.055221977490461246,
        "pearson_p_value": 0.44920738652210623,
        "circular_distance_spearman": 0.061545155695080934,
        "circular_distance_p_value": 0.3989246685531906,
        "hierarchical_distance_spearman": -0.01771733232319764,
        "hierarchical_distance_p_value": 0.8082966517339384,
        "lower_minus_opposite_cosine": -0.009875694801325378,
    },
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_caa_metrics(base_dir: str):
    base = ROOT / base_dir
    children = [p for p in base.iterdir() if p.is_dir()]
    if len(children) != 1:
        raise ValueError(f"Expected exactly one model subdir in {base}, found {[c.name for c in children]}")
    model_dir = children[0]
    config = load_json(model_dir / "config.json")
    metrics = load_json(model_dir / "geometry" / "geometry_metrics.json")
    return config["model_name"], metrics


def load_sparse_metrics(base_dir: str):
    base = ROOT / base_dir
    config = load_json(base / "pipeline_config.json")
    metrics = load_json(base / "geometry_centered" / "geometry_metrics.json")
    return config["model_name"], metrics


def fmt_p(p: float) -> str:
    return f"{p:.2e}"


def fmt_metric_cell(value: float, pvalue=None, bold=False) -> str:
    value_txt = f"{value:.4f}"
    if pvalue is None:
        if bold:
            return f"\\textbf{{{value_txt}}}"
        return value_txt

    p_txt = fmt_p(pvalue)
    if bold:
        return "\\shortstack{\\textbf{" + value_txt + "}\\\\{\\scriptsize \\textbf{$p$=" + p_txt + "}}}"
    return f"\\shortstack{{{value_txt}\\\\{{\\scriptsize $p$={p_txt}}}}}"


def render_stage1(rows):
    best = {}
    for metric_key, _label, _p_key in METRICS:
        best[metric_key] = max(row["metrics"][metric_key] for row in rows)

    lines = []
    lines.append("%% Stage 1: Effect of model family and size (all CAA)")
    lines.append("\\begin{table*}[ht]")
    lines.append("    \\centering")
    lines.append("    \\caption{Stage 1: Effect of model family and size on steering-geometry metrics (all using CAA, centered-renorm geometry). For the first four columns, p-values are shown beneath the metric values.}")
    lines.append("    \\label{tab:stage1_geometry}")
    lines.append("    \\setlength{\\tabcolsep}{6pt}")
    lines.append("    \\renewcommand{\\arraystretch}{1.15}")
    lines.append("    \\small")
    lines.append("    \\begin{tabular}{ll c c c c c}")
    lines.append("        \\toprule")
    lines.append("        \\textbf{Model Family} & \\textbf{Backbone Model} & \\textbf{Spearman $\\rho$} & \\textbf{Pearson $r$} & \\textbf{Circ. Dist. $\\rho$} & \\textbf{Hier. Dist. $\\rho$} & \\textbf{Lower$-$Opp. Cos.} \\\\")
    lines.append("        \\midrule")

    current_family = None
    family_order = ["Gemma 3", "Qwen 2.5", "Qwen 3.5", "Gemma 4"]
    family_to_rows = {fam: [r for r in rows if r["family"] == fam] for fam in family_order}
    for fam in family_order:
        fam_rows = family_to_rows.get(fam, [])
        if not fam_rows:
            continue
        if current_family is not None:
            lines.append("        \\midrule")
        current_family = fam
        for row in fam_rows:
            cell_values = []
            for metric_key, _label, p_key in METRICS:
                bold = row["metrics"][metric_key] == best[metric_key]
                pvalue = row["metrics"].get(p_key) if p_key else None
                cell_values.append(fmt_metric_cell(row["metrics"][metric_key], pvalue, bold=bold))
            lines.append(
                f"        {row['family']} & {row['model_name']} & " + " & ".join(cell_values) + " \\\\"
            )

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def render_stage2(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model_name"], []).append(row)

    lines = []
    lines.append("%% Stage 2: Effect of steering method (fixed model)")
    lines.append("\\begin{table*}[ht]")
    lines.append("    \\centering")
    lines.append("    \\caption{Stage 2: Effect of steering method on steering-geometry metrics for matched Qwen 3.5 backbones. For the first four columns, p-values are shown beneath the metric values.}")
    lines.append("    \\label{tab:stage2_geometry}")
    lines.append("    \\setlength{\\tabcolsep}{6pt}")
    lines.append("    \\renewcommand{\\arraystretch}{1.15}")
    lines.append("    \\small")
    lines.append("    \\begin{tabular}{ll c c c c c}")
    lines.append("        \\toprule")
    lines.append("        \\textbf{Backbone Model} & \\textbf{Method} & \\textbf{Spearman $\\rho$} & \\textbf{Pearson $r$} & \\textbf{Circ. Dist. $\\rho$} & \\textbf{Hier. Dist. $\\rho$} & \\textbf{Lower$-$Opp. Cos.} \\\\")
    lines.append("        \\midrule")

    model_order = ["Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B"]
    method_order = ["OPT", "CAA", "SparseCAA"]

    first_group = True
    for model_name in model_order:
        model_rows = sorted(grouped[model_name], key=lambda r: method_order.index(r["method"]))
        if not first_group:
            lines.append("        \\midrule")
        first_group = False
        best = {metric_key: max(r["metrics"][metric_key] for r in model_rows) for metric_key, _l, _p in METRICS}
        for row in model_rows:
            cell_values = []
            for metric_key, _label, p_key in METRICS:
                bold = row["metrics"][metric_key] == best[metric_key]
                pvalue = row["metrics"].get(p_key) if p_key else None
                cell_values.append(fmt_metric_cell(row["metrics"][metric_key], pvalue, bold=bold))
            lines.append(
                f"        {row['model_name']} & {row['method']} & " + " & ".join(cell_values) + " \\\\"
            )

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main():
    stage1_specs = [
        ("Gemma 3", "CAA/Geometry/outputs/gemma_3_4b_pt_mid50_75_v1_centered_renorm"),
        ("Gemma 3", "CAA/Geometry/outputs/gemma_3_12b_pt_v3_centered_renorm"),
        ("Qwen 2.5", "CAA/Geometry/outputs/qwen2_5_7b_v3_centered_renorm"),
        ("Qwen 2.5", "CAA/Geometry/outputs/qwen2_5_14b_v3_centered_renorm"),
        ("Qwen 2.5", "CAA/Geometry/outputs/qwen2_5_32b_v3_centered_renorm"),
        ("Qwen 3.5", "CAA/Geometry/outputs/qwen3_5_0_8b_base_v3_centered_renorm"),
        ("Qwen 3.5", "CAA/Geometry/outputs/qwen3_5_2b_base_v3_centered_renorm"),
        ("Qwen 3.5", "CAA/Geometry/outputs/qwen3_5_4b_base_v3_centered_renorm"),
        ("Qwen 3.5", "CAA/Geometry/outputs/qwen3_5_9b_base_v2_centered_renorm"),
        ("Gemma 4", "CAA/Geometry/outputs/gemma_4_31b_v4_centered_renorm"),
    ]

    stage1_rows = []
    for family, path in stage1_specs:
        model_name, metrics = load_caa_metrics(path)
        stage1_rows.append({"family": family, "model_name": model_name, "metrics": metrics})

    stage2_rows = []
    # OPT rows from provided metrics
    for model_name in ["Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B"]:
        stage2_rows.append({"model_name": model_name, "method": "OPT", "metrics": OPT_METRICS[model_name]})
    # CAA rows
    for path in [
        "CAA/Geometry/outputs/qwen3_5_9b_base_v2_centered_renorm",
        "CAA/Geometry/outputs/qwen3_5_9b_v3_centered_renorm",
    ]:
        model_name, metrics = load_caa_metrics(path)
        stage2_rows.append({"model_name": model_name, "method": "CAA", "metrics": metrics})
    # SparseCAA rows
    for path in [
        "SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B-Base",
        "SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B",
    ]:
        model_name, metrics = load_sparse_metrics(path)
        stage2_rows.append({"model_name": model_name, "method": "SparseCAA", "metrics": metrics})

    latex = render_stage1(stage1_rows) + "\n\n" + render_stage2(stage2_rows) + "\n"
    OUT_PATH.write_text(latex)

    OUT_JSON.write_text(json.dumps({"stage1": stage1_rows, "stage2": stage2_rows}, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
