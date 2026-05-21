# Qwen3.5-9B Base Accuracy and Geometry

All non-OPT values below were read from local JSON artifacts. OPT values are the existing paper-wired values already present in the repo; no local OPT output artifact matching those rows was found. `Raw LLM activations` is the unsteered positive-answer activation geometry, so it has geometry metrics but no steering accuracy row.

## Accuracy Gain: A/B Next-Token

| Model | Method | Baseline Acc. | Best Steered Acc. | Gain | Relative Gain | Best Alpha | Source |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3.5-9B Base | OPT | 23.8% | 76.0% | +52.3 pp | -- | 40 | Existing reported row in `accuracy_gains.md`; local OPT artifact unavailable |
| Qwen3.5-9B Base | CAA | 41.18% | 59.25% | +18.07 pp | +43.87% | 20.0 | `CAA/Geometry/outputs/qwen3_5_9b_base_best_final_20260520_194559/Qwen__Qwen3.5-9B-Base/evaluation/evaluation_summary.json` |
| Qwen3.5-9B Base | SparseCAA | 44.34% | 51.20% | +6.86 pp | +15.48% | 4.0 | `SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B-Base/evaluation/eval_results.json` |
| Qwen3.5-9B Base | SphericalSteer | 41.18% | 50.57% | +9.38 pp | +22.78% | 0.9 | `SphericalSteer/focused_tuning/k2_bneg0p6_final_20260520_194617/Qwen__Qwen3.5-9B-Base/evaluation/evaluation_summary.json` |
| Qwen3.5-9B Base | QwenScopeBest | 41.18% | 54.44% | +13.26 pp | +32.19% | 8.0 | `SAE/QwenScopeCAA/outputs_qwenscope_l15_k100_final_20260520_183805/Qwen__Qwen3.5-9B-Base_layer15_k100/evaluation_caa_base/evaluation_summary.json` |

## Accuracy Gain: Full-Answer Mean Logprob

| Model | Method | Baseline Acc. | Best Steered Acc. | Gain | Relative Gain | Best Alpha | Source |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3.5-9B Base | CAA | 40.55% | 71.09% | +30.54 pp | +75.31% | 40.0 | `CAA/Geometry/outputs/qwen3_5_9b_base_best_final_20260520_194559/Qwen__Qwen3.5-9B-Base/evaluation/evaluation_summary_full_logprob.json` |
| Qwen3.5-9B Base | SphericalSteer | 40.55% | 42.65% | +2.10 pp | +5.18% | 0.9 | `SphericalSteer/focused_tuning/k2_bneg0p6_final_20260520_194617/Qwen__Qwen3.5-9B-Base/evaluation/evaluation_summary_full_logprob.json` |
| Qwen3.5-9B Base | QwenScopeBest | 40.55% | 81.44% | +40.89 pp | +100.82% | 20.0 | `SAE/QwenScopeCAA/outputs_qwenscope_l15_k100_final_20260520_183805/Qwen__Qwen3.5-9B-Base_layer15_k100/evaluation_caa_base/evaluation_summary_full_logprob.json` |

## Geometry Metrics

| Model | Method | rho_T | p(rho_T) | r_T | p(r_T) | rho_C | p(rho_C) | rho_H | p(rho_H) | Delta_pol | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-9B Base | Raw LLM activations | 0.2228 | 2.00e-03 | 0.2087 | 3.85e-03 | 0.2228 | 2.00e-03 | 0.2115 | 3.39e-03 | 0.0009 | `CAA/Geometry/outputs/qwen3_5_9b_base_v2_centered_renorm/Qwen__Qwen3.5-9B-Base/activation_geometry_raw/geometry_metrics.json` |
| Qwen3.5-9B Base | OPT | -0.0049 | 9.46e-01 | -0.0167 | 8.19e-01 | -0.0049 | 9.46e-01 | 0.1180 | 1.05e-01 | -0.0109 | `CAA/generate_geometry_tables.py` `OPT_METRICS` |
| Qwen3.5-9B Base | CAA | 0.4599 | 2.48e-11 | 0.4750 | 4.38e-12 | 0.4599 | 2.48e-11 | 0.3407 | 1.51e-06 | 0.3874 | `CAA/Geometry/outputs/qwen3_5_9b_base_best_final_20260520_194559/Qwen__Qwen3.5-9B-Base/geometry/geometry_metrics.json` |
| Qwen3.5-9B Base | SparseCAA | 0.4584 | 2.92e-11 | 0.4520 | 5.91e-11 | 0.4584 | 2.92e-11 | 0.4392 | 2.32e-10 | 0.4215 | `SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B-Base/geometry_centered/geometry_metrics.json` |
| Qwen3.5-9B Base | SphericalSteer | 0.2406 | 8.25e-04 | 0.2529 | 4.31e-04 | 0.3962 | 1.52e-08 | 0.2746 | 1.26e-04 | 0.3620 | `SphericalSteer/focused_tuning/k2_bneg0p6_final_20260520_194617/Qwen__Qwen3.5-9B-Base/geometry/geometry_metrics.json` |
| Qwen3.5-9B Base | QwenScopeBest | -0.0824 | 2.58e-01 | -0.0544 | 4.56e-01 | -0.0455 | 5.33e-01 | -0.0768 | 2.92e-01 | 0.0553 | `SAE/QwenScopeCAA/outputs_qwenscope_l15_k100_final_20260520_183805/Qwen__Qwen3.5-9B-Base_layer15_k100/geometry_centered/geometry_metrics.json` |
