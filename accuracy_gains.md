# Steering Accuracy Gains

Accuracy gain is computed as:

`mean steered accuracy at best alpha - mean baseline accuracy`

The mean is macro-averaged over the 20 Schwartz values. The best alpha is selected by highest mean accuracy gain on the held-out validation/eval split.

For the dual-metric runs:

- `A/B next-token` compares `P(A)` vs. `P(B)` on the multiple-choice evaluation prompt.
- `Full-answer mean logprob` compares mean token log-probability of the full positive answer vs. the full negative answer.

## CAA: Model Sweep

| Model | Baseline Acc. | Best Steered Acc. | Gain | Relative Gain | Best Alpha |
|---|---:|---:|---:|---:|---:|
| Gemma-3-4B | 29.99% | 29.84% | -0.15 pp | -0.50% | 1 |
| Gemma-3-12B | 33.15% | 33.84% | +0.69 pp | +2.08% | 16 |
| Qwen2.5-7B | 33.32% | 52.56% | +19.24 pp | +57.73% | 40 |
| Qwen2.5-14B | 34.18% | 49.41% | +15.22 pp | +44.54% | 40 |
| Qwen2.5-32B | 24.45% | 25.89% | +1.44 pp | +5.90% | 25 |
| Gemma-4-31B | 25.51% | 39.75% | +14.25 pp | +55.85% | 40 |
| Qwen3.5-0.8B Base | 18.95% | 52.09% | +33.13 pp | +174.79% | 8 |
| Qwen3.5-2B Base | 18.51% | 52.71% | +34.20 pp | +184.77% | 20 |
| Qwen3.5-4B Base | 35.40% | 53.82% | +18.42 pp | +52.04% | 30 |
| Qwen3.5-9B Base | 41.40% | 53.13% | +11.74 pp | +28.36% | 40 |

## Qwen3.5-9B: Method Comparison

| Model | Method | Baseline Acc. | Best Steered Acc. | Gain | Relative Gain | Best Alpha |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Base | OPT | 23.8% | 76.0% | +52.3 pp | -- | 40 |
| Qwen3.5-9B Base | Cold-Steer (FD) | -- | -- | -- | -- | -- |
| Qwen3.5-9B Base | ODE-Steer | 48.4% | 65.2% | +16.7 pp | -- | 20 |
| Qwen3.5-9B Base | SphericalSteer | 41.18% | 50.57% | +9.38 pp | +22.78% | 0.9 |
| Qwen3.5-9B Base | CAA | 41.18% | 59.25% | +18.07 pp | +43.87% | 20.0 |
| Qwen3.5-9B Base | BiPO | 41.18% | 50.05% | +8.87 pp | +21.54% | 10.0 |
| Qwen3.5-9B Base | SparseCAA | 44.34% | 51.20% | +6.86 pp | +15.48% | 4.0 |
| Qwen3.5-9B Instruct | OPT | -- | -- | -- | -- | -- |
| Qwen3.5-9B Instruct | SphericalSteer | 49.25% | 48.99% | -0.26 pp | -0.53% | 0.9 |
| Qwen3.5-9B Instruct | CAA | 49.25% | 50.04% | +0.79 pp | +1.60% | 20.0 |
| Qwen3.5-9B Instruct | SparseCAA | 51.20% | 51.20% | +0.00 pp | +0.00% | 0.5 |

Note: OPT, Cold-Steer, and ODE-Steer accuracy artifacts were not present locally for the matched Qwen3.5-9B Instruct setting. The available Base OPT and ODE-Steer rows are retained from the existing reported values. No local Cold-Steer accuracy artifact was found.

Note: The paper-wired SparseCAA rows use `SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B-Base` and `SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B` with `SAE/sae_base_best.pt`.

## Qwen3.5-9B: Dual-Metric Clean Runs

| Model | Method | Metric | Baseline Acc. | Best Steered Acc. | Gain | Relative Gain | Best Alpha | Run |
|---|---|---|---:|---:|---:|---:|---:|---|
| Qwen3.5-9B Base | CAA | A/B next-token | 41.18% | 59.25% | +18.07 pp | +43.87% | 20.0 | `CAA/Geometry/outputs/qwen3_5_9b_base_best_dual_metrics_20260520_183805/Qwen__Qwen3.5-9B-Base` |
| Qwen3.5-9B Base | CAA | Full-answer mean logprob | 40.55% | 71.12% | +30.57 pp | +75.38% | 40.0 | `CAA/Geometry/outputs/qwen3_5_9b_base_best_dual_metrics_20260520_183805/Qwen__Qwen3.5-9B-Base` |
| Qwen3.5-9B Base | SphericalSteer | A/B next-token | 41.18% | 50.57% | +9.38 pp | +22.78% | 0.9 | `SphericalSteer/focused_tuning/k2_bneg0p6_base_final_dual_new_relations/Qwen__Qwen3.5-9B-Base` |
| Qwen3.5-9B Base | SphericalSteer | Full-answer mean logprob | 40.55% | 42.65% | +2.10 pp | +5.18% | 0.9 | `SphericalSteer/focused_tuning/k2_bneg0p6_base_final_dual_new_relations/Qwen__Qwen3.5-9B-Base` |
| Qwen3.5-9B Base | BiPO | A/B next-token | 41.18% | 50.05% | +8.87 pp | +21.54% | 10.0 | `BiPO/focused_tuning/qwen35_opt_20260520_221258/Qwen__Qwen3.5-9B-Base` |
| Qwen3.5-9B Base | BiPO | Full-answer mean logprob | 40.55% | 61.18% | +20.63 pp | +50.87% | 10.0 | `BiPO/focused_tuning/qwen35_opt_20260520_221258/Qwen__Qwen3.5-9B-Base` |
| Qwen3.5-9B Base | SparseCAA | A/B next-token | 44.34% | 51.20% | +6.86 pp | +15.48% | 4.0 | `SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B-Base` |
| Qwen3.5-9B Instruct | CAA | A/B next-token | 49.25% | 50.04% | +0.79 pp | +1.60% | 20.0 | `CAA/Geometry/outputs/qwen3_5_9b_instruct_best_final_dual/Qwen__Qwen3.5-9B` |
| Qwen3.5-9B Instruct | CAA | Full-answer mean logprob | 17.04% | 48.56% | +31.52 pp | +185.04% | 40.0 | `CAA/Geometry/outputs/qwen3_5_9b_instruct_best_final_dual/Qwen__Qwen3.5-9B` |
| Qwen3.5-9B Instruct | SphericalSteer | A/B next-token | 49.25% | 48.99% | -0.26 pp | -0.53% | 0.9 | `SphericalSteer/focused_tuning/k2_bneg0p6_instruct_final_dual/Qwen__Qwen3.5-9B` |
| Qwen3.5-9B Instruct | SphericalSteer | Full-answer mean logprob | 17.04% | 18.65% | +1.61 pp | +9.47% | 0.9 | `SphericalSteer/focused_tuning/k2_bneg0p6_instruct_final_dual/Qwen__Qwen3.5-9B` |
| Qwen3.5-9B Instruct | SparseCAA | A/B next-token | 51.20% | 51.20% | +0.00 pp | +0.00% | 0.5 | `SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B` |

Note: SparseCAA full-answer mean-logprob evaluation artifacts were not present locally, so only A/B next-token rows are reported for SparseCAA.

## Qwen3.5-9B: Accuracy + Geometry Summary

This table uses A/B next-token accuracy gains plus the five geometry metrics from `geometry_metrics.json`. `Raw LLM activations` is unsteered positive-answer activation geometry (no accuracy row). OPT uses the existing reported accuracy row and the hard-coded `OPT_METRICS` block in `CAA/generate_geometry_tables.py`; no matching local OPT output artifact was found.

| Model | Method | Baseline Acc. | Best Steered Acc. | Gain | Best Alpha | rho_T | r_T | rho_C | rho_H | Delta_pol |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B Base | Raw LLM activations | -- | -- | -- | -- | 0.2228 | 0.2087 | 0.2228 | 0.2115 | 0.0009 |
| Qwen3.5-9B Base | OPT | 23.8% | 76.0% | +52.3 pp | 40 | -0.0049 | -0.0167 | -0.0049 | 0.1180 | -0.0109 |
| Qwen3.5-9B Base | Cold-Steer (FD) | -- | -- | -- | -- | 0.0265 | -0.0100 | 0.0265 | 0.0013 | 0.0162 |
| Qwen3.5-9B Base | ODE-Steer | 48.4% | 65.2% | +16.7 pp | 20 | 0.2730 | 0.2963 | 0.2730 | 0.2988 | 0.0255 |
| Qwen3.5-9B Base | SphericalSteer | 41.18% | 50.57% | +9.38 pp | 0.9 | 0.3962 | 0.4061 | 0.3962 | 0.2746 | 0.3620 |
| Qwen3.5-9B Base | CAA | 41.18% | 59.25% | +18.07 pp | 20.0 | 0.4599 | 0.4750 | 0.4599 | 0.3407 | 0.3874 |
| Qwen3.5-9B Base | SparseCAA | 44.34% | 51.20% | +6.86 pp | 4.0 | 0.4584 | 0.4520 | 0.4584 | 0.4392 | 0.4215 |
| Qwen3.5-9B Instruct | Raw LLM activations | -- | -- | -- | -- | 0.1253 | 0.1112 | 0.1253 | 0.1455 | 0.0004 |
| Qwen3.5-9B Instruct | OPT | -- | -- | -- | -- | 0.0615 | 0.0552 | 0.0615 | -0.0177 | -0.0099 |
| Qwen3.5-9B Instruct | SphericalSteer | 49.25% | 48.99% | -0.26 pp | 0.9 | 0.2048 | 0.2170 | 0.2048 | 0.1009 | 0.2397 |
| Qwen3.5-9B Instruct | CAA | 49.25% | 50.04% | +0.79 pp | 20.0 | 0.2351 | 0.2527 | 0.2351 | 0.1248 | 0.2603 |
| Qwen3.5-9B Instruct | SparseCAA | 51.20% | 51.20% | +0.00 pp | 0.5 | 0.2625 | 0.2627 | 0.2625 | 0.2251 | 0.1747 |
