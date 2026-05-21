# Steering Accuracy Gains

Accuracy gain is computed as:

`mean steered accuracy at best alpha - mean baseline accuracy`

The mean is macro-averaged over the 20 Schwartz values. The best alpha is selected by highest mean accuracy gain on the held-out validation/eval split.

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
| Qwen3.5-9B Base | CAA | 41.40% | 53.13% | +11.74 pp | +28.36% | 40 |
| Qwen3.5-9B Base | SparseCAA | 44.34% | 51.20% | +6.86 pp | +15.48% | 4 |
| Qwen3.5-9B Base | QwenScopeCAA SparseCAA (k=1024) | 48.14% | 56.70% | +8.57 pp | +17.79% | 2 |
| Qwen3.5-9B Base | OPT | 23.8% | 76.0% | +52.3% | -- | 40 |
| Qwen3.5-9B Base | ODE-Steer | 48.4% | 65.2% | +16.7% | -- | 20 |
| Qwen3.5-9B Instruct | OPT | -- | -- | -- | -- | -- | 
| Qwen3.5-9B Instruct | CAA | 49.25% | 49.25% | +0.00 pp | +0.00% | 0.25 |
| Qwen3.5-9B Instruct | SparseCAA | 51.20% | 51.20% | +0.00 pp | +0.00% | 0.5 |

Note: OPT accuracy artifacts were not present locally. The OPT pipeline should produce `steering_eval_metrics.json` under `llm-steering-opt/steering_results/...`; those values can be filled in later.

Note: The paper-wired SparseCAA row uses `SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B-Base` with `SAE/sae_base_best.pt`. The QwenScopeCAA row uses `SAE/QwenScopeCAA/outputs/Qwen__Qwen3.5-9B-Base_layer16_k1024` with the pretrained `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` SAE, so it should be treated as a separate sparse variant.