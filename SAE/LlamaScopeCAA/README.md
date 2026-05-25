# LlamaScopeCAA

SAS/SparseCAA-style steering for `meta-llama/Llama-3.1-8B` using the
[OpenMOSS Llama-Scope](https://huggingface.co/OpenMOSS-Team/Llama-Scope)
residual-stream SAEs.

This is the Llama-native sibling of `SAE/QwenScopeCAA`. It hooks the residual
stream after a selected transformer layer, maps activations through the
Llama-Scope SAE, adds value persona directions in SAE space, re-applies the
SAE sparsifier, decodes back to residual space, and adds the SAE reconstruction
residual by default.

Defaults:

- Model: `meta-llama/Llama-3.1-8B`
- SAE repo: `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x`
- SAE site: `R` residual stream
- Feature width: `32768` (8x)
- Sparsifier: Llama-Scope checkpoint metadata (`jumprelu` for the tested layer)
- Base dataset: `CAA/value_data/final_dataset_200.csv`

Run from the project root:

```bash
python -m SAE.LlamaScopeCAA.run_pipeline \
  --model_name meta-llama/Llama-3.1-8B \
  --layer 10 \
  --modules all \
  --alpha 1.0,4.0,8.0,20.0,40.0 \
  --geometry_vector displacement \
  --geometry_source neg \
  --relations_path CAA/value_data/schwartz_relations-new.json \
  --output_dir SAE/LlamaScopeCAA/outputs_llama_3_1_8b_l10
```

To skip SAE fine-tuning and use the pretrained Llama-Scope SAE directly:

```bash
python -m SAE.LlamaScopeCAA.run_pipeline \
  --model_name meta-llama/Llama-3.1-8B \
  --layer 10 \
  --skip_finetune \
  --alpha 1.0,4.0,8.0,20.0,40.0 \
  --geometry_vector displacement \
  --geometry_source neg \
  --relations_path CAA/value_data/schwartz_relations-new.json \
  --output_dir SAE/LlamaScopeCAA/outputs_llama_3_1_8b_l10
```

Outputs are written under `<output_dir>/meta-llama__Llama-3.1-8B_layer10_k50/`.
