# COLD-Steer: Steering Large Language Models via in-Context One Step Learning Dynamics

Supplementary code for COLD-Steer: Steering Large Language Models via in-Context One Step Learning Dynamics published at ICLR 2026. [Link](https://openreview.net/forum?id=afV4qzquBN)

This is to help contributors get started with the codebase. Please reach out to `ksartik@gatech.edu` in case of any issues.

## Quick overview
- Purpose: research on steering large causal LMs via intervention-style updates and inference-time hooks. Key capabilities: dataset-driven SFT/DPO-style training and steer-at-inference interventions.
- Main components:
  - `src/llm.py` — `SteerableLLM` wrapper around HuggingFace models; provides `generate`, `functional_forward`, `register_steering_hooks`, and PEFT integration.
  - `src/steerer.py` — implementations of steering algorithms and training loops (e.g., `LossFDSteerer`, `LossFDThreshSteerer`). These classes register forward hooks into `SteerableLLM` layers to modify activations.
  - `src/dataset.py` — dataset classes: `CAADataset`, `BiPODataset`, `AxBenchDataset`, `OpinionsQADataset`. They produce tensors named like `prompt_input_ids`, `prompt_attention_mask`, and label variations (`matching_*`, `not_matching_*`).
  - `src/dataset_icl.py`, `src/dataset_llama.py` — dataset classes for the in-context learning baseline and llama model results in the paper (uses a slightly different formatting more suitable for Llama-2-7b).
  - `test.py` — Hydra entrypoint that instantiates the `steerer` and `dataset` from `configs/` and runs training/testing loops.

## Project layout (important files)
- `test.py` — run entrypoint using Hydra (`config_path='configs', config_name='config.yaml'`). See example commands below.
- `configs/` — structured Hydra configs for `dataset`, `llm`, and `steerer` specifications (pick a steerer, llm, dataset by name via overrides).
- `src/llm.py`, `src/steerer.py`, `src/dataset.py`, `src/utils.py` — core logic. Inspect these for examples of tokenization, masking, and hook semantics.
- `data/` — tasks and preprocessed datasets used by dataset classes.

## Setup & dependencies
This repo expects a Python environment with at least:
- `torch` (with CUDA), `transformers`, `omegaconf`, `hydra-core`, `wandb`, and optionally `peft` if using LoRA/PEFT features.

Quick pip install (adjust versions for your environment):
```bash
python -m pip install torch transformers hydra-core omegaconf wandb peft
```

Exact install:
```bash
pip install -r requirements.txt
```

Notes:
- `SteerableLLM` loads models with `AutoModelForCausalLM.from_pretrained(..., device_map='balanced')`. 
- Tokenizers are expected to use left-padding and to set `pad_token = eos_token` as done in datasets.

## Getting started
- Run the standard experiment defined by `configs/config.yaml`:
```bash
python test.py
```

- Override Hydra config values (example: choose a steerer, disable WandB):
```bash
python test.py steerer=cold_fd dataset=caa_dpo_gen llm=llama7b_chat
```

- Steerers are implemented using hook logic in `src/steerer.py`. Add a new steerer class inheriting from the `BaseSteerer.py` by implementing:

```python
class NewSteerer(BaseSteerer):
    def __init__(self, self,
        steerable_llm,
        epsilon: float = 1e-2,
        eta: float = 1e-2,
        training: str = 'sft',
        training_batch_size: int = 1,
        test_batch_size: int = 1,
        log_dir: str = '.',
        steer_masking: str = 'all',
        gen_masking: str = 'prompt',):
        super().__init__(hydra.utils.instantiate(steerable_llm), log_dir=log_dir, batch_size=test_batch_size, steer_masking=steer_masking, gen_masking=gen_masking)
        pass

    def train(self, dataset: Dataset,):
        # 
        # Implement your logic here to make the steering vector from examples 
        # in the dataset
        # 
        pass

    def reset_steering(self):
        # if you want to reset any steering parameters, add logic
        pass

    def steer_output_hook(self, module, input, output, inputs={}, layer_idx=-1, steering_mask=None):
        # 
        # For the new inputs and the intermediate activation output, change it using 
        # this forward hook. Add logic
        # 
        pass
```

## Troubleshooting / tips
- If model loading fails on large models, try smaller checkpoints or use HF `accelerate` with a matching device_map.
- WandB: `test.py` initializes `wandb.init(...)` with values from Hydra. To disable logging during debugging use `wandb.mode=disabled` when running.
- If generation appears truncated or tokens misaligned, check tokenizer templates and that datasets use `tokenizer.padding_side='left'` and `pad_token` set.

## Examples
- `run_caa.sh` shows how to run all steering methods for behavior inference on the CAA dataset for Llama-2-7b-hf and Llama-2-7b-chat-hf models.

## Citation

> @inproceedings{\
> sharma2026coldsteer,\
> title={{COLD}-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics},\
> author={Kartik Sharma and Rakshit Trivedi},\
> booktitle={The Fourteenth International Conference on Learning Representations},\
> year={2026},\
> url={https://openreview.net/forum?id=afV4qzquBN},\
> }
