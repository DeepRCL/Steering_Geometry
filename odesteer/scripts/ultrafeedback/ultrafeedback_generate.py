import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json

import torch
from transformers import GenerationConfig
from lightning import seed_everything

from odesteer.lm import HuggingFaceLM, batch_generate
from odesteer.utils import get_project_dir
from odesteer.utils.data import load_ultrafeedback_data, load_ultrafeedback_prompts


@hydra.main(
    config_path = str(get_project_dir() / 'confs'),
    config_name = 'ultrafeedback',
    version_base = '1.3',
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    output_dir: Path = get_project_dir() / "results" / "ultrafeedback" / "raw_outputs" / cfg.model
    output_dir.mkdir(parents = True, exist_ok = True)
    filename = f"{cfg.model}-l{cfg.layer_idx}-{cfg.steer.name}-UltrafeedbackBinarized-seed{cfg.seed}.jsonl"
    if (output_dir / filename).exists():
        print(f"✓ Output file {filename} already exists. Skipping generation.")
        print("-" * 120)
        exit()

    try:
        print(f"→ Running {cfg.model}-{cfg.steer.name} on layer {cfg.layer_idx}")

        # Load model and fit steering model
        print("→ Loading LLM & Fitting Steer Model ...")
        default_generation_config = GenerationConfig(
            max_new_tokens = 128, do_sample = True, temperature = 0.7,
            top_p = 0.9, repetition_penalty = 1.1, no_repeat_ngram_size = 2, seed = cfg.seed, 
        )
        steer_model_kwargs = OmegaConf.to_container(cfg.steer.kwargs, resolve = True)

        model = HuggingFaceLM(
            cfg.model, cfg.steer.type,
            default_generation_config = default_generation_config,
            steer_model_kwargs = steer_model_kwargs,
            steer_layer_idx = cfg.layer_idx,
            device = 'auto', dtype = torch.float32
        )

        # Load training data and fit
        pos_train, neg_train = load_ultrafeedback_data(cfg.model, cfg.layer_idx, 'train')
        model.fit_steer_model(pos_train, neg_train)

        # Load test prompts
        print("→ Loading test prompts ...")
        prompts = load_ultrafeedback_prompts('test')

        print(f"→ Generating {len(prompts)} responses with T={cfg.steer.T} ...")
        outputs = batch_generate(model, prompts, T = cfg.steer.T, batch_size = cfg.batch_size)
        print(f"→ Generated {len(outputs)} outputs")

        print(f"\n→ Saving outputs to {filename} ...")
        with open(output_dir / filename, "w") as f:
            for prompt, output in zip(prompts, outputs):
                f.write(json.dumps(
                    {
                        "prompt": prompt,
                        "output": output,
                        "generator": f"{cfg.model}-{cfg.steer.name}",
                        "dataset": "UltrafeedbackBinarized",
                        "T": cfg.steer.T,
                    }
                ) + "\n")

        print(f"✓ Completed {cfg.model}-{cfg.steer.name} on UltrafeedbackBinarized")
        print(f"  Total responses generated: {len(outputs)}")
        print(f"  Configuration: T={cfg.steer.T}")
        print("-" * 120)

    except Exception as e:
        print(f"→ Error: {e}")
        import traceback
        traceback.print_exc()
        exit()


if __name__ == '__main__':
    main()
