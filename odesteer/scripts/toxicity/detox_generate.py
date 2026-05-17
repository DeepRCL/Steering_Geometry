import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json

import torch
from transformers import GenerationConfig
from lightning import seed_everything

from odesteer.lm import HuggingFaceLM, batch_generate
from odesteer.utils import get_project_dir
from odesteer.utils.data import load_jigsaw_activations, load_rtp_prompts


@hydra.main(
    config_path = str(get_project_dir() / 'confs'), 
    config_name = 'detox',
    version_base = '1.3',
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    output_dir: Path = get_project_dir() / "results" / "toxicity" / "raw_outputs" / cfg.model
    output_dir.mkdir(parents = True, exist_ok = True)
    filename = f"{cfg.model}-l{cfg.layer_idx}-{cfg.steer.name}-RealToxicityPrompts-seed{cfg.seed}.jsonl"
    if (output_dir / filename).exists():
        print(f"✓ Output file {filename} already exists. Skipping RealToxicityPrompts generation.")
        print("-" * 120)    
        exit()
    
    try:        
        print("→ Loading LLM & Fitting Steer Model ...")
        default_generation_config = GenerationConfig(
            max_new_tokens = 50, do_sample = True, temperature = 0.7,
            top_p = 0.9, repetition_penalty = 1.1, seed = cfg.seed 
        )
        steer_model_kwargs = OmegaConf.to_container(cfg.steer.kwargs, resolve = True)
            
        model = HuggingFaceLM(
            cfg.model, cfg.steer.type,
            default_generation_config = default_generation_config,
            steer_model_kwargs = steer_model_kwargs, 
            steer_layer_idx = cfg.layer_idx,
            device = 'auto', dtype = torch.float32
        )

        pos_train, neg_train = load_jigsaw_activations(cfg.model, cfg.layer_idx)
        model.fit_steer_model(pos_train, neg_train)
        
        print("→ Loading test prompts from real-toxicity-prompts ...")
        prompts = load_rtp_prompts("test")

        print(f"→ Generating {len(prompts)} responses with T={cfg.steer.T} ...")
        outputs = batch_generate(model, prompts, T = cfg.steer.T, batch_size = cfg.batch_size)
        print(f"→ Generated {len(outputs)} outputs")
        
        del model
        torch.cuda.empty_cache()

        print(f"\n→ Saving all outputs to {filename} ...")
        with open(output_dir / filename, "w") as f:
            for prompt, output in zip(prompts, outputs):
                f.write(json.dumps(
                    {
                        "prompt": prompt, 
                        "output": output,    
                        "generator": f"{cfg.model}-{cfg.steer.name}",
                        "dataset": "realtoxicityprompts",
                        "T": cfg.steer.T,
                    }
                ) + "\n")
        
        print(f"✓ Completed {cfg.model}-{cfg.steer.name} on real-toxicity-prompts with training on Jigsaw")
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