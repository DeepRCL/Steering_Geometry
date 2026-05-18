from omegaconf import DictConfig
import hydra
import wandb
from omegaconf import OmegaConf
import random
import torch
import numpy as np
import os


def set_seed_everywhere(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Additional PyTorch settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.1")
def main(config: DictConfig):
    
    wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    # OmegaConf.register_new_resolver("eval", eval)

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode if 'mode' in config.wandb else None, #"disabled",
        config=wandb.config
    )
    
    steerer = hydra.utils.instantiate(config.steerer, log_dir=config.log_dir)
    set_seed_everywhere(config.seed)
    trainset = hydra.utils.instantiate(config.dataset, tokenizer=steerer.steerable_llm.tokenizer, split='train')
    testset = hydra.utils.instantiate(config.dataset, tokenizer=steerer.steerable_llm.tokenizer, split='test')
        
    steerer.train(trainset)
    results = steerer.test_all(testset)
    
    print (results)
    wandb.log(results)
    
    wandb.finish()

if __name__ == '__main__':
    main()