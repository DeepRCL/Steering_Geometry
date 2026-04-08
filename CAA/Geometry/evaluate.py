import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Any
from .config import PipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER
from .model_loader import ModelInfo
from .data_loader import DataLoader, PromptFormatter
from .steering.base import SteeringMethod

def evaluate_steering(config: PipelineConfig,
                      data_loader: DataLoader,
                      model_info: ModelInfo,
                      steering_method: SteeringMethod,
                      vectors: Dict[str, torch.Tensor],
                      layer_idx: int):
    """
    Evaluates steering on the 10% held-out test set for each value.
    vectors: {value_name: steering_vector}
    """
    print(f"Evaluating steering on layer {layer_idx} with alphas: {config.alpha_values}")
    formatter = PromptFormatter(model_info.tokenizer, model_info.is_instruct)
    
    out_dir = config.subdir("evaluation")
    results_all = {}
    
    model_info.model.eval()
    
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        eval_instances = data_loader.get_eval_instances(val)
        if not eval_instances:
            continue
            
        vector = vectors[val]
        val_results = {}
        
        for alpha in config.alpha_values:
            handles = steering_method.apply(model_info, layer_idx, vector, alpha)
            
            correct_count = 0
            alpha_instance_results = []
            
            print(f"Evaluating {val} (alpha={alpha})...")
            
            for inst in eval_instances:
                tokens, a_id, b_id = formatter.format_eval_prompt(inst)
                
                input_ids = torch.tensor([tokens]).to(model_info.device)
                
                with torch.no_grad():
                    logits = model_info.model(input_ids).logits
                    
                # Get last token logits
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                
                prob_a = probs[a_id].item()
                prob_b = probs[b_id].item()
                
                # Model chose A if P(A) > P(B)
                chose_a = prob_a > prob_b
                
                # Did it match the positive behavior?
                is_correct = (chose_a == inst.pos_is_a)
                if is_correct:
                    correct_count += 1
                    
                alpha_instance_results.append({
                    "sample_id": inst.sample_id,
                    "prob_a": prob_a,
                    "prob_b": prob_b,
                    "chose_a": chose_a,
                    "pos_is_a": inst.pos_is_a,
                    "is_correct": is_correct
                })
                
            steering_method.cleanup(handles)
            
            accuracy = correct_count / len(eval_instances)
            val_results[alpha] = {
                "accuracy": accuracy,
                "num_eval": len(eval_instances),
                "details": alpha_instance_results
            }
            
        results_all[val] = val_results
        
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(results_all, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {out_dir}")
