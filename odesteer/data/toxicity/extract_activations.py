import argparse
from tqdm import trange

import pandas as pd
import torch

from odesteer.utils import get_project_dir
from odesteer.lm import HuggingFaceLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type = str, default = 'Llama3.1-8B-Base')
    parser.add_argument('-l', '--layer_idx', type = int, default = -1)
    parser.add_argument('-b', '--batch_size', type = int, default = 10)
    args = parser.parse_args()
    
    data_dir = get_project_dir() / 'data' / 'toxicity'
    jigsaw_dir = data_dir / 'jigsaw'
    activations_dir = data_dir / 'activations' / args.model
    activations_dir.mkdir(parents = True, exist_ok = True)
    
    model = HuggingFaceLM(args.model, device = "auto", dtype = torch.float32)
    layer_idx = model.steer_layer_idx if args.layer_idx == -1 else args.layer_idx
    
    final_train_df = pd.read_json(jigsaw_dir / 'final_train.jsonl', lines = True, orient = 'records')
    num_batches = (len(final_train_df) + args.batch_size - 1) // args.batch_size
    
    pos_activations_lst = []
    neg_activations_lst = []
    for i in trange(num_batches):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(final_train_df))
        batch_data = final_train_df.iloc[start_idx:end_idx]
        
        batch_pos_texts, batch_neg_texts = [], []
        for idx, label in enumerate(final_train_df['label'].iloc[start_idx:end_idx]):
            if label <= 0.5:
                batch_pos_texts.append(batch_data['text'].iloc[idx])
            else:
                batch_neg_texts.append(batch_data['text'].iloc[idx])
        
        if len(batch_pos_texts) > 0:
            pos_activations_lst.append(model.extract_prompt_eos_activations(batch_pos_texts, layer_idx = layer_idx).cpu())
        if len(batch_neg_texts) > 0:
            neg_activations_lst.append(model.extract_prompt_eos_activations(batch_neg_texts, layer_idx = layer_idx).cpu())
            
    pos_activations = torch.cat(pos_activations_lst, dim = 0)
    neg_activations = torch.cat(neg_activations_lst, dim = 0)
    torch.save(pos_activations, activations_dir / f'jigsaw_pos_activations_layer{layer_idx}.pt')
    torch.save(neg_activations, activations_dir / f'jigsaw_neg_activations_layer{layer_idx}.pt')