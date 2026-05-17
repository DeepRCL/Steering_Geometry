from tqdm import tqdm
import ast
import random
from typing import Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from odesteer.utils import get_project_dir


SEED = 20250415
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def split_data(df: pd.DataFrame, val_test_size: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = list(range(len(df)))
    random.shuffle(indices)
    df: pd.DataFrame = df.iloc[indices]
    train_size: int = len(df) - 2 * val_test_size
    train_df: pd.DataFrame = df.iloc[:train_size]
    val_df: pd.DataFrame = df.iloc[train_size:train_size + val_test_size]
    test_df: pd.DataFrame = df.iloc[train_size + val_test_size:]
    return train_df, val_df, test_df


def download_save_data():
    ds = load_dataset("allenai/real-toxicity-prompts", split = "train")
    save_dir = get_project_dir() / 'data' / 'toxicity' / 'real_tox_prompts'
    save_dir.mkdir(parents = True, exist_ok = True)
    ds.to_pandas().to_csv(save_dir / 'real-toxicity-prompts.csv', index = False)


def process_prompt_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'prompt' column containing dictionary-like strings and expand into separate columns.
    """
    # Extract the prompt column
    prompt_series: pd.Series = df['prompt']
    
    # Parse each dictionary string and create a list of dictionaries
    parsed_prompts: list[dict] = []
    for prompt_str in prompt_series:
        try:
            # Use ast.literal_eval to safely parse the dictionary string
            prompt_dict: dict = ast.literal_eval(prompt_str)
            parsed_prompts.append(prompt_dict)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing prompt: {prompt_str[:100]}... Error: {e}")
            # Add empty dict for failed parsing
            parsed_prompts.append({})
    
    # Convert list of dictionaries to DataFrame
    prompt_df: pd.DataFrame = pd.DataFrame(parsed_prompts)
    print(f"Total data: {len(prompt_df)}")
    prompt_df: pd.DataFrame = prompt_df[prompt_df['toxicity'].notna()]
    print(f"Total data with toxicity score: {len(prompt_df)}")
    
    return prompt_df


def show_toxicity_distribution(df: pd.DataFrame, save_path: Path, figure_name: str):
    print("=== Toxicity Score Distribution Analysis ===")
    print(f"Total toxic comments: {len(df)}")
    print(f"Toxicity score range: {df['toxicity'].min():.3f} - {df['toxicity'].max():.3f}")
    print(f"Mean toxicity: {df['toxicity'].mean():.3f}")
    print(f"Median toxicity: {df['toxicity'].median():.3f}")
    print(f"Standard deviation: {df['toxicity'].std():.3f}")

        # Show distribution by percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nToxicity score percentiles:")
    for p in percentiles:
        score = np.percentile(df['toxicity'], p)
        print(f"{p}th percentile: {score:.3f}")
        # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['toxicity'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Toxicity Scores')
    plt.xlabel('Toxicity Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path / f'{figure_name}_show_toxicity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nHistogram saved to {str(save_path)}/{figure_name}_show_toxicity_distribution.png")


def show_ppl_distribution(df: pd.DataFrame, save_path: Path, figure_name: str):
    print("=== PPL Score Distribution Analysis ===")
    print(f"Total toxic comments: {len(df)}")
    print(f"PPL score range: {df['ppl'].min():.3f} - {df['ppl'].max():.3f}")
    print(f"Mean PPL: {df['ppl'].mean():.3f}")
    print(f"Median PPL: {df['ppl'].median():.3f}")
    print(f"Standard deviation: {df['ppl'].std():.3f}")

        # Show distribution by percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nPPL score percentiles:")
    for p in percentiles:
        score = np.percentile(df['ppl'], p)
        print(f"{p}th percentile: {score:.3f}")
        # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['ppl'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of PPL Scores')
    plt.xlabel('PPL Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path / f'{figure_name}_ppl_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nHistogram saved to {str(save_path)}/{figure_name}_ppl_distribution.png")


# Define toxicity level bins for even sampling
def sample_evenly_by_toxicity_levels(
    df: pd.DataFrame, 
    n_samples_per_bin: int = 1000, 
    n_bins: int = 5, 
) -> pd.DataFrame:
    """
    Sample evenly across different toxicity levels
    
    Parameters:
    - df: DataFrame with toxicity scores
    - n_samples_per_bin: Number of samples to take from each bin
    - n_bins: Number of bins to divide the toxicity range into
    
    Returns:
    - sampled_df: DataFrame with evenly sampled toxic comments
    """
    # Create bins based on toxicity score range
    min_score = df['toxicity'].min()
    max_score = df['toxicity'].max()
    bin_edges: np.ndarry = np.linspace(min_score, max_score, n_bins + 1)
    
    print("\n=== Even Sampling Across Toxicity Levels ===")
    print(f"Creating {n_bins} bins from {min_score:.3f} to {max_score:.3f}")
    print(f"Target samples per bin: {n_samples_per_bin}")
    
    sampled_dfs = []
    
    for i in range(n_bins):
        bin_min: float = bin_edges[i]
        bin_max: float = bin_edges[i + 1]
        
        # For the last bin, include the maximum value
        if i == n_bins - 1:
            bin_mask: pd.Series = (df['toxicity'] >= bin_min) & (df['toxicity'] <= bin_max)
        else:
            bin_mask: pd.Series = (df['toxicity'] >= bin_min) & (df['toxicity'] < bin_max)
        
        bin_df: pd.DataFrame = df[bin_mask].copy()
        bin_size: int = len(bin_df)
        
        print(f"Bin {i+1}: [{bin_min:.3f}, {bin_max:.3f}{'inclusive' if i == n_bins - 1 else 'exclusive'}] - {bin_size} samples")
        
        if bin_size > 0:
            # Sample from this bin (with replacement if needed)
            if bin_size >= n_samples_per_bin:
                sampled_bin: pd.DataFrame = bin_df.sample(n=n_samples_per_bin, random_state=SEED)
            else:
                # If not enough samples, take all available
                sampled_bin: pd.DataFrame = bin_df.copy()
                print(f"  Warning: Only {bin_size} samples available, taking all")
            
            sampled_dfs.append(sampled_bin)
            print(f"  Sampled: {len(sampled_bin)} samples")
        else:
            print("  Warning: No samples in this bin")
    
    # Combine all sampled data
    if sampled_dfs:
        sampled_df: pd.DataFrame = pd.concat(sampled_dfs, ignore_index=True)
        print(f"\nTotal sampled toxic comments: {len(sampled_df)}")
        
        # Show distribution of sampled data
        print("\nSampled data toxicity distribution:")
        for i in range(n_bins):
            bin_min: float = bin_edges[i]
            bin_max: float = bin_edges[i + 1]
            if i == n_bins - 1:
                bin_count: int = len(sampled_df[(sampled_df['toxicity'] >= bin_min) & (sampled_df['toxicity'] <= bin_max)])
            else:
                bin_count: int = len(sampled_df[(sampled_df['toxicity'] >= bin_min) & (sampled_df['toxicity'] < bin_max)])
            print(f"  Bin {i+1}: {bin_count} samples")
        
        return sampled_df
    else:
        print("No samples collected!")
        return pd.DataFrame()


def compare_distribution(ori_df: pd.DataFrame, sampled_df: pd.DataFrame, figure_name: str, save_dir: Path):
    plt.figure(figsize=(15, 5))
    
    # Original distribution
    plt.subplot(1, 2, 1)
    plt.hist(ori_df['toxicity'], bins=50, alpha=0.7, edgecolor='black', color='blue')
    plt.title(f'Original Toxic Comments Distribution\n(n={len(ori_df)})')
    plt.xlabel('Toxicity Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Sampled distribution
    plt.subplot(1, 2, 2)
    plt.hist(sampled_df['toxicity'], bins=50, alpha=0.7, edgecolor='black', color='red')
    plt.title(f'Evenly Sampled Toxic Comments\n(n={len(sampled_df)})')
    plt.xlabel('Toxicity Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{figure_name}_toxicity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {str(save_dir)}/{figure_name}_toxicity_comparison.png")



def calculate_perplexity(text: str, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> float:
    """
    Calculates the perplexity of a given text using a GPT-2 model.

    Perplexity is a measurement of how well a probability model predicts a sample.
    In NLP, a lower perplexity score indicates that the language model is more confident
    in its prediction of the text sequence, which often correlates with higher
    grammatical and semantic quality.

    Args:
        text (str): The input string to evaluate.
        model (GPT2LMHeadModel): The pre-trained GPT-2 model.
        tokenizer (GPT2Tokenizer): The tokenizer for the model.

    Returns:
        float: The perplexity score of the text. Returns float('inf') for empty strings.
    """
    if not text:
        return float('inf')

    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(model.device) # Ensure tensor is on the same device as the model

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss

    perplexity = torch.exp(neg_log_likelihood)

    return perplexity.item()


def score_samples(df: pd.DataFrame, model_name: str = 'gpt2') -> pd.DataFrame:
    """
    Score the samples
    """
    ## score the samples with GPT2
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to('cuda')
    model.eval()
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Scoring samples'):
        ppl: float = calculate_perplexity(row['text'], model, tokenizer)
        df.at[index, 'ppl'] = ppl
    return df



def main():
    data_dir = get_project_dir() / 'data' / 'toxicity' / 'real_tox_prompts'
    download_save_data()
    df: pd.DataFrame = pd.read_csv(data_dir / "real-toxicity-prompts.csv")
    prompt_df: pd.DataFrame = process_prompt_column(df)    
    prompt_df: pd.DataFrame = score_samples(prompt_df)
    prompt_df.to_csv(data_dir / "processed_real_tox_prompts_scored.csv", index = False)
    print(f"Total data with PPL score: {len(prompt_df)}")
    
    ppl_threshold: int = 200 # threshold for PPL score
    prompt_df: pd.DataFrame = prompt_df[prompt_df['ppl'] < ppl_threshold]
    print(f"Total data with PPL score < 200: {len(prompt_df)}")
    show_ppl_distribution(prompt_df, data_dir, f'real_tox_prompts_scored_ppl_{ppl_threshold}')

    # Separate the data into toxic and non-toxic
    toxic_df = prompt_df[prompt_df['toxicity'] > 0.5]
    non_toxic_df = prompt_df[prompt_df['toxicity'] <= 0.5]
    print(f"Total toxic data: {len(toxic_df)}")
    print(f"Total non-toxic data: {len(non_toxic_df)}")

    toxic_df_sampled = sample_evenly_by_toxicity_levels(
        toxic_df, 
        n_samples_per_bin=1200,  # number of samples per bin
        n_bins=5  # number of bins
    )

    non_toxic_df_sampled = sample_evenly_by_toxicity_levels(
        non_toxic_df, 
        n_samples_per_bin=1200,  # number of samples per bin
        n_bins=5  # number of bins
    )

    df_sampled: pd.DataFrame = pd.concat([toxic_df_sampled, non_toxic_df_sampled])
    df_sampled.to_csv(data_dir / f"11k_evenly_sampled_real_tox_prompts_ppl_{ppl_threshold}.csv", index = False)

    compare_distribution(prompt_df, df_sampled, f'real_tox_prompts_ppl_{ppl_threshold}', data_dir)
    prompt_df = pd.read_csv(data_dir / f"11k_evenly_sampled_real_tox_prompts_ppl_{ppl_threshold}.csv")

    # # split into train val test
    print("-> Splitting into train val test")
    print("Number of toxic data: ", len(toxic_df_sampled))
    print("Number of non-toxic data: ", len(non_toxic_df_sampled))
    train_toxic_df, val_toxic_df, test_toxic_df = split_data(toxic_df_sampled)
    train_non_toxic_df, val_non_toxic_df, test_non_toxic_df = split_data(non_toxic_df_sampled)

    print(f"Total non-toxic data: train {len(train_non_toxic_df)}, val {len(val_non_toxic_df)}, test {len(test_non_toxic_df)}")
    print(f"Total toxic data: train {len(train_toxic_df)}, val {len(val_toxic_df)}, test {len(test_toxic_df)}")

    train_num = min(len(train_non_toxic_df), len(train_toxic_df))
    train_non_toxic_df = train_non_toxic_df.iloc[:train_num]
    train_toxic_df = train_toxic_df.iloc[:train_num]

    train_non_toxic_df.to_json(data_dir / 'train_0.jsonl', orient = 'records', lines = True)
    val_non_toxic_df.to_json(data_dir / 'val_0.jsonl', orient = 'records', lines = True)
    test_non_toxic_df.to_json(data_dir / 'test_0.jsonl', orient = 'records', lines = True)
    
    train_toxic_df.to_json(data_dir / 'train_1.jsonl', orient = 'records', lines = True)
    val_toxic_df.to_json(data_dir / 'val_1.jsonl', orient = 'records', lines = True)
    test_toxic_df.to_json(data_dir / 'test_1.jsonl', orient = 'records', lines = True)

if __name__ == "__main__":
    main()