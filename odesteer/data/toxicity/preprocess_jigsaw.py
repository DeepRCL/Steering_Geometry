import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from odesteer.utils import get_project_dir
import pandas as pd

SEED = 20250415
random.seed(SEED)

def split_data(df: pd.DataFrame, val_test_size: int = 2000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = list(range(len(df)))
    random.shuffle(indices)
    df: pd.DataFrame = df.iloc[indices]
    train_size = len(df) - 2 * val_test_size
    train_df: pd.DataFrame = df.iloc[:train_size]
    val_df: pd.DataFrame = df.iloc[train_size:train_size + val_test_size]
    test_df: pd.DataFrame = df.iloc[train_size + val_test_size:]
    return train_df, val_df, test_df

# Define toxicity level bins for even sampling
def sample_evenly_by_toxicity_levels(df: pd.DataFrame, n_samples_per_bin: int = 1000, n_bins: int = 5) -> pd.DataFrame:
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
    bin_edges = np.linspace(min_score, max_score, n_bins + 1)
    
    print("\n=== Even Sampling Across Toxicity Levels ===")
    print(f"Creating {n_bins} bins from {min_score:.3f} to {max_score:.3f}")
    print(f"Target samples per bin: {n_samples_per_bin}")
    
    sampled_dfs = []
    
    for i in range(n_bins):
        bin_min = bin_edges[i]
        bin_max = bin_edges[i + 1]
        
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
                sampled_bin: pd.DataFrame = bin_df.sample(n = n_samples_per_bin, random_state=SEED)
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
        sampled_df = pd.concat(sampled_dfs, ignore_index = True)
        print(f"\nTotal sampled toxic comments: {len(sampled_df)}")
        
        # Show distribution of sampled data
        print("\nSampled data toxicity distribution:")
        for i in range(n_bins):
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]
            if i == n_bins - 1:
                bin_count = len(sampled_df[(sampled_df['toxicity'] >= bin_min) & (sampled_df['toxicity'] <= bin_max)])
            else:
                bin_count = len(sampled_df[(sampled_df['toxicity'] >= bin_min) & (sampled_df['toxicity'] < bin_max)])
            print(f"  Bin {i+1}: {bin_count} samples")
        
        return sampled_df
    else:
        print("No samples collected!")
        return pd.DataFrame()


def remove_dulicates(df: pd.DataFrame) -> pd.DataFrame:
    # remove duplicates
    dup_mask = df['comment_text'].duplicated('first')
    df = df[~dup_mask][['comment_text', 'toxicity']]
    df['comment_text'] = df['comment_text'].astype(str)
    return df

def show_toxicity_distribution(df: pd.DataFrame, figure_name: str):
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
    plt.savefig(data_dir / f'{figure_name}_show_toxicity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nHistogram saved to {str(data_dir)}/{figure_name}_show_toxicity_distribution.png")


def compare_distribution(ori_df, sampled_df, figure_name):
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
    plt.savefig(data_dir / f'{figure_name}_toxicity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {str(data_dir)}/{figure_name}_toxicity_comparison.png")
        


if __name__ == '__main__':
    data_dir = get_project_dir() / 'data' / 'toxicity' / 'jigsaw'
    if (data_dir / 'all_data_no_dup.csv').exists():
        jigsaw_df = pd.read_csv(data_dir / 'all_data_no_dup.csv')
    else:
        jigsaw_df = pd.read_csv(data_dir / 'all_data.csv')
        jigsaw_df = remove_dulicates(jigsaw_df)
        jigsaw_df.to_csv(data_dir / 'all_data_no_dup.csv', index = False)
    
    # separate toxic and non-toxic comments
    toxic_df = jigsaw_df[jigsaw_df['toxicity'] > 0.5]
    non_toxic_df = jigsaw_df[jigsaw_df['toxicity'] <= 0.5]
    
    # Analyze toxicity score distribution in toxic_df
    show_toxicity_distribution(toxic_df, 'toxic')
    show_toxicity_distribution(non_toxic_df, 'non_toxic')

    # Apply even sampling and Create comparison histogram on toxic df
    evenly_sampled_toxic_df = sample_evenly_by_toxicity_levels(
        toxic_df, 
        n_samples_per_bin=1000,  # Adjust this number based on your needs
        n_bins=5  # Adjust number of bins as needed
    )

    if len(evenly_sampled_toxic_df) > 0:
        print(f"\nUsing evenly sampled toxic_df with {len(toxic_df)} samples for further processing")
        compare_distribution(toxic_df, evenly_sampled_toxic_df, 'toxic')
        toxic_df = evenly_sampled_toxic_df  # Use evenly sampled version

    # Apply even sampling and Create comparison histogram on non-toxic df
    evenly_sampled_non_toxic_df = sample_evenly_by_toxicity_levels(
        non_toxic_df, 
        n_samples_per_bin=1000,  # Adjust this number based on your needs
        n_bins=5  # Adjust number of bins as needed
    )
    if len(evenly_sampled_non_toxic_df) > 0:
        print(f"\nUsing evenly sampled non_toxic_df with {len(non_toxic_df)} samples for further processing")
        compare_distribution(non_toxic_df, evenly_sampled_non_toxic_df, 'non_toxic')
        non_toxic_df = evenly_sampled_non_toxic_df  # Use evenly sampled version


    assert len(non_toxic_df) == len(toxic_df), "The number of non-toxic and toxic comments should be the same"
    non_toxic_df.to_json(data_dir / 'train_0.jsonl', orient = 'records', lines = True)
    toxic_df.to_json(data_dir / 'train_1.jsonl', orient = 'records', lines = True)