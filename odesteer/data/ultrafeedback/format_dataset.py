import pandas as pd
from datasets import load_dataset

from odesteer.utils import get_project_dir


gen_dir = get_project_dir() / 'data' / 'ultrafeedback' / 'texts'
gen_dir.mkdir(parents = True, exist_ok = True)


def format_ultrafeedback_binarized(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Format ultrafeedback_binarized dataset to standard format."""
    # Filter for samples where score_chosen > score_rejected
    if 'score_chosen' in raw_df.columns and 'score_rejected' in raw_df.columns:
        score_filtered_df = raw_df[raw_df['score_chosen'] > raw_df['score_rejected']].copy()
        print(f"Score filtering: {len(raw_df)} -> {len(score_filtered_df)} samples (removed {len(raw_df) - len(score_filtered_df)} with score_chosen <= score_rejected)")
    else:
        score_filtered_df = raw_df.copy()
        print("Warning: score columns not found, skipping score filtering")

    # Extract the last assistant response from chosen and rejected
    def extract_assistant_response(conversation):
        # Handle both list and numpy array cases
        if hasattr(conversation, '__iter__') and not isinstance(conversation, str):
            conversation_list = list(conversation)
            for msg in reversed(conversation_list):
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    return msg.get('content', '')
        return ''

    pos_pairs, neg_pairs = [], []
    for idx, row in score_filtered_df.iterrows():
        prompt = row['prompt']
        pos_resp = extract_assistant_response(row['chosen'])
        neg_resp = extract_assistant_response(row['rejected'])

        # Skip if either response is empty
        if pos_resp and neg_resp:
            pos_pairs.append({
                'idx': idx,
                'prompt': prompt,
                'response': pos_resp,
            })
            neg_pairs.append({
                'idx': idx,
                'prompt': prompt,
                'response': neg_resp,
            })

    return pd.DataFrame(pos_pairs), pd.DataFrame(neg_pairs)


def format_dataset():
    """Load and format UltrafeedbackBinarized dataset."""
    print("Loading UltrafeedbackBinarized dataset...")

    # Load train_prefs and test_prefs splits (following original preprocess.py)
    train_prefs_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", "default", split="train_prefs")
    test_prefs_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", "default", split="test_prefs")

    print(f"Loaded {len(train_prefs_ds)} train_prefs and {len(test_prefs_ds)} test_prefs samples")

    import numpy as np
    rng = np.random.default_rng(42)

    # Process train_prefs split -> train + val data (following original logic)
    print("\nProcessing train_prefs split...")
    train_prefs_df = train_prefs_ds.to_pandas()
    all_train_pos_df, all_train_neg_df = format_ultrafeedback_binarized(train_prefs_df)

    # Remove duplicates
    all_train_pos_df = all_train_pos_df.drop_duplicates(subset=['prompt']).reset_index(drop=True)
    all_train_neg_df = all_train_neg_df.drop_duplicates(subset=['prompt']).reset_index(drop=True)

    print(f"After formatting and dedup: {len(all_train_pos_df)} samples available from train_prefs")

    # Split train_prefs into train (10k) and val (500)
    # Following original logic: sample 10k for train, 500 for val
    n_available = len(all_train_pos_df)
    train_size = min(10000, n_available - 500)
    val_size = min(500, n_available - train_size)

    if n_available < 10500:
        print(f"Warning: Only {n_available} samples available, adjusting sizes...")
        print(f"Adjusted sizes - Train: {train_size}, Val: {val_size}")

    # Randomly sample indices
    available_indices = np.arange(n_available)
    selected_indices = rng.choice(available_indices, size=train_size + val_size, replace=False)

    train_indices = selected_indices[:train_size]
    val_indices = selected_indices[train_size:]

    # Save train split
    train_pos_df = all_train_pos_df.iloc[train_indices].reset_index(drop=True)
    train_neg_df = all_train_neg_df.iloc[train_indices].reset_index(drop=True)
    train_pos_df.to_json(gen_dir / 'train_pos.jsonl', orient='records', lines=True)
    train_neg_df.to_json(gen_dir / 'train_neg.jsonl', orient='records', lines=True)
    print(f"Saved {len(train_pos_df)} train positive and {len(train_neg_df)} train negative pairs")

    # Save val split
    val_pos_df = all_train_pos_df.iloc[val_indices].reset_index(drop=True)
    val_neg_df = all_train_neg_df.iloc[val_indices].reset_index(drop=True)
    val_pos_df.to_json(gen_dir / 'val_pos.jsonl', orient='records', lines=True)
    val_neg_df.to_json(gen_dir / 'val_neg.jsonl', orient='records', lines=True)
    print(f"Saved {len(val_pos_df)} validation positive and {len(val_neg_df)} validation negative pairs")

    # Process test_prefs split -> test data
    print("\nProcessing test_prefs split...")
    test_prefs_df = test_prefs_ds.to_pandas()
    all_test_pos_df, all_test_neg_df = format_ultrafeedback_binarized(test_prefs_df)

    print(f"After formatting: {len(all_test_pos_df)} test samples available")

    # Sample 500 for test (or all if less than 500)
    test_size = min(500, len(all_test_pos_df))
    test_available_indices = np.arange(len(all_test_pos_df))
    test_selected_indices = rng.choice(test_available_indices, size=test_size, replace=False)

    test_pos_df = all_test_pos_df.iloc[test_selected_indices].reset_index(drop=True)
    test_neg_df = all_test_neg_df.iloc[test_selected_indices].reset_index(drop=True)
    test_pos_df.to_json(gen_dir / 'test_pos.jsonl', orient='records', lines=True)
    test_neg_df.to_json(gen_dir / 'test_neg.jsonl', orient='records', lines=True)
    print(f"Saved {len(test_pos_df)} test positive and {len(test_neg_df)} test negative pairs")

    print(f"\nFinal dataset sizes - Train: {len(train_pos_df)}, Val: {len(val_pos_df)}, Test: {len(test_pos_df)}")

    print("\n✓ Dataset formatting complete!")


if __name__ == '__main__':
    format_dataset()
