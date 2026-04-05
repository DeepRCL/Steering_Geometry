import pandas as pd
import os

def sample_dataset(input_path, sampled_output_path, remaining_output_path, sample_size=200):
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Shuffle the dataset randomly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    sampled_dfs = []
    remaining_dfs = []
    
    # Get unique values
    unique_values = df['value'].unique()
    
    for val in unique_values:
        val_df = df[df['value'] == val]
        
        if len(val_df) <= sample_size:
            # If fewer than sample_size records, take all
            sampled_dfs.append(val_df)
        else:
            # Sample 200 records
            sampled = val_df.sample(n=sample_size, random_state=42)
            sampled_dfs.append(sampled)
            
            # Keep the rest
            remaining = val_df.drop(sampled.index)
            remaining_dfs.append(remaining)
            
    # Concatenate results
    final_sampled_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # For remaining, we also need to include the values that had <= sample_size (if any were left)
    # Actually, if len(val_df) <= sample_size, remaining_dfs will not have anything for that value.
    # That is correct based on the requirement "The rest records that are not selected should be saved in another csv file".
    
    if remaining_dfs:
        final_remaining_df = pd.concat(remaining_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        final_remaining_df = pd.DataFrame(columns=df.columns)

    # Save to CSV
    final_sampled_df.to_csv(sampled_output_path, index=False)
    final_remaining_df.to_csv(remaining_output_path, index=False)
    
    print(f"Sampled dataset saved to: {sampled_output_path}")
    print(f"Remaining dataset saved to: {remaining_output_path}")
    print(f"Total sampled: {len(final_sampled_df)}")
    print(f"Total remaining: {len(final_remaining_df)}")

if __name__ == "__main__":
    input_csv = "dataset_construction/Touche23-ValueEval/data/touche_positive_only.csv"
    sampled_csv = "dataset_construction/Touche23-ValueEval/data/touche_positive_only_sampled_200.csv"
    remaining_csv = "dataset_construction/Touche23-ValueEval/data/touche_positive_only_remaining.csv"
    
    sample_dataset(input_csv, sampled_csv, remaining_csv)
