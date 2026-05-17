import pandas as pd
from odesteer.utils import get_project_dir

def format_jigsaw_realtoxicprompts(non_toxic_df: pd.DataFrame, toxic_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([non_toxic_df, toxic_df], axis = 0)
    # Only keep the whole sentence as text column
    if 'comment_text' in df.columns:
        df = df.rename({'comment_text': 'text', 'toxicity': 'label'}, axis = 1)
    elif 'text' in df.columns:
        df = df.rename({'toxicity': 'label'}, axis = 1)

    df = df[["text", "label"]]
    # Concatenate strings of prompt, neg_resp, and pos_resp and check duplicates
    df_contents: pd.Series = df.text + df.label.astype(str)
    duplicated_indices = df_contents.duplicated()
    if len(duplicated_indices) > 0:
        print(f"Removed {duplicated_indices.sum()} duplicated samples.")
    return df[~duplicated_indices]


if __name__ == '__main__':
    # Jigsaw dataset
    data_dir = get_project_dir() / 'data' / 'toxicity' / 'jigsaw'
    non_toxic_df = pd.read_json(data_dir / 'train_0.jsonl', orient = 'records', lines = True)
    toxic_df = pd.read_json(data_dir / 'train_1.jsonl', orient = 'records', lines = True)
    df = format_jigsaw_realtoxicprompts(non_toxic_df, toxic_df)
    df.to_json(data_dir / 'final_train.jsonl', orient = 'records', lines = True)
    
    # Real Toxic Prompts dataset
    data_dir = get_project_dir() / 'data' / 'toxicity' / 'real_tox_prompts'
    train_non_toxic_df = pd.read_json(data_dir / 'train_0.jsonl', orient = 'records', lines = True)
    train_toxic_df = pd.read_json(data_dir / 'train_1.jsonl', orient = 'records', lines = True)
    
    val_non_toxic_df = pd.DataFrame({'text': [], 'toxicity': []})
    val_toxic_df = pd.read_json(data_dir / 'val_1.jsonl', orient = 'records', lines = True)
    
    test_non_toxic_df = pd.DataFrame({'text': [], 'toxicity': []})
    test_toxic_df = pd.read_json(data_dir / 'test_1.jsonl', orient = 'records', lines = True)
    
    train_df = format_jigsaw_realtoxicprompts(train_non_toxic_df, train_toxic_df)
    val_df = format_jigsaw_realtoxicprompts(val_non_toxic_df, val_toxic_df)
    test_df = format_jigsaw_realtoxicprompts(test_non_toxic_df, test_toxic_df)
    
    train_df.to_json(data_dir / 'final_train.jsonl', orient = 'records', lines = True)
    val_df.to_json(data_dir / 'final_val.jsonl', orient = 'records', lines = True)
    test_df.to_json(data_dir / 'final_test.jsonl', orient = 'records', lines = True)