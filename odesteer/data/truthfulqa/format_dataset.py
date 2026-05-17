import pandas as pd
from datasets import load_dataset

from odesteer.utils import get_project_dir

gen_dir = get_project_dir() / 'data' / 'truthfulqa' / 'texts'
gen_dir.mkdir(parents = True, exist_ok = True)


def format_truthfulqa_generation():
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split = "validation")
    ds = ds.train_test_split(test_size = 0.5, seed = 42)
    df1, df2 = ds['train'].to_pandas(), ds['test'].to_pandas()
    for df_idx, df in enumerate([df1, df2]):
        pos_pairs, neg_pairs = [], []
        for idx, row in df.iterrows():
            for correct_answer in row.correct_answers:
                pos_pairs.append({
                    'idx': idx,
                    'question': row.question.strip(),
                    'answer': correct_answer,
                })
            for incorrect_answer in row.incorrect_answers:
                neg_pairs.append({
                    'idx': idx,
                    'question': row.question.strip(),
                    'answer': incorrect_answer,
                })
        pos_df = pd.DataFrame(pos_pairs)
        neg_df = pd.DataFrame(neg_pairs)
        pos_df.to_json(gen_dir / f'pos_{df_idx}.jsonl', orient = 'records', lines = True)
        neg_df.to_json(gen_dir / f'neg_{df_idx}.jsonl', orient = 'records', lines = True)
        print(f'Saved {len(pos_df)} positive pairs and {len(neg_df)} negative pairs for {df_idx}th split')

    
if __name__ == '__main__':
    format_truthfulqa_generation()