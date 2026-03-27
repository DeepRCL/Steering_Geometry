import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import config
from pipeline import DatasetConstructionPipeline

TARGET_COL = "negative_answer"

print("Loading CSV...")
df = pd.read_csv(config.INPUT_CSV)
sample = df.head(config.DEBUG_ROWS).copy()

if TARGET_COL not in sample.columns:
    sample[TARGET_COL] = ""
sample[TARGET_COL] = sample[TARGET_COL].astype(object)

print(f"Initializing model: {config.MODEL_ID}")
pipe = DatasetConstructionPipeline()

for i, (idx, row) in enumerate(sample.iterrows()):
    print(f"\n[{i+1}/{config.DEBUG_ROWS}] question: {row['question']}")
    print("-" * 80)
    try:
        sample.at[idx, TARGET_COL] = pipe.create_answer(row, mode="negative")
        print(f"  {TARGET_COL}: {sample.at[idx, TARGET_COL]}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sample.at[idx, TARGET_COL] = f"ERROR: {e}"

out_path = config.INPUT_CSV.parent / "debug_output.csv"
sample.to_csv(out_path, index=False)
print(f"\nSaved debug output → {out_path}")
