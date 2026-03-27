import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import config
from pipeline import DatasetConstructionPipeline

print("Loading CSV...")
df = pd.read_csv(config.INPUT_CSV)
sample = df.head(config.DEBUG_ROWS)

print(f"Initializing model: {config.MODEL_ID}")
pipe = DatasetConstructionPipeline()  # all defaults come from config / .env

results = []
for i, (_, row) in enumerate(sample.iterrows()):
    print(f"\n[{i+1}/{config.DEBUG_ROWS}] question: {row['question']}")
    print("-" * 80)
    try:
        neg = pipe.create_answer(row, mode="negative")
        pos = (
            pipe.create_answer(row, mode="positive")
            if pd.notna(row.get("negative_answer")) and str(row.get("negative_answer", "")).strip()
            else None
        )
        print(f"  negative_answer: {neg}")
        if pos:
            print(f"  positive_answer (from existing negative): {pos}")
        results.append({**row.to_dict(), "generated_negative": neg})
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append({**row.to_dict(), "generated_negative": f"ERROR: {e}"})

out_path = config.INPUT_CSV.parent / "debug_output.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"\nSaved debug output → {out_path}")
