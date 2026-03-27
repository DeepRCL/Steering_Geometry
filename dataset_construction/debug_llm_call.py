import pandas as pd
from pathlib import Path
from pipeline import DatasetConstructionPipeline

# Load the CSV to get the first row
print("Loading CSV...")
csv_path = Path(__file__).resolve().parent / "data" / "dataset_positive_only.csv"
df = pd.read_csv(csv_path)
row = df.iloc[10]
print(f"Row: {row}")

# Initialize pipelines
model_id = "Qwen/Qwen3.5-2B"
print(f"Initializing model: {model_id}")
pipeline = DatasetConstructionPipeline(model_id=model_id, max_new_tokens=1024)

# Create the answer
print("\nCREATE ANSWER")
print("-"*80)
try:
    result = pipeline.create_answer(row)
    print(f"Result: {result}\n")
except Exception as e:
    print(f"ERROR: {e}\n")
    import traceback
    traceback.print_exc()