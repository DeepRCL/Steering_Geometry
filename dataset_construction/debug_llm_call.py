import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import config
from pipeline import DatasetConstructionPipeline

TARGET_COL  = "negative_answer"
DEBUG_INPUT  = config.INPUT_CSV.parent / "debug_input.csv"
DEBUG_OUTPUT = config.INPUT_CSV.parent / "debug_output.csv"

# Create debug_input.csv only once; reuse on subsequent runs
if not DEBUG_INPUT.exists():
    print("Loading CSV and creating debug sample...")
    df = pd.read_csv(config.INPUT_CSV)
    sample = df.head(config.DEBUG_ROWS).copy()
    sample[TARGET_COL] = sample[TARGET_COL].astype(object)
    sample.to_csv(DEBUG_INPUT, index=False)
    print(f"Saved {config.DEBUG_ROWS}-row sample → {DEBUG_INPUT}")
else:
    print(f"Reusing existing debug sample → {DEBUG_INPUT}")

print(f"Initializing model: {config.MODEL_ID}")
pipe = DatasetConstructionPipeline()
pipe.build_dataset(
    input_csv=DEBUG_INPUT,
    output_csv=DEBUG_OUTPUT,
    target_col=TARGET_COL,
    mode="negative",
    batch_size=config.BATCH_SIZE,
)
