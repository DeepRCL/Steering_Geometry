import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline import DatasetConstructionPipeline

if __name__ == "__main__":
    pipe = DatasetConstructionPipeline() 
    pipe.build_dataset_single(
        input_csv=config.INPUT_CSV,
        output_csv=config.OUTPUT_CSV,
        target_col="negative_answer",
        batch_size=config.BATCH_SIZE,
    )
