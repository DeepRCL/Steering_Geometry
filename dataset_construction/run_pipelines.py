import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline import DatasetConstructionPipeline
from prompt import PROMPT_CONFIG


def run(pipe: DatasetConstructionPipeline, direction: str, method: str) -> None:
    """
    Args:
        pipe:      initialised pipeline
        direction: PROMPT_CONFIG key, e.g. "positive_to_negative" or "negative_to_positive"
        method:    "single" — one model call per row  (safe on CPU / Mac)
                   "batch"  — one model call per batch (faster on GPU)
    """
    target_col = PROMPT_CONFIG[direction]["target_col"]
    output_csv = config.OUTPUT_CSV.parent / f"dataset_{target_col}.csv"

    kwargs = dict(
        input_csv=config.INPUT_CSV,
        output_csv=output_csv,
        direction=direction,
        batch_size=config.BATCH_SIZE,
    )

    if method == "single":
        pipe.build_dataset_single(**kwargs)
    else:
        pipe.build_dataset_batch(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset construction pipeline.")
    parser.add_argument(
        "--direction", choices=list(PROMPT_CONFIG.keys()), default="positive_to_negative",
        help="Direction of answer generation (default: positive_to_negative)",
    )
    parser.add_argument(
        "--method", choices=["single", "batch"], default="batch",
        help="single: one call per row | batch: one call per batch (default: single)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = DatasetConstructionPipeline()
    run(pipe, direction=args.direction, method=args.method)


if __name__ == "__main__":
    main()
