import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline import DatasetConstructionPipeline


def run(pipe: DatasetConstructionPipeline, mode: str, method: str) -> None:
    """
    Args:
        pipe:   initialised pipeline
        mode:   "negative" — generate negative answers from positive_answer col
                "positive" — generate positive answers from negative_answer col
        method: "single"   — one model call per row  (safe on CPU / Mac)
                "batch"    — one model call per batch (faster on GPU)
    """
    target_col = "negative_answer" if mode == "negative" else "positive_answer"
    output_csv = config.OUTPUT_CSV.parent / f"dataset_{target_col}.csv"

    kwargs = dict(
        input_csv=config.INPUT_CSV,
        output_csv=output_csv,
        target_col=target_col,
        mode=mode,
        batch_size=config.BATCH_SIZE,
    )

    if method == "single":
        pipe.build_dataset_single(**kwargs)
    else:
        pipe.build_dataset_batch(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset construction pipeline.")
    parser.add_argument(
        "--mode", choices=["negative", "positive"], default="negative",
        help="Which answer type to generate (default: negative)",
    )
    parser.add_argument(
        "--method", choices=["single", "batch"], default="single",
        help="single: one call per row | batch: one call per batch (default: single)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = DatasetConstructionPipeline()
    run(pipe, mode=args.mode, method=args.method)


if __name__ == "__main__":
    main()
