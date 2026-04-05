"""
Entry point for the Touche23-ValueEval negative-answer generation pipeline.

Usage:
    python dataset_construction/Touche23-ValueEval/run_pipelines.py [--method METHOD] [--input INPUT] [--output OUTPUT]

Options:
    --method single   One model call per row  (default; safe on CPU / Mac)
    --method batch    One model call per batch (faster on GPU)
    --input  INPUT    Path to input CSV (default: data/touche_positive_only.csv)
    --output OUTPUT   Path to output CSV (default: data/touche_dataset_negative_answer.csv)

The script is resumable: if interrupted, re-run the same command and it will
skip already-completed rows, picking up where it left off.
"""

import sys
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE.parent))  # Touche23-ValueEval/ for prompt.py

import config
from pipeline import TouchePipeline  # local (Touche23-ValueEval/)

_DATA_DIR   = _HERE.parent / "data"
_DEFAULT_INPUT  = _DATA_DIR / "touche_positive_only.csv"
_DEFAULT_OUTPUT = _DATA_DIR / "touche_dataset_negative_answer.csv"
_DIRECTION  = "positive_to_negative"


def run(pipe: TouchePipeline, method: str, input_csv: Path, output_csv: Path) -> None:
    """
    Run the pipeline.

    Parameters
    ----------
    pipe   : initialised TouchePipeline
    method : "single" — one model call per row
             "batch"  — one model call per batch
    input_csv : Path to input CSV
    output_csv : Path to output CSV
    """
    kwargs = dict(
        input_csv=input_csv,
        output_csv=output_csv,
        direction=_DIRECTION,
        batch_size=config.BATCH_SIZE,
    )
    if method == "single":
        pipe.build_dataset_single(**kwargs)
    else:
        pipe.build_dataset_batch(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate negative answers for Touche23-ValueEval dataset."
    )
    parser.add_argument(
        "--method",
        choices=["single", "batch"],
        default="single",
        help="single: one call per row (default) | batch: one call per batch",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(_DEFAULT_INPUT),
        help=f"Path to input CSV (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help=f"Path to output CSV (default: {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = TouchePipeline()
    run(
        pipe, 
        method=args.method, 
        input_csv=Path(args.input), 
        output_csv=Path(args.output)
    )


if __name__ == "__main__":
    main()
