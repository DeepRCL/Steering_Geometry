
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import value_stability.config as vs_config
from value_stability.pipeline import PerturbationPipeline



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the perturbation pipeline on the full dataset.")
    parser.add_argument(
        "--type",
        choices=["paraphrase", "adversarial", "both"],
        default="both",
        help="Which perturbation type to run (default: both).",
    )
    args = parser.parse_args()
    print(f"Input  : {vs_config.INPUT_CSV}")
    print(f"Type(s): {args.type}")
    pipe = PerturbationPipeline()
    if args.type in ("paraphrase", "both"):
        pipe.run_single_perturbation("paraphrase", vs_config.INPUT_CSV, vs_config.PARAPHRASE_OUTPUT_CSV)
    if args.type in ("adversarial", "both"):
        pipe.run_single_perturbation("adversarial", vs_config.INPUT_CSV, vs_config.ADVERSARIAL_OUTPUT_CSV)


if __name__ == "__main__":
    main()
