import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CAA.Geometry.config import PipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from SAE.SparseCAA.config import SparseCAAPipelineConfig

STAGE1_CAA_CONFIGS = [
    ROOT / "CAA/Geometry/outputs/gemma_3_4b_pt_mid50_75_v1_centered_renorm/google__gemma-3-4b-pt/config.json",
    ROOT / "CAA/Geometry/outputs/gemma_3_12b_pt_v3_centered_renorm/google__gemma-3-12b-pt/config.json",
    ROOT / "CAA/Geometry/outputs/qwen2_5_7b_v3_centered_renorm/Qwen__Qwen2.5-7B/config.json",
    ROOT / "CAA/Geometry/outputs/qwen2_5_14b_v3_centered_renorm/Qwen__Qwen2.5-14B/config.json",
    ROOT / "CAA/Geometry/outputs/qwen2_5_32b_v3_centered_renorm/Qwen__Qwen2.5-32B/config.json",
    ROOT / "CAA/Geometry/outputs/qwen3_5_0_8b_base_v3_centered_renorm/Qwen__Qwen3.5-0.8B-Base/config.json",
    ROOT / "CAA/Geometry/outputs/qwen3_5_2b_base_v3_centered_renorm/Qwen__Qwen3.5-2B-Base/config.json",
    ROOT / "CAA/Geometry/outputs/qwen3_5_4b_base_v3_centered_renorm/Qwen__Qwen3.5-4B-Base/config.json",
    ROOT / "CAA/Geometry/outputs/qwen3_5_9b_base_v2_centered_renorm/Qwen__Qwen3.5-9B-Base/config.json",
    ROOT / "CAA/Geometry/outputs/gemma_4_31b_v4_centered_renorm/google__gemma-4-31B/config.json",
]

STAGE2_CAA_CONFIGS = [
    ROOT / "CAA/Geometry/outputs/qwen3_5_9b_base_v2_centered_renorm/Qwen__Qwen3.5-9B-Base/config.json",
    ROOT / "CAA/Geometry/outputs/qwen3_5_9b_v3_centered_renorm/Qwen__Qwen3.5-9B/config.json",
]

STAGE2_SPARSE_CONFIGS = [
    ROOT / "SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B-Base/pipeline_config.json",
    ROOT / "SAE/SparseCAA/outputs/Qwen__Qwen3.5-9B/pipeline_config.json",
]

OPT_NOTE = (
    "OPT metrics used in the paper are currently sourced from the hard-coded "
    "OPT_METRICS block in CAA/generate_geometry_tables.py. No local llm-steering-opt "
    "output directory matching those paper rows was found in this repo."
)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def list_paper_runs() -> None:
    print("Stage 1 CAA configs:")
    for path in STAGE1_CAA_CONFIGS:
        print(f"  - {_relative(path)}")

    print("\nStage 2 CAA configs:")
    for path in STAGE2_CAA_CONFIGS:
        print(f"  - {_relative(path)}")

    print("\nStage 2 SparseCAA configs:")
    for path in STAGE2_SPARSE_CONFIGS:
        print(f"  - {_relative(path)}")

    print("\nOPT:")
    print(f"  - {OPT_NOTE}")


def _refresh_caa_config(config_path: Path) -> None:
    import torch

    from CAA.Geometry.geometry import analyze_geometry as analyze_caa_geometry

    config = PipelineConfig.load(str(config_path))
    layer_path = Path(config.model_output_dir) / "layer_selection" / "selected_layer.json"
    if layer_path.exists():
        with layer_path.open() as f:
            selected_layer = json.load(f)["selected_layer"]
    else:
        experiment_metadata_path = Path(config.model_output_dir) / "experiment_metadata.json"
        evaluation_summary_path = Path(config.model_output_dir) / "evaluation" / "evaluation_summary.json"
        if experiment_metadata_path.exists():
            with experiment_metadata_path.open() as f:
                selected_layer = json.load(f)["selected_layer"]
        elif evaluation_summary_path.exists():
            with evaluation_summary_path.open() as f:
                selected_layer = json.load(f)["layer_idx"]
        elif config.layer_override is not None:
            selected_layer = config.layer_override
        else:
            first_value_dir = Path(config.model_output_dir) / "vectors" / safe_name(SCHWARTZ_CIRCUMPLEX_ORDER[0])
            candidate_layers = sorted(
                int(path.stem.split("_")[1])
                for path in first_value_dir.glob("layer_*.pt")
            )
            if len(candidate_layers) != 1:
                raise FileNotFoundError(
                    f"Could not determine a unique selected layer for {config.model_output_dir}."
                )
            selected_layer = candidate_layers[0]

    vectors = {}
    for value in SCHWARTZ_CIRCUMPLEX_ORDER:
        vec_path = Path(config.model_output_dir) / "vectors" / safe_name(value) / f"layer_{selected_layer}.pt"
        vectors[value] = torch.load(vec_path, map_location="cpu")

    analyze_caa_geometry(config, vectors)


def _refresh_sparse_config(config_path: Path) -> None:
    import torch

    from SAE.SparseCAA.geometry import run_geometry as analyze_sparse_geometry

    config = SparseCAAPipelineConfig.load(str(config_path))
    vectors = {}
    for value in SCHWARTZ_CIRCUMPLEX_ORDER:
        vec_path = Path(config.sparse_vectors_dir) / f"{safe_name(value)}.pt"
        vectors[value] = torch.load(vec_path, map_location="cpu")

    analyze_sparse_geometry(config, vectors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List or refresh the geometry outputs currently used by the paper tables."
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print the paper-used directories; do not regenerate any figures.",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip the stage-1 CAA family/size runs when refreshing outputs.",
    )
    args = parser.parse_args()

    list_paper_runs()
    if args.list_only:
        return

    if not args.skip_stage1:
        for config_path in STAGE1_CAA_CONFIGS:
            print(f"\nRefreshing stage-1 CAA geometry: {_relative(config_path)}")
            _refresh_caa_config(config_path)

    for config_path in STAGE2_CAA_CONFIGS:
        print(f"\nRefreshing stage-2 CAA geometry: {_relative(config_path)}")
        _refresh_caa_config(config_path)

    for config_path in STAGE2_SPARSE_CONFIGS:
        print(f"\nRefreshing stage-2 SparseCAA geometry: {_relative(config_path)}")
        _refresh_sparse_config(config_path)

    print(f"\nOPT note: {OPT_NOTE}")


if __name__ == "__main__":
    main()
