from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Dict, List

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CAA.Geometry.representation_geometry import analyze_representation_geometry, mean_center_vectors
from pipeline.config import SCHWARTZ_CIRCUMPLEX_ORDER, SteeringConfig
from pipeline.steering_pipeline import SteeringPipeline
from pipeline import data_utils


def _load_config(config_path: Path) -> SteeringConfig:
    with config_path.open() as f:
        data = json.load(f)
    return SteeringConfig(**data)


def _infer_layer(config: SteeringConfig, explicit_layer: int | None) -> int:
    if explicit_layer is not None:
        return explicit_layer

    match = re.search(r"layer_(\d+)", config.output_dir)
    if match:
        return int(match.group(1))

    if config.layer_sweep_candidates and len(config.layer_sweep_candidates) == 1:
        return int(config.layer_sweep_candidates[0])

    raise ValueError(
        "Could not infer the target layer from the saved OPT config. "
        "Please pass --layer explicitly."
    )


def _collect_positive_activation_means(
    pipeline: SteeringPipeline,
    layer: int,
) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for value in SCHWARTZ_CIRCUMPLEX_ORDER:
        rows = data_utils.get_rows_for_value(pipeline.train_rows, value)
        if pipeline.config.n_training_samples is not None and pipeline.config.n_training_samples < len(rows):
            rng = random.Random(pipeline.config.random_seed)
            rows = rng.sample(rows, pipeline.config.n_training_samples)

        if not rows:
            d_model = pipeline.model.config.hidden_size
            result[value] = torch.zeros(d_model, dtype=torch.float32)
            continue

        acts: List[torch.Tensor] = []
        for row in rows:
            prompt = data_utils.format_prompt(
                row["question"],
                pipeline.tokenizer,
                pipeline.config.use_chat_template,
                pipeline.config.prompt_template,
            )
            pos_text = prompt + " " + row["positive_answer"]
            acts.append(pipeline._extract_last_token_activation(pos_text, layer))

        result[value] = torch.stack(acts).mean(dim=0)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run theory-alignment geometry directly on raw positive last-token activations for llm-steering-opt."
    )
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--raw_subdir", default="activation_geometry_raw")
    parser.add_argument("--centered_subdir", default="activation_geometry_centered")
    args = parser.parse_args()

    config = _load_config(args.config_path)
    layer = _infer_layer(config, args.layer)

    pipeline = SteeringPipeline(config)
    pipeline.load_model()
    pipeline.prepare_data()

    raw_vectors = _collect_positive_activation_means(pipeline, layer)
    centered_vectors = mean_center_vectors(raw_vectors)

    common_metadata = {
        "representation_type": "mean_positive_last_token_activation",
        "selected_layer": layer,
        "model_name": config.model_name,
        "split": "train",
        "normalization_note": "geometry uses cosine similarity after per-value vector normalization",
    }

    analyze_representation_geometry(
        raw_vectors,
        relations_path=config.relations_path,
        output_dir=Path(config.output_dir) / args.raw_subdir,
        seed=config.random_seed,
        title_prefix="Raw Activation ",
        metadata={**common_metadata, "transform": "none"},
    )

    analyze_representation_geometry(
        centered_vectors,
        relations_path=config.relations_path,
        output_dir=Path(config.output_dir) / args.centered_subdir,
        seed=config.random_seed,
        title_prefix="Centered Activation ",
        metadata={**common_metadata, "transform": "mean_center"},
    )

    print(f"Wrote {Path(config.output_dir) / args.raw_subdir}")
    print(f"Wrote {Path(config.output_dir) / args.centered_subdir}")


if __name__ == "__main__":
    main()
