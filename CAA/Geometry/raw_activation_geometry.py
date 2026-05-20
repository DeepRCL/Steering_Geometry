from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAA.Geometry.config import PipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name
from CAA.Geometry.data_loader import DataLoader, PromptFormatter
from CAA.Geometry.model_loader import get_decoder_layers, load_model
from CAA.Geometry.representation_geometry import analyze_representation_geometry, mean_center_vectors


def _resolve_model_dir_from_config(config_path: Path) -> Path:
    config = PipelineConfig.load(str(config_path))
    return (PROJECT_ROOT / config.model_output_dir).resolve()


def _resolve_source_model_dir(model_dir: Path) -> Path:
    experiment_metadata_path = model_dir / "experiment_metadata.json"
    if not experiment_metadata_path.exists():
        return model_dir.resolve()

    with experiment_metadata_path.open() as f:
        metadata = json.load(f)

    source_output_dir = metadata.get("source_output_dir")
    if not source_output_dir:
        return model_dir.resolve()

    config_path = model_dir / "config.json"
    config = PipelineConfig.load(str(config_path))
    return (PROJECT_ROOT / source_output_dir / config.model_name_safe).resolve()


def _resolve_selected_layer(model_dir: Path) -> int:
    selected_layer_path = model_dir / "layer_selection" / "selected_layer.json"
    if selected_layer_path.exists():
        with selected_layer_path.open() as f:
            return int(json.load(f)["selected_layer"])

    experiment_metadata_path = model_dir / "experiment_metadata.json"
    if experiment_metadata_path.exists():
        with experiment_metadata_path.open() as f:
            data = json.load(f)
        if "selected_layer" in data:
            return int(data["selected_layer"])

    evaluation_summary_path = model_dir / "evaluation" / "evaluation_summary.json"
    if evaluation_summary_path.exists():
        with evaluation_summary_path.open() as f:
            data = json.load(f)
        if "layer_idx" in data:
            return int(data["layer_idx"])

    raise FileNotFoundError(f"Could not resolve selected layer for {model_dir}")


def _format_positive_tokens(instance, formatter: PromptFormatter) -> list[int]:
    q_text = instance.question
    pos_is_a = instance.pos_is_a

    if pos_is_a:
        a_text = instance.positive_answer
        b_text = instance.negative_answer
        pos_option = "A"
    else:
        a_text = instance.negative_answer
        b_text = instance.positive_answer
        pos_option = "B"

    prompt_text = (
        f"Question: {q_text}\n"
        f"(A) {a_text}\n"
        f"(B) {b_text}"
    )

    base_prompt = formatter._format_base_prompt(prompt_text)
    pos_prompt = base_prompt + f" ({pos_option}"
    return formatter.tokenizer.encode(pos_prompt, add_special_tokens=True)


def _collect_positive_activation_means(
    config: PipelineConfig,
    model_name: str,
    selected_layer: int,
    split: str,
) -> dict[str, torch.Tensor]:
    data_loader = DataLoader(config.dataset_path, eval_split=config.eval_split, seed=config.seed)
    model_info = load_model(model_name, device=config.device)
    formatter = PromptFormatter(model_info.tokenizer, model_info.is_instruct)
    decoder_layers = get_decoder_layers(model_info)

    current_activations: dict[int, torch.Tensor] = {}

    def _hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        current_activations[selected_layer] = hidden_states[0, -1, :].detach().to(device="cpu", dtype=torch.float32)

    handle = decoder_layers[selected_layer].register_forward_hook(_hook)
    vectors: dict[str, torch.Tensor] = {}

    try:
        for value in SCHWARTZ_CIRCUMPLEX_ORDER:
            if split == "train":
                instances = data_loader.get_train_pairs(value)
            else:
                instances = data_loader.get_eval_instances(value)

            if not instances:
                vectors[value] = torch.zeros(model_info.hidden_dim, dtype=torch.float32)
                continue

            acts = []
            for instance in tqdm(instances, desc=f"Raw activation [{split}] {value}", leave=False):
                tokens = _format_positive_tokens(instance, formatter)
                current_activations.clear()
                input_ids = torch.tensor([tokens]).to(model_info.device)
                with torch.no_grad():
                    model_info.model(input_ids)
                acts.append(current_activations[selected_layer])

            vectors[value] = torch.stack(acts).mean(dim=0)
    finally:
        handle.remove()

    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run theory-alignment geometry directly on raw positive activations for a saved CAA run."
    )
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--raw_subdir", default="activation_geometry_raw")
    parser.add_argument("--centered_subdir", default="activation_geometry_centered")
    args = parser.parse_args()

    model_dir = _resolve_model_dir_from_config(args.config_path)
    source_model_dir = _resolve_source_model_dir(model_dir)
    selected_layer = _resolve_selected_layer(model_dir)

    config = PipelineConfig.load(str(args.config_path))
    source_config = PipelineConfig.load(str(source_model_dir / "config.json"))
    raw_vectors = _collect_positive_activation_means(config, source_config.model_name, selected_layer, args.split)
    centered_vectors = mean_center_vectors(raw_vectors)

    raw_output_dir = model_dir / f"{args.raw_subdir}_{args.split}"
    centered_output_dir = model_dir / f"{args.centered_subdir}_{args.split}"

    common_metadata = {
        "representation_type": "mean_positive_last_token_activation",
        "source_model_dir": str(source_model_dir.relative_to(PROJECT_ROOT)),
        "selected_layer": selected_layer,
        "split": args.split,
        "normalization_note": "geometry uses cosine similarity after per-value vector normalization",
    }

    analyze_representation_geometry(
        raw_vectors,
        relations_path=config.relations_path,
        output_dir=raw_output_dir,
        seed=config.seed,
        title_prefix="Raw Activation ",
        metadata={**common_metadata, "transform": "none"},
    )

    analyze_representation_geometry(
        centered_vectors,
        relations_path=config.relations_path,
        output_dir=centered_output_dir,
        seed=config.seed,
        title_prefix="Centered Activation ",
        metadata={**common_metadata, "transform": "mean_center"},
    )

    print(f"Wrote {raw_output_dir}")
    print(f"Wrote {centered_output_dir}")


if __name__ == "__main__":
    main()
