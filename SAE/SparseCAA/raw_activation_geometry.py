from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAA.Geometry.representation_geometry import analyze_representation_geometry, mean_center_vectors
from SAE.SparseCAA.config import SparseCAAPipelineConfig, SCHWARTZ_CIRCUMPLEX_ORDER
from SAE.SparseCAA.data_loader import ContrastivePair, format_prompts, load_combined, split_dataset


def _collect_positive_activation_means(config: SparseCAAPipelineConfig) -> Dict[str, torch.Tensor]:
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if config.device == "auto" else config.device,
    )
    model.eval()
    torch.set_grad_enabled(False)

    name_lower = config.model_name.lower()
    is_instruct = "base" not in name_lower and "pt" not in name_lower

    df = load_combined(config)
    train_data, _ = split_dataset(df, config)

    current: Dict[str, torch.Tensor] = {}

    def _mlp_hook(module, inp, output):
        act = output[0] if isinstance(output, tuple) else output
        current["act"] = act[0, -1, :].detach().to(dtype=torch.float32, device="cpu")

    handle = model.model.layers[config.mlp_layer].mlp.register_forward_hook(_mlp_hook)
    activation_means: Dict[str, torch.Tensor] = {}

    try:
        for value in SCHWARTZ_CIRCUMPLEX_ORDER:
            pairs: List[ContrastivePair] = train_data.get(value, [])
            if not pairs:
                activation_means[value] = torch.zeros(config.d_in, dtype=torch.float32)
                continue

            pos_acts: List[torch.Tensor] = []
            for pair in tqdm(pairs, desc=f"Raw activations [{value}]", leave=False):
                pos_tokens, _ = format_prompts(pair, tokenizer, is_instruct)
                current.clear()
                with torch.no_grad():
                    model(torch.tensor([pos_tokens]).to(device))
                pos_acts.append(current["act"])

            activation_means[value] = torch.stack(pos_acts).mean(dim=0)
    finally:
        handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return activation_means


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run theory-alignment geometry directly on raw positive MLP activations for SparseCAA."
    )
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--raw_subdir", default="activation_geometry_raw")
    parser.add_argument("--centered_subdir", default="activation_geometry_centered")
    args = parser.parse_args()

    config = SparseCAAPipelineConfig.load(str(args.config_path))
    raw_vectors = _collect_positive_activation_means(config)
    centered_vectors = mean_center_vectors(raw_vectors)

    common_metadata = {
        "representation_type": "mean_positive_last_token_mlp_activation",
        "selected_layer": config.mlp_layer,
        "model_name": config.model_name,
        "split": "train",
        "normalization_note": "geometry uses cosine similarity after per-value vector normalization",
    }

    analyze_representation_geometry(
        raw_vectors,
        relations_path=config.relations_path,
        output_dir=Path(config.run_dir) / args.raw_subdir,
        seed=config.seed,
        title_prefix="Raw Activation ",
        metadata={**common_metadata, "transform": "none"},
    )

    analyze_representation_geometry(
        centered_vectors,
        relations_path=config.relations_path,
        output_dir=Path(config.run_dir) / args.centered_subdir,
        seed=config.seed,
        title_prefix="Centered Activation ",
        metadata={**common_metadata, "transform": "mean_center"},
    )

    print(f"Wrote {Path(config.run_dir) / args.raw_subdir}")
    print(f"Wrote {Path(config.run_dir) / args.centered_subdir}")


if __name__ == "__main__":
    main()
