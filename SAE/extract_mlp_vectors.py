"""
Extract contrastive activation (CAA) vectors from the MLP sub-module of a
specific transformer layer.

WHY A SEPARATE EXTRACTION?
────────────────────────────────────────────────────────────────────────────
The released SAE was trained on the *MLP output* of layer 16 (the tensor
produced by ``model.model.layers[16].mlp`` before it is added to the residual
stream).  The vectors stored in ``CAA/Geometry/outputs/.../vectors/`` capture
the *full residual stream* at each layer (i.e. the output of the complete
transformer block).  These two spaces are correlated but not identical.

For the SAE to correctly decompose steering vectors into interpretable features
we must feed it activations from the same space it was trained on, hence this
dedicated extraction step.

OUTPUT
───────
For each Schwartz value, saves:
  <output_dir>/<model_name_safe>/mlp_vectors/<value_safe>/layer_<N>.pt
  (a float32 tensor of shape (d_in,) = mean_pos_mlp − mean_neg_mlp)

These are structurally identical to the residual-stream vectors but live in
the MLP activation space.
"""
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import SAEConfig, SCHWARTZ_CIRCUMPLEX_ORDER, safe_name


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers (kept self-contained so no CAA/Geometry imports are needed)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class _Pair:
    sample_id: str
    value: str
    question: str
    positive_answer: str
    negative_answer: str
    pos_is_a: bool = True


def _load_train_pairs(
    dataset_path: str,
    eval_split: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[_Pair]]:
    """Load CSV and return training pairs grouped by value."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    grouped: Dict[str, List[_Pair]] = {}
    with open(dataset_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row["value"]
            if val not in SCHWARTZ_CIRCUMPLEX_ORDER:
                continue
            grouped.setdefault(val, []).append(
                _Pair(
                    sample_id=row["id"],
                    value=val,
                    question=row["question"],
                    positive_answer=row["positive_answer"],
                    negative_answer=row["negative_answer"],
                )
            )

    train_data: Dict[str, List[_Pair]] = {}
    for val in SCHWARTZ_CIRCUMPLEX_ORDER:
        pairs = grouped.get(val, [])
        rng.shuffle(pairs)
        n_eval = int(len(pairs) * eval_split)
        train = pairs[n_eval:]
        # Randomise which option (A/B) corresponds to the positive answer
        for p in train:
            p.pos_is_a = bool(np_rng.integers(0, 2))
        train_data[val] = train

    return train_data


def _build_prompts(
    pair: _Pair,
    tokenizer,
    is_instruct: bool,
) -> tuple:
    """Return (pos_tokens, neg_tokens) exactly as done in CAA/Geometry/data_loader.py."""
    if pair.pos_is_a:
        a_text, b_text = pair.positive_answer, pair.negative_answer
        pos_letter, neg_letter = "A", "B"
    else:
        a_text, b_text = pair.negative_answer, pair.positive_answer
        pos_letter, neg_letter = "B", "A"

    body = f"Question: {pair.question}\n(A) {a_text}\n(B) {b_text}"

    if is_instruct:
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        base = body + "\nResponse:"

    pos_prompt = base + f" ({pos_letter}"
    neg_prompt = base + f" ({neg_letter}"

    pos_tokens = tokenizer.encode(pos_prompt, add_special_tokens=True)
    neg_tokens = tokenizer.encode(neg_prompt, add_special_tokens=True)
    return pos_tokens, neg_tokens


# ──────────────────────────────────────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_mlp_vectors(config: SAEConfig) -> Dict[str, torch.Tensor]:
    """
    For each Schwartz value, extract mean(pos_mlp) − mean(neg_mlp) from the
    MLP at ``config.mlp_layer``.

    Vectors are cached to disk; already-computed values are loaded from cache
    so the function is safe to call multiple times (or after a partial run).

    Returns:
        Dict mapping value name → float32 tensor of shape (d_in,).
    """
    vec_dir = config.subdir("mlp_vectors")

    # Fast path: all vectors already on disk
    def _cached_path(val: str) -> str:
        return os.path.join(vec_dir, safe_name(val), f"layer_{config.mlp_layer}.pt")

    missing = [v for v in SCHWARTZ_CIRCUMPLEX_ORDER if not os.path.exists(_cached_path(v))]

    if not missing:
        print("All MLP vectors already cached – loading from disk.")
        return {v: torch.load(_cached_path(v), map_location="cpu") for v in SCHWARTZ_CIRCUMPLEX_ORDER}

    print(f"{len(missing)}/{len(SCHWARTZ_CIRCUMPLEX_ORDER)} values need extraction.")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {config.model_name}")
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

    # Detect whether this is an instruct model
    name_lower = config.model_name.lower()
    is_instruct = "base" not in name_lower and "pt" not in name_lower
    print(f"is_instruct={is_instruct}, target layer={config.mlp_layer}, device={device}")

    # ── Hook into MLP ─────────────────────────────────────────────────────────
    mlp_module = model.model.layers[config.mlp_layer].mlp
    print(f"MLP module: {type(mlp_module).__name__}")

    _current_mlp_act: Dict[str, Optional[torch.Tensor]] = {"val": None}

    def _mlp_hook(module, inp, output):
        # MLP output is a plain tensor for Qwen3; handle tuple just in case.
        act = output[0] if isinstance(output, tuple) else output
        # Last token, first (only) batch item → shape (d_in,)
        _current_mlp_act["val"] = act[0, -1, :].detach().to(device="cpu", dtype=torch.float32)

    handle = mlp_module.register_forward_hook(_mlp_hook)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_data = _load_train_pairs(config.dataset_path, config.eval_split, config.seed)

    # ── Extraction loop ───────────────────────────────────────────────────────
    vectors: Dict[str, torch.Tensor] = {}

    try:
        for val in SCHWARTZ_CIRCUMPLEX_ORDER:
            cache_path = _cached_path(val)
            if os.path.exists(cache_path):
                vectors[val] = torch.load(cache_path, map_location="cpu")
                print(f"  [cache] {val}")
                continue

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            pairs = train_data.get(val, [])

            if not pairs:
                print(f"  [WARNING] No training pairs for '{val}' – saving zero vector.")
                vec = torch.zeros(config.d_in)
                torch.save(vec, cache_path)
                vectors[val] = vec
                continue

            pos_acts: List[torch.Tensor] = []
            neg_acts: List[torch.Tensor] = []

            for pair in tqdm(pairs, desc=f"  Extracting [{val}]", leave=False):
                pos_tokens, neg_tokens = _build_prompts(pair, tokenizer, is_instruct)

                _current_mlp_act["val"] = None
                model(torch.tensor([pos_tokens]).to(device))
                pos_acts.append(_current_mlp_act["val"].clone())  # type: ignore[union-attr]

                _current_mlp_act["val"] = None
                model(torch.tensor([neg_tokens]).to(device))
                neg_acts.append(_current_mlp_act["val"].clone())  # type: ignore[union-attr]

            pos_mean = torch.stack(pos_acts).mean(dim=0)
            neg_mean = torch.stack(neg_acts).mean(dim=0)
            vec = pos_mean - neg_mean

            torch.save(vec, cache_path)
            vectors[val] = vec
            print(f"  Saved [{val}] | pairs={len(pairs)} | norm={vec.norm():.4f}")

    finally:
        handle.remove()

    return vectors
