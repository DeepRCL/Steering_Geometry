"""
Touche23-ValueEval pipeline.

Subclasses DatasetConstructionPipeline from value_bench/pipeline.py,
overriding only _build_messages() to use Touche-specific prompts and
value definitions built from value-categories.json + VALUEBENCH_DEFINITIONS.

All generation logic (_generate, _generate_batch, build_dataset_single,
build_dataset_batch, checkpointing/resumability) is inherited unchanged.
"""

import hashlib
import sys
import importlib.util
from pathlib import Path

# Make the project root importable
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

import config

# Import value_bench's pipeline explicitly by file path to avoid name collision
# (both files are named pipeline.py). We also temporarily place value_bench/ at
# the front of sys.path so that value_bench/pipeline.py resolves its own 'prompt'
# import correctly rather than picking up Touche23-ValueEval/prompt.py.
_VB_DIR = _ROOT / "dataset_construction" / "value_bench"
_VB_PIPELINE_PATH = _VB_DIR / "pipeline.py"

sys.path.insert(0, str(_VB_DIR))
_spec = importlib.util.spec_from_file_location("value_bench_pipeline", _VB_PIPELINE_PATH)
_vb_pipeline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vb_pipeline_module)
sys.path.remove(str(_VB_DIR))

# value_bench/pipeline.py caches its own 'prompt' and 'utils' modules under
# those generic names. Remove them so our local imports pick up the right files.
for _m in ("prompt", "utils"):
    sys.modules.pop(_m, None)

DatasetConstructionPipeline = _vb_pipeline_module.DatasetConstructionPipeline

from prompt import (
    TOUCHE_SYSTEM_PROMPT,
    TOUCHE_USER_PROMPT,
    TOUCHE_EXAMPLES,
    get_definition,
)

# ---------------------------------------------------------------------------
# Strategy rotation
#
# Strategies are assigned deterministically per (argument_id, value) pair via
# MD5 hashing, so the same row always gets the same strategy regardless of
# checkpoint state.  The distribution across ~29 k rows will be near-uniform
# across all six strategies (≈ 16–17 % each), enforcing diversity without
# ever exposing a Schwartz value name as the "target alternative."
# ---------------------------------------------------------------------------
_STRATEGY_ROTATION: list[str] = [
    "pragmatic: challenge the feasibility, cost, or practical implementation of the proposed policy",
    "empirical: challenge the factual or causal claims in the positive argument using evidence, data, or known real-world outcomes",
    "counter-example: cite a real or plausible case where the same policy was tried and led to opposite or harmful results",
    "side-effects: argue that the policy would produce serious unintended consequences in a domain unrelated to the target value",
    "institutional: argue that existing rules, professional norms, or institutional structures already handle this concern more effectively than the proposed policy",
    "contradict: challenge whether the concern invoked in the positive argument is valid, relevant, or accurately applied in this policy context — show the value-based premise is misplaced, overstated, or historically inaccurate",
]


def _get_strategy_hint(row) -> str:
    """
    Deterministically map a (argument_id, value) pair to one of the six
    strategies using MD5.  Using both fields ensures that multi-label rows
    sharing the same argument_id can receive different strategies.
    """
    key = f"{row.get('argument_id', '')}_{row.get('value', '')}".encode()
    idx = int(hashlib.md5(key).hexdigest(), 16) % len(_STRATEGY_ROTATION)
    return _STRATEGY_ROTATION[idx]


class TouchePipeline(DatasetConstructionPipeline):
    """
    Pipeline for the Touche23-ValueEval dataset.

    Inherits all model loading and generation logic from
    DatasetConstructionPipeline; only the prompt-building step is
    replaced to reflect the argumentative (policy-debate) style and
    the richer value definitions of Touche.
    """

    def _build_messages(self, row, direction: str = "positive_to_negative") -> list:
        """
        Build the chat messages for a single row.

        Parameters
        ----------
        row : pd.Series
            Must contain at least: 'question', 'value', 'positive_answer'.
        direction : str
            Currently only "positive_to_negative" is supported for Touche.
        """
        if direction != "positive_to_negative":
            raise ValueError(
                f"TouchePipeline only supports 'positive_to_negative', got {direction!r}"
            )

        definition_block = get_definition(row["value"])
        positive_word_count = len(str(row["positive_answer"]).split())
        strategy_hint = _get_strategy_hint(row)
        user = TOUCHE_USER_PROMPT.format(
            examples=TOUCHE_EXAMPLES,
            definition_block=definition_block,
            question=row["question"],
            positive_answer=row["positive_answer"],
            strategy_hint=strategy_hint,
            positive_word_count=positive_word_count,
            positive_word_count_plus_10=positive_word_count + 10,
        )

        return [
            {
                "role": "system",
                # Text-only HF chat pipelines are more stable when content is a
                # plain string instead of an OpenAI-style content block list.
                "content": TOUCHE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user,
            },
        ]


# ---------------------------------------------------------------------------
# Debug entry point — mirrors value_bench/pipeline.py __main__ block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    _DATA_DIR = _HERE.parent / "data"
    _INPUT    = _DATA_DIR / "touche_positive_only.csv"
    _DEBUG_IN  = _DATA_DIR / "debug_input.csv"
    _DEBUG_OUT = _DATA_DIR / "debug_output.csv"
    _TARGET_COL = "negative_answer"

    if not _DEBUG_IN.exists():
        print("Creating debug sample …")
        sample = pd.read_csv(_INPUT).head(config.DEBUG_ROWS).copy()
        sample[_TARGET_COL] = sample[_TARGET_COL].astype(object)
        sample.to_csv(_DEBUG_IN, index=False)
        print(f"Saved {config.DEBUG_ROWS}-row sample → {_DEBUG_IN}")
    else:
        print(f"Reusing existing debug sample → {_DEBUG_IN}")

    pipe = TouchePipeline()
    pipe.build_dataset_single(
        input_csv=_DEBUG_IN,
        output_csv=_DEBUG_OUT,
        direction="positive_to_negative",
        batch_size=config.BATCH_SIZE,
    )
