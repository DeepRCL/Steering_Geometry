"""
Touche23-ValueEval pipeline.

Subclasses DatasetConstructionPipeline from value_bench/pipeline.py,
overriding only _build_messages() to use Touche-specific prompts and
value definitions built from value-categories.json + VALUEBENCH_DEFINITIONS.

All generation logic (_generate, _generate_batch, build_dataset_single,
build_dataset_batch, checkpointing/resumability) is inherited unchanged.
"""

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
        user = TOUCHE_USER_PROMPT.format(
            examples=TOUCHE_EXAMPLES,
            definition_block=definition_block,
            question=row["question"],
            positive_answer=row["positive_answer"],
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
