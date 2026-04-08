import os
import sys
from pathlib import Path

_VS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _VS_ROOT.parent

sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID:       str = os.getenv("PERTURBATION_MODEL_ID", "Qwen/Qwen3.5-2B")
MAX_NEW_TOKENS: int = int(os.getenv("PERTURBATION_MAX_NEW_TOKENS", 2048))
DEVICE_MAP:     str = os.getenv("DEVICE_MAP", "auto")

# ── Data ─────────────────────────────────────────────────────────────────────
_DATA_DIR:   Path = _REPO_ROOT / "dataset_construction" / "data"
_RESULT_DIR: Path = _DATA_DIR / "perturb_result"
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV:              Path = _DATA_DIR / os.getenv("PERTURBATION_INPUT_CSV", "final_dataset_v3.csv")
PARAPHRASE_OUTPUT_CSV:  Path = _RESULT_DIR / os.getenv("PARAPHRASE_OUTPUT_CSV",  "final_dataset_paraphrased.csv")
ADVERSARIAL_OUTPUT_CSV: Path = _RESULT_DIR / os.getenv("ADVERSARIAL_OUTPUT_CSV", "final_dataset_adversarial.csv")

# ── Debug ─────────────────────────────────────────────────────────────────────
DEBUG_ROWS: int = int(os.getenv("PERTURBATION_DEBUG_ROWS", 10))
DEBUG_INPUT_CSV:              Path = _DATA_DIR  / "final_dataset_v3.csv"
DEBUG_PARAPHRASE_OUTPUT_CSV:  Path = _RESULT_DIR / "perturbation_debug_paraphrased.csv"
DEBUG_ADVERSARIAL_OUTPUT_CSV: Path = _RESULT_DIR / "perturbation_debug_adversarial.csv"
