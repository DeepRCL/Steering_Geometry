import os
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

# Hugging Face
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# Google Gemini
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
GEMINI_API_KEYS: list[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]

# OpenRouter
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

# Model
MODEL_ID:       str = os.getenv("MODEL_ID", "Qwen/Qwen3.5-2B")
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", 512))
DEVICE_MAP:     str = os.getenv("DEVICE_MAP", "auto")

# Dataset 
_DATA_DIR = _ROOT / "dataset_construction" / "value_bench" / "data"
INPUT_CSV:  Path = _DATA_DIR / os.getenv("INPUT_CSV",  "dataset_positive_only.csv")
OUTPUT_CSV: Path = _DATA_DIR / os.getenv("OUTPUT_CSV", "dataset_with_negatives.csv")
BATCH_SIZE: int  = int(os.getenv("BATCH_SIZE", 10))
DEBUG_ROWS: int  = int(os.getenv("DEBUG_ROWS", 10))
