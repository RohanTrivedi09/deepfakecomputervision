"""Shared path constants, resolved relative to the repo root."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "checkpoints"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _dir in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
