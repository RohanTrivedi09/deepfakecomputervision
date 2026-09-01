"""Shared path constants, resolved relative to the repo root.

Training happens in notebooks/deepfake_training.ipynb (self-contained, runs on
Colab) — this module only covers what the local inference/demo side needs:
where to find the checkpoint + metadata that notebook produces.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models" / "checkpoints"
