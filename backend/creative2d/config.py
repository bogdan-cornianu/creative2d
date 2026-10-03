"""Paths, device detection and model profiles."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("CREATIVE2D_OUTPUT_DIR", ROOT / "outputs"))
DATA_DIR = Path(os.environ.get("CREATIVE2D_DATA_DIR", ROOT / "data"))
MODELS_FILE = Path(os.environ.get("CREATIVE2D_MODELS_FILE", ROOT / "models.yaml"))
CONFIG_DIR = Path(os.environ.get("CREATIVE2D_CONFIG_DIR", Path.home() / ".config" / "creative2d"))
WEB_DIST = ROOT / "web" / "dist"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def detect_device() -> str:
    """Pick the best available torch device: mps > cuda > cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ml_available() -> bool:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def load_models_config() -> dict[str, Any]:
    if not MODELS_FILE.exists():
        return {}
    with MODELS_FILE.open() as f:
        return yaml.safe_load(f) or {}
