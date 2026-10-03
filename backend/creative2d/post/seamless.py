"""Seamless tile checks and fixes."""

from __future__ import annotations

import numpy as np
from PIL import Image


def seam_error(img: Image.Image) -> float:
    """Mean absolute difference across the wrap edges, compared with the mean
    difference between neighboring pixels inside the image. ~1.0 means the
    wrap seam looks like any other pixel boundary; much higher means visible seam."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    wrap = (np.abs(a[:, 0] - a[:, -1]).mean() + np.abs(a[0] - a[-1]).mean()) / 2
    inner = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2
    return float(wrap / max(inner, 1e-3))


def blend_seams(img: Image.Image, band: float = 0.15) -> Image.Image:
    """Make a texture tileable by cross-fading it with a half-offset copy.

    The offset copy has its seams in the middle; a mask that favors the
    original near the center and the offset copy near the edges hides both.
    Cheap and model-free; used when generation-time tiling is not available
    (e.g. cloud backends).
    """
    img = img.convert("RGBA")
    w, h = img.size
    a = np.asarray(img).astype(np.float32)
    shifted = np.roll(a, (h // 2, w // 2), axis=(0, 1))
    ys = np.abs(np.linspace(-1, 1, h))[:, None]
    xs = np.abs(np.linspace(-1, 1, w))[None, :]
    edge = np.maximum(xs, ys)
    start = 1 - 2 * band
    m = np.clip((edge - start) / (1 - start), 0, 1)[..., None]  # 0 center .. 1 edge
    out = a * (1 - m) + shifted * m
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def make_tileable(img: Image.Image, threshold: float = 2.0) -> Image.Image:
    return blend_seams(img) if seam_error(img) > threshold else img
