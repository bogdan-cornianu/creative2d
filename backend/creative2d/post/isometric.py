"""Project square top-down tiles to isometric (2:1 dimetric) tiles."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def _warp(src: Image.Image, size: tuple[int, int], coeffs: tuple, polygon: list, resample, supersample: int) -> Image.Image:
    """Affine-warp src into size, masked by polygon (output coordinates)."""
    ss = supersample
    w, h = size[0] * ss, size[1] * ss
    a, b, c, d, e, f = coeffs
    # Coefficients map output pixel -> source pixel; rescale for supersampling.
    scaled = (a / ss, b / ss, c, d / ss, e / ss, f)
    out = src.convert("RGBA").transform((w, h), Image.AFFINE, scaled, resample=resample)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon([(x * ss, y * ss) for x, y in polygon], fill=255)
    alpha = np.minimum(np.asarray(out.getchannel("A")), np.asarray(mask))
    out.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    return out.resize(size, Image.BOX) if ss > 1 else out


def _shade(img: Image.Image, factor: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr[..., :3] *= factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


def to_iso(tile: Image.Image, width: int, depth: int = 0, pixel: bool = False) -> Image.Image:
    """Map a square top-down tile to an isometric diamond `width` x `width/2`.

    depth > 0 adds shaded left/right side faces (an iso block), making the
    image `width` x (`width/2` + depth). Pixel art uses nearest sampling
    without supersampling to keep hard edges.
    """
    S = tile.width
    if tile.height != S:
        raise ValueError("iso projection needs a square tile")
    W, H, D = width, width // 2, depth
    resample = Image.NEAREST if pixel else Image.BICUBIC
    ss = 1 if pixel else 4
    total = (W, H + D)

    top_coeffs = (S / W, S / H, -S / 2, -S / W, S / H, S / 2)
    top = _warp(tile, total, top_coeffs, [(W / 2, 0), (W, H / 2), (W / 2, H), (0, H / 2)], resample, ss)
    if D <= 0:
        return top

    left_coeffs = (2 * S / W, 0, 0, -H * S / (W * D), S / D, -H * S / (2 * D))
    left = _warp(tile, total, left_coeffs, [(0, H / 2), (W / 2, H), (W / 2, H + D), (0, H / 2 + D)], resample, ss)
    right_coeffs = (2 * S / W, 0, -S, H * S / (W * D), S / D, -1.5 * H * S / D)
    right = _warp(tile, total, right_coeffs, [(W / 2, H), (W, H / 2), (W, H / 2 + D), (W / 2, H + D)], resample, ss)

    out = Image.new("RGBA", total, (0, 0, 0, 0))
    out.alpha_composite(_shade(left, 0.78))
    out.alpha_composite(_shade(right, 0.6))
    out.alpha_composite(top)
    return out
