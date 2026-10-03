"""Pixel-art post-processing: grid downscale, palette quantize, hard alpha."""

from __future__ import annotations

import numpy as np
from PIL import Image


def hard_alpha(img: Image.Image, threshold: int = 128) -> Image.Image:
    """Make every pixel fully opaque or fully transparent."""
    img = img.convert("RGBA")
    a = np.asarray(img.getchannel("A"))
    a = np.where(a >= threshold, 255, 0).astype(np.uint8)
    out = img.copy()
    out.putalpha(Image.fromarray(a, "L"))
    return out


def build_palette(img: Image.Image, colors: int) -> Image.Image:
    """Palette image ('P' mode) from the opaque pixels of img."""
    img = img.convert("RGBA")
    arr = np.asarray(img)
    opaque = arr[arr[..., 3] >= 128][:, :3]
    if len(opaque) == 0:
        opaque = np.zeros((1, 3), np.uint8)
    strip = Image.fromarray(opaque.reshape(1, -1, 3), "RGB")
    return strip.quantize(colors=max(2, min(colors, 256)), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def quantize(img: Image.Image, colors: int = 32, palette: Image.Image | None = None) -> Image.Image:
    """Reduce RGB to a small palette, keeping alpha. Pass `palette` (from
    build_palette) to lock colors across animation frames or tile variants."""
    img = img.convert("RGBA")
    alpha = img.getchannel("A")
    pal = palette or build_palette(img, colors)
    rgb = img.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE).convert("RGB")
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def downscale_pixel(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Downscale to the target pixel grid. BOX averages each block, which is
    stable for diffusion output where 'pixels' are soft, uneven blocks."""
    img = img.convert("RGBA")
    # Premultiply so transparent pixels do not bleed dark fringe colors.
    arr = np.asarray(img).astype(np.float32)
    a = arr[..., 3:4] / 255.0
    premul = np.concatenate([arr[..., :3] * a, arr[..., 3:4]], axis=-1).astype(np.uint8)
    small = np.asarray(Image.fromarray(premul, "RGBA").resize(size, Image.BOX)).astype(np.float32)
    sa = small[..., 3:4] / 255.0
    rgb = np.where(sa > 0, small[..., :3] / np.maximum(sa, 1e-6), 0)
    out = np.concatenate([np.clip(rgb, 0, 255), small[..., 3:4]], axis=-1).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def pixelize(
    img: Image.Image, size: tuple[int, int], colors: int = 32, palette: Image.Image | None = None, alpha: bool = True
) -> Image.Image:
    """Full pixel-art pass: downscale -> quantize -> hard alpha."""
    out = downscale_pixel(img, size) if img.size != size else img.convert("RGBA")
    out = quantize(out, colors, palette)
    return hard_alpha(out) if alpha else out
