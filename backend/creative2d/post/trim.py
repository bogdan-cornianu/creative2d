"""Trim, fit and anchor sprites into fixed-size cells."""

from __future__ import annotations

from PIL import Image


def trim(img: Image.Image) -> Image.Image:
    box = img.getchannel("A").getbbox()
    return img.crop(box) if box else img


def fit_to_cell(
    img: Image.Image, cell: tuple[int, int], anchor: str = "bottom", margin: int = 0, resample=Image.LANCZOS
) -> Image.Image:
    """Scale img (keeping aspect) to fit inside cell minus margin and place it.

    anchor: "bottom" (bottom-center, for characters standing on ground) or "center".
    """
    cw, ch = cell
    avail_w, avail_h = max(1, cw - 2 * margin), max(1, ch - 2 * margin)
    scale = min(avail_w / img.width, avail_h / img.height)
    new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    scaled = img.resize(new, resample) if new != img.size else img
    out = Image.new("RGBA", cell, (0, 0, 0, 0))
    x = (cw - scaled.width) // 2
    y = ch - margin - scaled.height if anchor == "bottom" else (ch - scaled.height) // 2
    out.paste(scaled, (x, y), scaled)
    return out


def fit_frames_to_cell(
    frames: list[Image.Image], cell: tuple[int, int], anchor: str = "bottom", margin: int = 0, resample=Image.LANCZOS
) -> list[Image.Image]:
    """Fit an animation with one shared scale and union bbox so the subject
    does not jitter or change size between frames."""
    boxes = [f.getchannel("A").getbbox() for f in frames]
    valid = [b for b in boxes if b]
    if not valid:
        return [Image.new("RGBA", cell, (0, 0, 0, 0)) for _ in frames]
    x0 = min(b[0] for b in valid)
    y0 = min(b[1] for b in valid)
    x1 = max(b[2] for b in valid)
    y1 = max(b[3] for b in valid)
    return [fit_to_cell(f.crop((x0, y0, x1, y1)), cell, anchor, margin, resample) for f in frames]
