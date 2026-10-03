"""Uniform-grid spritesheet for Phaser `load.spritesheet`."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from .common import Frame


def build_spritesheet(
    frames: list[Frame], cell: tuple[int, int] | None = None, columns: int | None = None, margin: int = 0, spacing: int = 0
) -> tuple[Image.Image, dict]:
    """Place frames left-to-right, top-to-bottom in equal cells.

    Frames smaller than the cell are centered horizontally and aligned to the
    cell bottom (feet stay on the same baseline across frames).
    """
    if not frames:
        raise ValueError("no frames")
    cw, ch = cell or (max(f.image.width for f in frames), max(f.image.height for f in frames))
    n = len(frames)
    cols = columns or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    width = 2 * margin + cols * cw + (cols - 1) * spacing
    height = 2 * margin + rows * ch + (rows - 1) * spacing
    sheet = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        img = f.image.convert("RGBA")
        if img.width > cw or img.height > ch:
            raise ValueError(f"frame {f.name} {img.size} larger than cell {(cw, ch)}")
        c, r = i % cols, i // cols
        x = margin + c * (cw + spacing) + (cw - img.width) // 2
        y = margin + r * (ch + spacing) + (ch - img.height)
        sheet.paste(img, (x, y))
    config = {
        "frameWidth": cw,
        "frameHeight": ch,
        "margin": margin,
        "spacing": spacing,
        "startFrame": 0,
        "endFrame": n - 1,
        "columns": cols,
        "rows": rows,
    }
    return sheet, config


def write_spritesheet(sheet: Image.Image, config: dict, out_dir: Path, key: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = f"{key}.png"
    sheet.save(out_dir / png, optimize=True)
    return {"loader": "spritesheet", "key": key, "texture": png, "frameConfig": config}
