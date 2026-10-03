"""Shared helpers for sheet/atlas packing."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class Frame:
    name: str
    image: Image.Image  # RGBA


def trim_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of non-transparent pixels, or None if fully transparent."""
    return img.getchannel("A").getbbox()


def paste_extruded(dst: Image.Image, src: Image.Image, x: int, y: int, extrude: int) -> None:
    """Paste src at (x, y) and repeat its edge pixels `extrude` times outward.

    Extrusion stops texture bleeding when the GPU samples across frame
    boundaries (scaling, sub-pixel camera positions).
    """
    dst.paste(src, (x, y))
    if extrude <= 0:
        return
    w, h = src.size
    top = src.crop((0, 0, w, 1))
    bottom = src.crop((0, h - 1, w, h))
    left = src.crop((0, 0, 1, h))
    right = src.crop((w - 1, 0, w, h))
    for i in range(1, extrude + 1):
        dst.paste(top, (x, y - i))
        dst.paste(bottom, (x, y + h - 1 + i))
        dst.paste(left, (x - i, y))
        dst.paste(right, (x + w - 1 + i, y))
    corners = [
        ((0, 0), (x - extrude, y - extrude)),
        ((w - 1, 0), (x + w, y - extrude)),
        ((0, h - 1), (x - extrude, y + h)),
        ((w - 1, h - 1), (x + w, y + h)),
    ]
    for (sx, sy), (dx, dy) in corners:
        dst.paste(Image.new("RGBA", (extrude, extrude), src.getpixel((sx, sy))), (dx, dy))
