"""Packed texture atlas with Phaser JSON Hash / multiatlas export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .common import Frame, paste_extruded, trim_box
from .maxrects import pack


@dataclass
class AtlasPage:
    image: Image.Image
    frames: dict[str, dict]


def build_atlas(
    frames: list[Frame], max_size: int = 2048, padding: int = 2, extrude: int = 1, trim: bool = True
) -> list[AtlasPage]:
    """Trim, pack and draw frames. Returns one page per atlas image."""
    if not frames:
        raise ValueError("no frames to pack")
    names = [f.name for f in frames]
    if len(set(names)) != len(names):
        raise ValueError("frame names must be unique")

    cropped: list[tuple[Frame, Image.Image, tuple[int, int, int, int]]] = []
    for f in frames:
        img = f.image.convert("RGBA")
        box = trim_box(img) if trim else None
        if box is None:
            # Fully transparent (or trim disabled): keep full frame, or a 1x1 when empty.
            box = (0, 0, img.width, img.height) if not trim else (0, 0, 1, 1)
        cropped.append((f, img.crop(box), box))

    # Each packed cell reserves extrude on every side plus padding between cells.
    border = extrude + padding
    sizes = [(c.width + 2 * border, c.height + 2 * border) for _, c, _ in cropped]
    placements, bin_dims = pack(sizes, max_size)

    pages = [AtlasPage(Image.new("RGBA", dims, (0, 0, 0, 0)), {}) for dims in bin_dims]
    for (frame, crop, box), (bin_index, rect) in zip(cropped, placements):
        page = pages[bin_index]
        x, y = rect.x + border, rect.y + border
        paste_extruded(page.image, crop, x, y, extrude)
        w, h = frame.image.size
        page.frames[frame.name] = {
            "frame": {"x": x, "y": y, "w": crop.width, "h": crop.height},
            "rotated": False,
            "trimmed": (crop.width, crop.height) != (w, h),
            "spriteSourceSize": {"x": box[0], "y": box[1], "w": crop.width, "h": crop.height},
            "sourceSize": {"w": w, "h": h},
        }
    return pages


def _meta(image: str, size: tuple[int, int]) -> dict:
    return {
        "app": "creative2d",
        "version": "1.0",
        "image": image,
        "format": "RGBA8888",
        "size": {"w": size[0], "h": size[1]},
        "scale": "1",
    }


def write_atlas(pages: list[AtlasPage], out_dir: Path, key: str) -> dict:
    """Write atlas PNG(s) + JSON. Single page -> JSON Hash (load.atlas),
    several pages -> multiatlas JSON (load.multiatlas). Returns loader info."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(pages) == 1:
        png = f"{key}.png"
        pages[0].image.save(out_dir / png, optimize=True)
        data = {"frames": pages[0].frames, "meta": _meta(png, pages[0].image.size)}
        (out_dir / f"{key}.json").write_text(json.dumps(data, indent=1))
        return {"loader": "atlas", "key": key, "texture": png, "json": f"{key}.json"}

    textures = []
    for i, page in enumerate(pages):
        png = f"{key}-{i}.png"
        page.image.save(out_dir / png, optimize=True)
        textures.append(
            {
                "image": png,
                "format": "RGBA8888",
                "size": {"w": page.image.width, "h": page.image.height},
                "scale": 1,
                "frames": [{"filename": name, **data} for name, data in page.frames.items()],
            }
        )
    data = {"textures": textures, "meta": {"app": "creative2d", "version": "1.0"}}
    (out_dir / f"{key}.json").write_text(json.dumps(data, indent=1))
    return {"loader": "multiatlas", "key": key, "json": f"{key}.json", "textures": [t["image"] for t in textures]}
