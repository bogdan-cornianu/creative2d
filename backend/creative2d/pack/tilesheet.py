"""Tile sheet (tileset image) with extrusion, plus Tiled tileset export."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

from .common import Frame, paste_extruded


def build_tilesheet(
    tiles: list[Frame], tile_w: int, tile_h: int, columns: int | None = None, extrude: int = 1
) -> tuple[Image.Image, dict]:
    """Grid of equally sized tiles.

    Layout follows the tile-extruder convention: margin = extrude,
    spacing = 2 * extrude, so Phaser `addTilesetImage(..., margin, spacing)`
    and Tiled read it directly.
    """
    if not tiles:
        raise ValueError("no tiles")
    for t in tiles:
        if t.image.size != (tile_w, tile_h):
            raise ValueError(f"tile {t.name} is {t.image.size}, expected {(tile_w, tile_h)}")
    n = len(tiles)
    cols = columns or min(n, max(1, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)
    margin, spacing = extrude, 2 * extrude
    width = 2 * margin + cols * tile_w + (cols - 1) * spacing
    height = 2 * margin + rows * tile_h + (rows - 1) * spacing
    sheet = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for i, t in enumerate(tiles):
        c, r = i % cols, i // cols
        x = margin + c * (tile_w + spacing)
        y = margin + r * (tile_h + spacing)
        paste_extruded(sheet, t.image.convert("RGBA"), x, y, extrude)
    config = {
        "tileWidth": tile_w,
        "tileHeight": tile_h,
        "margin": margin,
        "spacing": spacing,
        "columns": cols,
        "tileCount": n,
        "names": [t.name for t in tiles],
    }
    return sheet, config


def write_tilesheet(
    sheet: Image.Image, config: dict, out_dir: Path, key: str, orientation: str = "orthogonal"
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = f"{key}.png"
    sheet.save(out_dir / png, optimize=True)
    tsj = {
        "type": "tileset",
        "version": "1.10",
        "name": key,
        "image": png,
        "imagewidth": sheet.width,
        "imageheight": sheet.height,
        "tilewidth": config["tileWidth"],
        "tileheight": config["tileHeight"],
        "margin": config["margin"],
        "spacing": config["spacing"],
        "columns": config["columns"],
        "tilecount": config["tileCount"],
        "tiles": [
            {"id": i, "properties": [{"name": "name", "type": "string", "value": name}]}
            for i, name in enumerate(config["names"])
        ],
    }
    if orientation == "isometric":
        # Iso tiles: Tiled grid is 2:1 diamond, image may be taller (blocks).
        tsj["grid"] = {"orientation": "isometric", "width": config["tileWidth"], "height": config["tileWidth"] // 2}
    (out_dir / f"{key}.tsj").write_text(json.dumps(tsj, indent=1))
    return {
        "loader": "tileset",
        "key": key,
        "texture": png,
        "tiled": f"{key}.tsj",
        "tileWidth": config["tileWidth"],
        "tileHeight": config["tileHeight"],
        "margin": config["margin"],
        "spacing": config["spacing"],
        "orientation": orientation,
    }
