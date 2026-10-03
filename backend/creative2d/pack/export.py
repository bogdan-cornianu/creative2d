"""Phaser animation JSON, manifest, loader snippet and zip export."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


def phaser_anims(anim_key: str, texture_key: str, frame_names: list[str | int], fps: int, loop: bool) -> dict:
    """Animation data for `this.anims.fromJSON(data)`.

    frame_names are atlas frame names (str) or spritesheet frame indices (int).
    """
    return {
        "anims": [
            {
                "key": anim_key,
                "type": "frame",
                "frames": [{"key": texture_key, "frame": f} for f in frame_names],
                "frameRate": fps,
                "repeat": -1 if loop else 0,
            }
        ],
        "globalTimeScale": 1,
    }


def loader_snippet(entries: list[dict], anims_file: str | None) -> str:
    """Phaser 3 preload/create code matching the exported files."""
    lines = ["// Phaser 3 — copy the files next to your game and adjust the base path.", "function preload() {"]
    lines.append("  this.load.setPath('assets/');")
    for e in entries:
        if e["loader"] == "atlas":
            lines.append(f"  this.load.atlas('{e['key']}', '{e['texture']}', '{e['json']}');")
        elif e["loader"] == "multiatlas":
            lines.append(f"  this.load.multiatlas('{e['key']}', '{e['json']}');")
        elif e["loader"] == "spritesheet":
            fc = e["frameConfig"]
            lines.append(
                f"  this.load.spritesheet('{e['key']}', '{e['texture']}', "
                f"{{ frameWidth: {fc['frameWidth']}, frameHeight: {fc['frameHeight']}, "
                f"margin: {fc['margin']}, spacing: {fc['spacing']} }});"
            )
        elif e["loader"] == "tileset":
            lines.append(f"  this.load.image('{e['key']}', '{e['texture']}');")
        elif e["loader"] == "image":
            lines.append(f"  this.load.image('{e['key']}', '{e['texture']}');")
    if anims_file:
        lines.append(f"  this.load.json('{Path(anims_file).stem}', '{anims_file}');")
    lines.append("}")
    lines.append("")
    lines.append("function create() {")
    if anims_file:
        lines.append(f"  this.anims.fromJSON(this.cache.json.get('{Path(anims_file).stem}'));")
    for e in entries:
        if e["loader"] == "tileset":
            lines.append(
                f"  // map.addTilesetImage('{e['key']}', '{e['key']}', {e['tileWidth']}, {e['tileHeight']}, "
                f"{e['margin']}, {e['spacing']});"
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


def zip_dir(src: Path, dest: Path) -> Path:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file() and p != dest:
                z.write(p, p.relative_to(src))
    return dest
