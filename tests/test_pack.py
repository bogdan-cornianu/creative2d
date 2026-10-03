import json
import random

import pytest
from PIL import Image

from creative2d.pack.atlas import build_atlas, write_atlas
from creative2d.pack.common import Frame
from creative2d.pack.export import loader_snippet, phaser_anims
from creative2d.pack.maxrects import pack
from creative2d.pack.spritesheet import build_spritesheet
from creative2d.pack.tilesheet import build_tilesheet, write_tilesheet


def _sprite(w, h, box=None, color=(255, 0, 0, 255)):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    box = box or (0, 0, w, h)
    img.paste(Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), color), box[:2])
    return img


def test_maxrects_no_overlap_and_in_bounds():
    rng = random.Random(1)
    sizes = [(rng.randint(4, 90), rng.randint(4, 90)) for _ in range(120)]
    placements, dims = pack(sizes, 512)
    by_bin: dict[int, list] = {}
    for (w, h), (b, r) in zip(sizes, placements):
        assert (r.w, r.h) == (w, h)
        assert r.right <= dims[b][0] and r.bottom <= dims[b][1]
        by_bin.setdefault(b, []).append(r)
    for rects in by_bin.values():
        for i, a in enumerate(rects):
            for c in rects[i + 1 :]:
                assert not a.intersects(c)


def test_maxrects_rejects_oversize():
    with pytest.raises(ValueError):
        pack([(600, 10)], 512)


def test_maxrects_power_of_two_bins():
    _, dims = pack([(30, 30)] * 5, 1024)
    assert len(dims) == 1 and all((d & (d - 1)) == 0 for d in dims[0])


def test_atlas_trims_and_records_source_size(tmp_path):
    frames = [Frame("a", _sprite(64, 64, (10, 20, 30, 60))), Frame("b", _sprite(32, 32))]
    pages = build_atlas(frames, padding=2, extrude=1)
    assert len(pages) == 1
    fa = pages[0].frames["a"]
    assert fa["trimmed"] is True
    assert fa["spriteSourceSize"] == {"x": 10, "y": 20, "w": 20, "h": 40}
    assert fa["sourceSize"] == {"w": 64, "h": 64}
    assert pages[0].frames["b"]["trimmed"] is False
    # Pixel data lands where the JSON says.
    f = fa["frame"]
    assert pages[0].image.getpixel((f["x"], f["y"])) == (255, 0, 0, 255)
    # Extrusion copies the edge one pixel outward.
    assert pages[0].image.getpixel((f["x"] - 1, f["y"])) == (255, 0, 0, 255)

    info = write_atlas(pages, tmp_path, "hero")
    data = json.loads((tmp_path / "hero.json").read_text())
    assert info["loader"] == "atlas"
    assert data["meta"]["image"] == "hero.png"
    assert set(data["frames"]) == {"a", "b"}


def test_atlas_multi_page(tmp_path):
    frames = [Frame(f"f{i}", _sprite(200, 200)) for i in range(6)]
    pages = build_atlas(frames, max_size=512, padding=0, extrude=0)
    assert len(pages) > 1
    info = write_atlas(pages, tmp_path, "big")
    data = json.loads((tmp_path / "big.json").read_text())
    assert info["loader"] == "multiatlas"
    names = [fr["filename"] for t in data["textures"] for fr in t["frames"]]
    assert sorted(names) == sorted(f.name for f in frames)


def test_atlas_duplicate_names():
    with pytest.raises(ValueError):
        build_atlas([Frame("x", _sprite(4, 4)), Frame("x", _sprite(4, 4))])


def test_spritesheet_grid_and_bottom_anchor():
    frames = [Frame(str(i), _sprite(16, 16)) for i in range(5)]
    frames.append(Frame("small", _sprite(8, 8)))
    sheet, cfg = build_spritesheet(frames, cell=(16, 16))
    assert cfg["columns"] == 3 and cfg["rows"] == 2
    assert sheet.size == (48, 32)
    # Small frame (index 5 -> col 2, row 1) sits bottom-center in its cell.
    assert sheet.getpixel((32 + 4, 16 + 15)) == (255, 0, 0, 255)
    assert sheet.getpixel((32 + 4, 16 + 7))[3] == 0


def test_tilesheet_margin_spacing(tmp_path):
    tiles = [Frame(f"t{i}", _sprite(16, 16, color=(0, i * 40, 0, 255))) for i in range(4)]
    sheet, cfg = build_tilesheet(tiles, 16, 16, extrude=2)
    assert cfg["margin"] == 2 and cfg["spacing"] == 4 and cfg["columns"] == 2
    assert sheet.size == (2 * 2 + 2 * 16 + 4, 2 * 2 + 2 * 16 + 4)
    # Tile 1 starts at margin + tile + spacing.
    x = 2 + 16 + 4
    assert sheet.getpixel((x, 2)) == (0, 40, 0, 255)
    assert sheet.getpixel((x - 1, 2)) == (0, 40, 0, 255)  # extruded
    info = write_tilesheet(sheet, cfg, tmp_path, "grass", orientation="isometric")
    tsj = json.loads((tmp_path / "grass.tsj").read_text())
    assert tsj["tilecount"] == 4 and tsj["spacing"] == 4 and tsj["grid"]["orientation"] == "isometric"
    assert info["loader"] == "tileset"


def test_tilesheet_rejects_wrong_size():
    with pytest.raises(ValueError):
        build_tilesheet([Frame("t", _sprite(8, 16))], 16, 16)


def test_phaser_anims_and_snippet():
    anims = phaser_anims("hero-walk", "hero", ["walk_0", "walk_1"], 12, True)
    a = anims["anims"][0]
    assert a["repeat"] == -1 and a["frames"][1] == {"key": "hero", "frame": "walk_1"}
    code = loader_snippet([{"loader": "atlas", "key": "hero", "texture": "hero.png", "json": "hero.json"}], "anims.json")
    assert "this.load.atlas('hero', 'hero.png', 'hero.json');" in code
    assert "this.anims.fromJSON" in code
