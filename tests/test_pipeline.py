import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from creative2d.pipeline.runner import Backends, JobContext, resize_wrap, run
from creative2d.post.seamless import seam_error
from creative2d.spec import AnimateOptions, JobSpec

from fakes import FakeImageBackend, FakeVideoBackend


@pytest.fixture(autouse=True)
def no_ml(monkeypatch):
    # Force the model-free border-key matting in tests.
    monkeypatch.setattr("creative2d.config.ml_available", lambda: False)


def _run(tmp_path, spec, video=False, **kw):
    events = []
    ctx = JobContext("t1", spec, tmp_path / "t1", lambda f, s, m: events.append((f, s, m)))
    backends = Backends(image=FakeImageBackend(), video=FakeVideoBackend() if video else None, **kw)
    manifest = run(ctx, backends)
    return manifest, events, tmp_path / "t1" / "assets"


def test_character_pixel_static(tmp_path):
    spec = JobSpec(prompt="red slime", name="slime", asset_type="character", style="pixel", frame_size=32, seed=1)
    m, events, assets = _run(tmp_path, spec)
    assert m["result"]["kind"] == "sprites"
    assert [f for f in m["result"]["frames"]] == ["slime"]
    atlas = json.loads((assets / "slime.json").read_text())
    assert "slime" in atlas["frames"]
    frame = Image.open(assets / "frames" / "slime.png")
    assert frame.size == (32, 32)
    a = np.asarray(frame.getchannel("A"))
    assert set(np.unique(a)) <= {0, 255}  # pixel style -> hard alpha
    assert a[0, 0] == 0 and a[16, 16] == 255  # background removed, subject kept
    assert (assets / "slime_sheet.png").exists()
    assert "this.load.atlas('slime'" in (assets / "phaser-loader.js").read_text()
    # Progress is monotonic and ends at 1.
    fracs = [e[0] for e in events]
    assert fracs == sorted(fracs) and fracs[-1] == 1.0
    with zipfile.ZipFile(tmp_path / "t1" / "slime.zip") as z:
        assert "manifest.json" in z.namelist()


def test_character_animated_with_directions(tmp_path):
    spec = JobSpec(
        prompt="blue blob", name="blob", style="painted", frame_size=48, directions=4, seed=3,
        animate=AnimateOptions(enabled=True, action="walk", frames=6, fps=8),
    )
    m, _, assets = _run(tmp_path, spec, video=True)
    r = m["result"]
    assert len(r["anims"]) == 4
    assert r["anims"][0]["key"] == "blob_down_walk"
    assert len(r["anims"][0]["frames"]) == 6
    anims = json.loads((assets / "blob_anims.json").read_text())
    assert anims["anims"][0]["frameRate"] == 8 and anims["anims"][0]["repeat"] == -1
    sheet_anims = json.loads((assets / "blob_sheet_anims.json").read_text())
    assert all(isinstance(f["frame"], int) for f in sheet_anims["anims"][0]["frames"])
    # Frames share one baseline (feet planted).
    bottoms = {Image.open(assets / "frames" / f"{n}.png").getchannel("A").getbbox()[3] for n in r["anims"][0]["frames"]}
    assert bottoms == {48}


def test_tiles_iso(tmp_path):
    spec = JobSpec(prompt="grass", name="grass", asset_type="tile", style="painted", view="iso", tile_size=64,
                   iso_depth=16, variants=3, seed=5)
    m, _, assets = _run(tmp_path, spec)
    r = m["result"]
    assert r["tile_width"] == 64 and r["tile_height"] == 32 + 16
    tsj = json.loads((assets / "grass_tiles.tsj").read_text())
    assert tsj["tilecount"] == 3 * r["chunk"] ** 2 and tsj["grid"]["orientation"] == "isometric"


def test_tiles_are_seamless(tmp_path):
    spec = JobSpec(prompt="dirt", name="dirt", asset_type="tile", style="painted", tile_size=64, tile_chunk=1, seed=2)
    _, _, assets = _run(tmp_path, spec)
    tile = Image.open(assets / "frames" / "dirt_0.png")
    assert tile.size == (64, 64)
    assert seam_error(tile) < 2.0


def test_small_pixel_tiles_become_seamless_block(tmp_path):
    spec = JobSpec(prompt="stone", name="st", asset_type="tile", style="pixel", tile_size=32, seed=2)
    m, _, assets = _run(tmp_path, spec)
    assert m["result"]["chunk"] == 4 and len(m["result"]["frames"]) == 16
    block = Image.new("RGBA", (128, 128))
    for r in range(4):
        for c in range(4):
            block.paste(Image.open(assets / "frames" / f"st_0_r{r}c{c}.png"), (c * 32, r * 32))
    assert seam_error(block) < 2.0


def test_background_parallax(tmp_path):
    spec = JobSpec(prompt="forest", name="forest", asset_type="background", style="pixel",
                   background_width=320, background_height=180, parallax_layers=3, seed=1)
    m, _, assets = _run(tmp_path, spec)
    keys = [e["key"] for e in m["result"]["entries"]]
    assert keys == ["forest_far", "forest_mid", "forest_near"]
    far = Image.open(assets / "forest_far.png")
    near = Image.open(assets / "forest_near.png")
    assert far.size == (320, 180)
    assert far.getchannel("A").getextrema() == (255, 255)
    assert near.getchannel("A").getextrema()[0] == 0  # near layer has transparency


def test_enhance_and_rank_used(tmp_path):
    calls = {}

    def enhance(request, asset_type, style, action):
        calls["enhance"] = (request, action)
        return {"subject": "a round crimson slime with glossy highlights", "motion": ""}

    def rank(request, images):
        calls["rank"] = len(images)
        return [1.0] * (len(images) - 1) + [9.0]

    spec = JobSpec(prompt="slime", name="s", candidates=3, enhance_prompt=True, text_model="x/y", seed=1)
    m, _, _ = _run(tmp_path, spec, enhance=enhance, rank=rank)
    assert calls == {"enhance": ("slime", None), "rank": 3}
    assert m["subject"].startswith("a round crimson")


def test_enhance_failure_does_not_fail_job(tmp_path):
    def enhance(*a):
        raise RuntimeError("boom")

    spec = JobSpec(prompt="slime", name="s", enhance_prompt=True, text_model="x/y", seed=1)
    m, _, _ = _run(tmp_path, spec, enhance=enhance)
    assert any("skipped" in line for line in m["log"])


def test_animation_needs_video_backend(tmp_path):
    spec = JobSpec(prompt="x", animate=AnimateOptions(enabled=True))
    with pytest.raises(ValueError):
        _run(tmp_path, spec)


def test_spec_validation():
    with pytest.raises(ValueError):
        JobSpec(prompt="x", asset_type="tile", animate=AnimateOptions(enabled=True))
    with pytest.raises(ValueError):
        JobSpec(prompt="x", enhance_prompt=True)
    with pytest.raises(ValueError):
        JobSpec(prompt="x", name="bad name!")


def test_resize_wrap_keeps_seams():
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8), "RGB")
    tiled = Image.new("RGB", (64, 64))
    for x in range(4):
        for y in range(4):
            tiled.paste(base, (x * 16, y * 16))
    small = resize_wrap(tiled.resize((256, 256), Image.NEAREST), (32, 32))
    assert seam_error(small) < 1.5


def test_pattern_candidates_trigger_reseed(tmp_path):
    from PIL import ImageDraw

    class PatternFirst(FakeImageBackend):
        def generate(self, req, progress=lambda f, m: None):
            res = super().generate(req, progress)
            if len(self.requests) == 1:
                img = Image.new("RGBA", (256, 256), "white")
                d = ImageDraw.Draw(img)
                for x in range(0, 256, 40):
                    for y in range(0, 256, 40):
                        d.ellipse((x, y, x + 25, y + 25), fill=(200, 30, 30, 255))
                res.images = [img]
            return res

    backend = PatternFirst()
    spec = JobSpec(prompt="slime", name="s", seed=1)
    ctx = JobContext("t", spec, tmp_path / "t")
    m = run(ctx, Backends(image=backend))
    assert len(backend.requests) == 2
    assert backend.requests[1].seed != 1
    assert any("attempt 2" in line for line in m["log"])
