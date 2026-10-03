"""Pipeline orchestration: prompt -> generate -> post-process -> animate -> pack."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from .. import config
from ..gen.base import Cancelled, ImageBackend, ImageRequest, VideoBackend, VideoRequest
from ..pack.atlas import build_atlas, write_atlas
from ..pack.common import Frame
from ..pack.export import loader_snippet, phaser_anims, write_manifest, zip_dir
from ..pack.spritesheet import build_spritesheet, write_spritesheet
from ..pack.tilesheet import build_tilesheet, write_tilesheet
from ..post import matting
from ..post.isometric import to_iso
from ..post.pixelate import build_palette, hard_alpha, pixelize, quantize
from ..post.seamless import make_tileable, seam_error
from ..post.trim import fit_frames_to_cell, fit_to_cell, trim
from ..spec import JobSpec
from . import prompts

# Emit(overall_fraction 0..1, stage, message)
Emit = Callable[[float, str, str], None]


@dataclass
class Backends:
    image: ImageBackend
    video: VideoBackend | None = None
    enhance: Callable[[str, str, str, str | None], dict] | None = None  # (request, asset_type, style, action)
    rank: Callable[[str, list[Image.Image]], list[float]] | None = None  # (request, images) -> scores


@dataclass
class JobContext:
    job_id: str
    spec: JobSpec
    out_dir: Path
    emit: Emit = lambda f, s, m: None
    is_cancelled: Callable[[], bool] = lambda: False
    cost: float = 0.0
    log: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.log.append(msg)


class Stages:
    """Maps per-stage progress into one overall fraction."""

    def __init__(self, ctx: JobContext, weights: dict[str, float]) -> None:
        total = sum(weights.values())
        self.ctx = ctx
        self.spans: dict[str, tuple[float, float]] = {}
        self.last = 0.0  # progress never moves backwards (e.g. on reseed retries)
        acc = 0.0
        for name, w in weights.items():
            self.spans[name] = (acc / total, (acc + w) / total)
            acc += w

    def progress(self, stage: str):
        start, end = self.spans[stage]

        def cb(frac: float, msg: str) -> None:
            if self.ctx.is_cancelled():
                raise Cancelled()
            self.last = max(self.last, start + (end - start) * max(0.0, min(1.0, frac)))
            self.ctx.emit(self.last, stage, msg)

        return cb


def _style_cfg(spec: JobSpec) -> dict:
    return config.load_models_config().get("styles", {}).get(spec.style, {})


def resize_wrap(img: Image.Image, size: tuple[int, int], resample=Image.LANCZOS) -> Image.Image:
    """Resize a tileable texture without breaking the wrap: resample a 3x3
    tiled copy and crop the middle."""
    w, h = img.size
    big = Image.new(img.mode, (w * 3, h * 3))
    for dx in range(3):
        for dy in range(3):
            big.paste(img, (dx * w, dy * h))
    big = big.resize((size[0] * 3, size[1] * 3), resample)
    return big.crop((size[0], size[1], size[0] * 2, size[1] * 2))


def auto_tile_chunk(tile_size: int, pixel: bool) -> int:
    """How many tiles per side one generated texture is split into.

    Diffusion textures carry roughly a 128 px pixel-art grid (256 px of
    useful detail for painted styles). Squeezing that into a tiny tile turns
    it into noise, so small tiles get a seamless block of several tiles.
    """
    grid = 128 if pixel else 256
    return max(1, min(4, round(grid / tile_size)))


def _pick_best(ctx: JobContext, backends: Backends, request: str, images: list[Image.Image]) -> Image.Image:
    if len(images) == 1 or backends.rank is None:
        return images[0]
    try:
        scores = backends.rank(request, images)
        best = max(range(len(images)), key=lambda i: scores[i])
        ctx.note(f"QA scores {scores} -> candidate {best}")
        return images[best]
    except Exception as e:  # QA is best-effort; never fail the job over it
        ctx.note(f"QA ranking skipped: {e}")
        return images[0]


MIN_SUBJECT_SHARE = 0.6
MAX_RESEEDS = 3


def _generate_single_subject(ctx: JobContext, backends: Backends, req: ImageRequest, subject: str, cb,
                             label: str) -> Image.Image:
    """Generate candidates; drop ones that show many subjects (sprite sheets,
    patterns) or cropped ones, and reseed when every candidate fails."""
    best: tuple[float, Image.Image] | None = None
    for attempt in range(MAX_RESEEDS + 1):
        if attempt:
            req.seed += 7919
            cb(0.0, f"retrying with new seed (attempt {attempt + 1})")
        result = backends.image.generate(req, cb)
        ctx.cost += result.cost
        scored = [(matting.subject_share(im), im) for im in result.images]
        good = [im for sc, im in scored if sc >= MIN_SUBJECT_SHARE]
        top = max(scored, key=lambda t: t[0])
        if best is None or top[0] > best[0]:
            best = top
        if good:
            ctx.note(f"{label}: subject check passed on attempt {attempt + 1} (seed {req.seed})")
            return _pick_best(ctx, backends, subject, good)
    ctx.note(f"{label}: no candidate passed the single-subject check; kept best ({best[0]:.2f})")
    return best[1]


def finish_sprite(img: Image.Image, cell: tuple[int, int], style: dict, anchor: str, palette=None) -> Image.Image:
    """Matted RGBA -> trimmed, fitted cell, pixelized for pixel styles."""
    img = matting.clean_alpha(img)
    if style.get("pixelate"):
        fitted = fit_to_cell(trim(img), cell, anchor, resample=Image.BOX)
        out = quantize(fitted, int(style.get("palette_colors", 32)), palette)
    else:
        out = fit_to_cell(trim(img), cell, anchor)
    return hard_alpha(out) if style.get("hard_alpha") else matting.clean_alpha(out)


# ---------------------------------------------------------------- sprites


def run_sprites(ctx: JobContext, backends: Backends, subject: str, motion: str, seed: int, st: Stages) -> dict:
    spec = ctx.spec
    style = _style_cfg(spec)
    styles = config.load_models_config().get("styles", {})
    cell = (spec.frame_size, spec.frame_size)
    anchor = "bottom" if spec.asset_type == "character" else "center"
    directions = prompts.DIRECTIONS[spec.directions]
    raw_dir = ctx.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Frame] = []
    anim_groups: list[tuple[str, list[str]]] = []
    units = [(v, d) for v in range(spec.variants) for d in directions]
    gen_cb = st.progress("generate")
    anim_cb = st.progress("animate") if spec.animate.enabled else None
    if spec.animate.enabled and backends.video is None:
        raise ValueError("animation requested but no video backend configured")
    reference: Image.Image | None = None

    for u, (v, d) in enumerate(units):
        base = spec.name if len(units) == 1 else f"{spec.name}_{v}" + (f"_{d}" if d else "")
        if spec.variants == 1 and d:
            base = f"{spec.name}_{d}"
        pp = prompts.build_prompt(spec, styles, subject, direction=d)

        def unit_cb(frac, msg, u=u):
            gen_cb((u + frac) / len(units), f"[{base}] {msg}")

        req = ImageRequest(
            prompt=pp.positive,
            negative=pp.negative,
            width=1024,
            height=1024,
            seed=seed + v * 1000,  # same seed across directions helps consistency
            n=spec.candidates,
            transparent=not spec.animate.enabled,
            style=spec.style,
            references=[reference] if (reference is not None and d) else [],
        )
        raw = _generate_single_subject(ctx, backends, req, subject, unit_cb, base)
        raw.save(raw_dir / f"{base}.png")
        if d and reference is None:
            reference = raw.convert("RGB")

        unit_cb(1.0, "removing background")
        matted = matting.keep_main_subject(matting.matte(raw))
        sprite = finish_sprite(matted, cell, style, anchor)
        palette = build_palette(sprite, int(style.get("palette_colors", 32))) if style.get("pixelate") else None
        frames.append(Frame(base, sprite))

        if spec.animate.enabled:
            anim_frames = _animate(ctx, backends, raw, matted, base, subject, motion, d, seed + v * 1000, cell,
                                   style, anchor, palette, anim_cb, u, len(units))
            anim_groups.append((f"{base}_{spec.animate.action}", [f.name for f in anim_frames]))
            frames.extend(anim_frames)

    return _pack_sprites(ctx, frames, anim_groups, st)


def _animate(ctx, backends, raw, matted, base, subject, motion, direction, seed, cell, style, anchor, palette,
             cb, u, n_units) -> list[Frame]:
    spec = ctx.spec
    mp = prompts.motion_prompt(spec, subject, motion, direction)

    def vcb(frac, msg):
        cb((u + frac * 0.85) / n_units, f"[{base}] {msg}")

    # Video models want an opaque keyframe; composite the matted subject on white
    # so background noise from the still does not animate.
    key = Image.new("RGBA", raw.size, (255, 255, 255, 255))
    key.alpha_composite(matted.convert("RGBA").resize(raw.size))
    video_frames, cost = backends.video.generate(
        VideoRequest(image=key.convert("RGB"), prompt=mp.positive, negative=mp.negative, seed=seed), vcb
    )
    ctx.cost += cost
    from ..animate.frames import sample_frames, stabilize

    picked = sample_frames(video_frames, spec.animate.frames, spec.animate.loop)
    matted_frames = []
    for i, f in enumerate(picked):
        cb((u + 0.85 + 0.15 * i / len(picked)) / n_units, f"[{base}] matting frame {i + 1}/{len(picked)}")
        matted_frames.append(matting.clean_alpha(matting.matte(f)))
    matted_frames = stabilize(matted_frames)
    resample = Image.BOX if style.get("pixelate") else Image.LANCZOS
    fitted = fit_frames_to_cell(matted_frames, cell, anchor, resample=resample)
    out = []
    for i, f in enumerate(fitted):
        if style.get("pixelate"):
            f = quantize(f, int(style.get("palette_colors", 32)), palette)
        f = hard_alpha(f) if style.get("hard_alpha") else matting.clean_alpha(f)
        out.append(Frame(f"{base}_{spec.animate.action}_{i:02d}", f))
    anim_dir = ctx.out_dir / "raw" / f"{base}_{spec.animate.action}"
    anim_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(picked):
        f.save(anim_dir / f"{i:02d}.png")
    return out


def _pack_sprites(ctx: JobContext, frames: list[Frame], anim_groups: list[tuple[str, list[str]]], st: Stages) -> dict:
    spec = ctx.spec
    cb = st.progress("pack")
    assets = ctx.out_dir / "assets"
    entries: list[dict] = []
    anim_files: list[str] = []
    names = [f.name for f in frames]

    cb(0.1, "packing atlas")
    if spec.output.atlas:
        pages = build_atlas(frames, spec.output.max_atlas_size, spec.output.padding, spec.output.extrude)
        entries.append(write_atlas(pages, assets, spec.name))
        if anim_groups:
            data = {"anims": [], "globalTimeScale": 1}
            for key, fnames in anim_groups:
                data["anims"] += phaser_anims(key, spec.name, fnames, spec.animate.fps, spec.animate.loop)["anims"]
            (assets / f"{spec.name}_anims.json").write_text(json.dumps(data, indent=1))
            anim_files.append(f"{spec.name}_anims.json")

    cb(0.5, "packing spritesheet")
    if spec.output.spritesheet:
        key = f"{spec.name}_sheet"
        sheet, cfg = build_spritesheet(frames, cell=(spec.frame_size, spec.frame_size), spacing=spec.output.padding)
        entries.append(write_spritesheet(sheet, cfg, assets, key))
        if anim_groups:
            data = {"anims": [], "globalTimeScale": 1}
            for anim_key, fnames in anim_groups:
                idx = [names.index(n) for n in fnames]
                data["anims"] += phaser_anims(anim_key, key, idx, spec.animate.fps, spec.animate.loop)["anims"]
            (assets / f"{key}_anims.json").write_text(json.dumps(data, indent=1))
            anim_files.append(f"{key}_anims.json")

    frames_dir = assets / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        f.image.save(frames_dir / f"{f.name}.png")
    return {
        "kind": "sprites",
        "entries": entries,
        "anims": [{"key": k, "frames": v} for k, v in anim_groups],
        "anim_files": anim_files,
        "frames": names,
        "frame_size": spec.frame_size,
    }


# ---------------------------------------------------------------- tiles


def run_tiles(ctx: JobContext, backends: Backends, subject: str, seed: int, st: Stages) -> dict:
    spec = ctx.spec
    style = _style_cfg(spec)
    styles = config.load_models_config().get("styles", {})
    pp = prompts.build_prompt(spec, styles, subject)
    gen_cb = st.progress("generate")
    raw_dir = ctx.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pixel = bool(style.get("pixelate"))
    size = (spec.tile_size, spec.tile_size)

    chunk = spec.tile_chunk or auto_tile_chunk(spec.tile_size, pixel)
    block = (spec.tile_size * chunk, spec.tile_size * chunk)
    tiles: list[Frame] = []
    palette = None
    for v in range(spec.variants):
        def vcb(frac, msg, v=v):
            gen_cb((v + frac) / spec.variants, f"[tile {v + 1}/{spec.variants}] {msg}")

        req = ImageRequest(prompt=pp.positive, negative=pp.negative, width=1024, height=1024, seed=seed + v,
                           n=spec.candidates, tiling=True, style=spec.style)
        result = backends.image.generate(req, vcb)
        ctx.cost += result.cost
        raw = _pick_best(ctx, backends, subject, result.images).convert("RGBA")
        raw.putalpha(255)
        raw.save(raw_dir / f"tile_{v}.png")
        if not result.native_tiling:
            raw = make_tileable(raw, threshold=0)
        texture = resize_wrap(raw, block, Image.BOX if pixel else Image.LANCZOS)
        if pixel:
            palette = palette or build_palette(texture, int(style.get("palette_colors", 32)))
            texture = quantize(texture, palette=palette)
        ctx.note(f"tile {v} seam error {seam_error(texture):.2f}")
        # A chunk x chunk block of tiles; the block as a whole repeats seamlessly.
        for r in range(chunk):
            for c in range(chunk):
                x, y = c * spec.tile_size, r * spec.tile_size
                tile = texture.crop((x, y, x + spec.tile_size, y + spec.tile_size))
                if spec.view == "iso":
                    tile = to_iso(tile, spec.tile_size, depth=spec.iso_depth, pixel=pixel)
                name = f"{spec.name}_{v}" if chunk == 1 else f"{spec.name}_{v}_r{r}c{c}"
                tiles.append(Frame(name, tile))

    cb = st.progress("pack")
    cb(0.2, "building tile sheet")
    assets = ctx.out_dir / "assets"
    entries = []
    tw, th = tiles[0].image.size
    if spec.output.tilesheet:
        sheet, cfg = build_tilesheet(tiles, tw, th, extrude=spec.output.extrude)
        entries.append(write_tilesheet(sheet, cfg, assets, f"{spec.name}_tiles",
                                       orientation="isometric" if spec.view == "iso" else "orthogonal"))
    if spec.output.atlas:
        pages = build_atlas(tiles, spec.output.max_atlas_size, spec.output.padding, spec.output.extrude, trim=False)
        entries.append(write_atlas(pages, assets, spec.name))
    frames_dir = assets / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for t in tiles:
        t.image.save(frames_dir / f"{t.name}.png")
    return {
        "kind": "tiles",
        "entries": entries,
        "frames": [t.name for t in tiles],
        "tile_width": tw,
        "tile_height": th,
        "orientation": "isometric" if spec.view == "iso" else "orthogonal",
        "iso_depth": spec.iso_depth if spec.view == "iso" else 0,
        "chunk": chunk,
        "variants": spec.variants,
    }


# ---------------------------------------------------------------- backgrounds


def run_background(ctx: JobContext, backends: Backends, subject: str, seed: int, st: Stages) -> dict:
    spec = ctx.spec
    style = _style_cfg(spec)
    styles = config.load_models_config().get("styles", {})
    layers = prompts.parallax_layers(spec.parallax_layers)
    gen_cb = st.progress("generate")
    assets = ctx.out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    raw_dir = ctx.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    size = (spec.background_width, spec.background_height)
    entries = []
    for i, layer in enumerate(layers):
        def lcb(frac, msg, i=i, layer=layer):
            gen_cb((i + frac) / len(layers), f"[layer {layer}] {msg}")

        pp = prompts.build_prompt(spec, styles, subject, layer=layer)
        req = ImageRequest(prompt=pp.positive, negative=pp.negative, width=size[0], height=size[1],
                           seed=seed + i, n=spec.candidates, style=spec.style,
                           transparent=layer not in ("full", "far"))
        result = backends.image.generate(req, lcb)
        ctx.cost += result.cost
        raw = _pick_best(ctx, backends, subject, result.images)
        raw.save(raw_dir / f"layer_{layer}.png")
        img = raw.convert("RGBA")
        if layer not in ("full", "far"):
            lcb(1.0, "removing background")
            img = matting.clean_alpha(matting.matte(raw))
        img = _cover(img, size)
        if style.get("pixelate"):
            scale = 4
            small = pixelize(img, (size[0] // scale, size[1] // scale), int(style.get("palette_colors", 32)),
                             alpha=layer not in ("full", "far"))
            img = small.resize(size, Image.NEAREST)
        key = f"{spec.name}_{layer}" if len(layers) > 1 else spec.name
        img.save(assets / f"{key}.png", optimize=True)
        entries.append({"loader": "image", "key": key, "texture": f"{key}.png", "layer": layer,
                        "scrollFactor": round(0.2 + 0.8 * i / max(1, len(layers) - 1), 2) if len(layers) > 1 else 1})
    st.progress("pack")(0.5, "writing layers")
    return {"kind": "background", "entries": entries, "width": size[0], "height": size[1]}


def _cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale to cover size and center-crop (like CSS object-fit: cover)."""
    scale = max(size[0] / img.width, size[1] / img.height)
    resized = img.resize((max(size[0], round(img.width * scale)), max(size[1], round(img.height * scale))),
                         Image.LANCZOS)
    x = (resized.width - size[0]) // 2
    y = (resized.height - size[1]) // 2
    return resized.crop((x, y, x + size[0], y + size[1]))


# ---------------------------------------------------------------- entry


def run(ctx: JobContext, backends: Backends) -> dict:
    spec = ctx.spec
    started = time.time()
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    seed = spec.seed if spec.seed is not None else random.randint(0, 2**31 - 1)
    animate = spec.animate.enabled and spec.asset_type in ("character", "prop")
    if animate and backends.video is None:
        raise ValueError("animation requested but no video backend configured")
    if animate:
        weights = {"enhance": 0.03, "generate": 0.35, "animate": 0.55, "pack": 0.07}
    else:
        weights = {"enhance": 0.03, "generate": 0.9, "pack": 0.07}
    st = Stages(ctx, weights)

    subject, motion = spec.prompt, ""
    if spec.enhance_prompt and backends.enhance is not None:
        st.progress("enhance")(0.0, "enhancing prompt")
        try:
            action = None
            if animate:
                action = spec.animate.custom_motion if spec.animate.action == "custom" else spec.animate.action
            enhanced = backends.enhance(spec.prompt, spec.asset_type, spec.style, action)
            subject, motion = enhanced["subject"], enhanced.get("motion", "")
            ctx.note(f"enhanced subject: {subject}")
        except Exception as e:
            ctx.note(f"prompt enhancement skipped: {e}")
        st.progress("enhance")(1.0, "prompt ready")
    st.progress("generate")(0.0, "loading image model")

    if spec.asset_type in ("character", "prop"):
        result = run_sprites(ctx, backends, subject, motion, seed, st)
    elif spec.asset_type == "tile":
        result = run_tiles(ctx, backends, subject, seed, st)
    else:
        result = run_background(ctx, backends, subject, seed, st)

    assets = ctx.out_dir / "assets"
    anim_files = result.get("anim_files", [])
    (assets / "phaser-loader.js").write_text(loader_snippet(result["entries"], anim_files[0] if anim_files else None))
    manifest = {
        "job_id": ctx.job_id,
        "spec": spec.model_dump(),
        "seed": seed,
        "subject": subject,
        "motion": motion,
        "result": result,
        "cost_usd": round(ctx.cost, 4),
        "seconds": round(time.time() - started, 1),
        "log": ctx.log,
    }
    write_manifest(assets, manifest)
    st.progress("pack")(0.9, "zipping")
    zip_dir(assets, ctx.out_dir / f"{spec.name}.zip")
    st.progress("pack")(1.0, "done")
    return manifest
