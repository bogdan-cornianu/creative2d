"""Prompt templates per asset type, view and style."""

from __future__ import annotations

from dataclasses import dataclass

from ..spec import JobSpec

VIEW_TEXT = {
    "side": "side view, full profile, facing right, orthographic, 2D platformer",
    "topdown": "top-down view, seen from directly above",
    "iso": "isometric view, three-quarter view from above, facing down-right",
}

PLAIN_BG = "isolated on a plain flat solid white background, no shadow, no ground, no text, no border"

ACTION_MOTION = {
    "idle": "subtle idle animation, gentle breathing, character stays in place",
    "walk": "walking in place, legs stepping forward and back, alternating feet, arms swinging, treadmill walk, "
            "character does not move across the frame",
    "run": "run cycle in place, fast leg movement, character does not move across the frame",
    "attack": "performs a quick attack swing then returns to the starting pose",
    "jump": "jumps straight up and lands back in the same spot",
}

VIDEO_SUFFIX = "static camera, locked-off shot, plain white background stays unchanged, smooth motion, same character"
VIDEO_NEGATIVE = "camera movement, zoom, pan, background change, extra characters, morphing, text, watermark"


@dataclass
class PromptPair:
    positive: str
    negative: str


def _style(styles: dict, key: str) -> dict:
    return styles.get(key) or {"prompt": key, "negative": ""}


DIRECTIONS = {
    1: [None],
    4: ["down", "left", "right", "up"],
    8: ["down", "down-left", "left", "up-left", "up", "up-right", "right", "down-right"],
}

FACING = {
    "down": "facing the viewer",
    "up": "facing away from the viewer, back view",
    "left": "facing left",
    "right": "facing right",
    "down-left": "facing down-left",
    "down-right": "facing down-right",
    "up-left": "facing up-left, back three-quarter view",
    "up-right": "facing up-right, back three-quarter view",
}


def view_text(view: str, direction: str | None) -> str:
    if direction is None:
        return VIEW_TEXT[view]
    base = {"side": "side view, orthographic", "topdown": "top-down view, seen from above",
            "iso": "isometric view, three-quarter view from above"}[view]
    return f"{base}, {FACING[direction]}"


def build_prompt(
    spec: JobSpec, styles: dict, subject: str | None = None, layer: str | None = None, direction: str | None = None
) -> PromptPair:
    s = _style(styles, spec.style)
    subject = subject or spec.prompt
    view = view_text(spec.view, direction)
    neg = [s.get("negative", ""), "blurry, cropped, cut off, multiple views, watermark, signature, text"]
    t = spec.asset_type
    if t == "character":
        # Subject first: SDXL's text encoders weight early tokens most.
        pos = f"{subject}, single full body character, {view}, centered, {s['prompt']}, {PLAIN_BG}"
        neg.append("sprite sheet, character sheet, multiple characters, many, group, grid, collection, "
                   "icons, pattern, turnaround, cropped feet, busy background")
    elif t == "prop":
        pos = f"{subject}, single game item, {view}, centered, {s['prompt']}, {PLAIN_BG}"
        neg.append("sprite sheet, item sheet, multiple objects, many, grid, collection, icons, pattern, hands, "
                   "busy background")
    elif t == "tile":
        # Iso tiles are generated top-down and projected afterwards.
        pos = (
            f"{subject}, seamless tileable texture, top-down view from directly above, "
            f"flat even lighting, no perspective, fills the entire image edge to edge, {s['prompt']}"
        )
        neg.append("perspective, horizon, vignette, border, frame, objects, characters, shadows from one side")
    elif t == "background":
        kind = {
            "side": "2D side-scrolling game background",
            "topdown": "top-down game map background",
            "iso": "isometric game scene background",
        }[spec.view]
        if layer in (None, "full", "far"):
            pos = f"{s['prompt']}, {kind}, {subject}, wide landscape, atmospheric depth"
            if layer == "far":
                pos += ", distant sky and far scenery only"
        else:
            depth = "foreground" if layer == "near" else "midground"
            pos = f"{s['prompt']}, {depth} elements for a {kind}: {subject}, horizontal strip of scenery, {PLAIN_BG}"
        neg.append("characters, UI, text")
    else:
        raise ValueError(t)
    return PromptPair(pos, ", ".join(n for n in neg if n))


def motion_prompt(
    spec: JobSpec, subject: str | None = None, motion: str | None = None, direction: str | None = None
) -> PromptPair:
    a = spec.animate
    if not motion:
        motion = a.custom_motion if a.action == "custom" else ACTION_MOTION[a.action]
    subject = subject or spec.prompt
    return PromptPair(f"{subject}, {motion}, {view_text(spec.view, direction)}, {VIDEO_SUFFIX}", VIDEO_NEGATIVE)


def parallax_layers(n: int) -> list[str]:
    return {1: ["full"], 2: ["far", "near"], 3: ["far", "mid", "near"], 4: ["far", "mid", "mid2", "near"]}[n]
