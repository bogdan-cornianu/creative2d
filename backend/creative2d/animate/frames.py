"""Turn raw video frames into clean, aligned sprite animation frames."""

from __future__ import annotations

import numpy as np
from PIL import Image


def sample_frames(frames: list[Image.Image], count: int, loop: bool) -> list[Image.Image]:
    """Pick `count` evenly spaced frames. For loops, end the clip at the frame
    most similar to the first one so the cycle closes without a jump."""
    if len(frames) <= count:
        return list(frames)
    end = len(frames)
    if loop:
        end = best_loop_end(frames, min_len=max(count, len(frames) // 2))
    idx = np.linspace(0, end, count, endpoint=False).round().astype(int)
    return [frames[i] for i in idx]


def best_loop_end(frames: list[Image.Image], min_len: int) -> int:
    """Index (exclusive end) whose frame best matches frame 0."""

    def small(f: Image.Image) -> np.ndarray:
        return np.asarray(f.convert("RGB").resize((64, 64), Image.BILINEAR)).astype(np.float32)

    first = small(frames[0])
    best, best_err = len(frames), float("inf")
    for i in range(min_len, len(frames)):
        err = float(np.abs(small(frames[i]) - first).mean())
        if err < best_err:
            best, best_err = i, err
    return best


def stabilize(frames: list[Image.Image]) -> list[Image.Image]:
    """Cancel horizontal drift: shift each RGBA frame so its alpha centroid x
    and bottom-most opaque row match the first frame (feet stay planted)."""

    def anchor(img: Image.Image) -> tuple[float, int] | None:
        a = np.asarray(img.getchannel("A")) > 127
        if not a.any():
            return None
        ys, xs = np.nonzero(a)
        return float(xs.mean()), int(ys.max())

    ref = anchor(frames[0])
    if ref is None:
        return frames
    out = []
    for f in frames:
        cur = anchor(f)
        if cur is None:
            out.append(f)
            continue
        dx = round(ref[0] - cur[0])
        dy = ref[1] - cur[1]
        shifted = Image.new("RGBA", f.size, (0, 0, 0, 0))
        shifted.paste(f, (dx, dy))
        out.append(shifted)
    return out
