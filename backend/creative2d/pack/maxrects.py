"""MaxRects bin packer (best short side fit), no rotation.

Phaser supports rotated atlas frames, but unrotated frames keep the export
simple and avoid surprises with pixel art, so rotation is not used.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def intersects(self, o: "Rect") -> bool:
        return not (o.x >= self.right or o.right <= self.x or o.y >= self.bottom or o.bottom <= self.y)

    def contains(self, o: "Rect") -> bool:
        return o.x >= self.x and o.y >= self.y and o.right <= self.right and o.bottom <= self.bottom


class MaxRectsBin:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.free: list[Rect] = [Rect(0, 0, width, height)]
        self.used: list[Rect] = []

    def insert(self, w: int, h: int) -> Rect | None:
        best: Rect | None = None
        best_short = best_long = None
        for f in self.free:
            if w <= f.w and h <= f.h:
                short = min(f.w - w, f.h - h)
                long_ = max(f.w - w, f.h - h)
                if best is None or (short, long_) < (best_short, best_long):
                    best, best_short, best_long = Rect(f.x, f.y, w, h), short, long_
        if best is None:
            return None
        self._split(best)
        self.used.append(best)
        return best

    def _split(self, used: Rect) -> None:
        new_free: list[Rect] = []
        for f in self.free:
            if not f.intersects(used):
                new_free.append(f)
                continue
            if used.x > f.x:
                new_free.append(Rect(f.x, f.y, used.x - f.x, f.h))
            if used.right < f.right:
                new_free.append(Rect(used.right, f.y, f.right - used.right, f.h))
            if used.y > f.y:
                new_free.append(Rect(f.x, f.y, f.w, used.y - f.y))
            if used.bottom < f.bottom:
                new_free.append(Rect(f.x, used.bottom, f.w, f.bottom - used.bottom))
        # Prune rectangles contained in another free rectangle.
        pruned = []
        for i, a in enumerate(new_free):
            if a.w <= 0 or a.h <= 0:
                continue
            if any(j != i and b.contains(a) and (b != a or j < i) for j, b in enumerate(new_free)):
                continue
            pruned.append(a)
        self.free = pruned


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def pack(
    sizes: list[tuple[int, int]], max_size: int, pow2: bool = True
) -> tuple[list[tuple[int, Rect]], list[tuple[int, int]]]:
    """Pack rectangles into as few bins as needed.

    Returns ``(placements, bin_dims)``: one ``(bin_index, rect)`` per input
    size, in input order, and the ``(width, height)`` of each bin. Bins grow
    from small to max_size so single small atlases stay compact.

    Raises ValueError when a rectangle is larger than max_size.
    """
    for w, h in sizes:
        if w > max_size or h > max_size:
            raise ValueError(f"frame {w}x{h} exceeds max atlas size {max_size}")

    order = sorted(range(len(sizes)), key=lambda i: (-max(sizes[i]), -sizes[i][0] * sizes[i][1]))
    remaining = list(order)
    results: dict[int, tuple[int, Rect]] = {}
    bin_dims: list[tuple[int, int]] = []

    while remaining:
        bin_index = len(bin_dims)
        area = sum(sizes[i][0] * sizes[i][1] for i in remaining)
        side = max(max(sizes[i]) for i in remaining)
        side = max(side, int(area**0.5))
        side = _next_pow2(side) if pow2 else side
        side = min(side, max_size)
        # Grow the bin until everything fits or max size is reached.
        while True:
            b = MaxRectsBin(side, side)
            placed: dict[int, Rect] = {}
            leftover = []
            for i in remaining:
                r = b.insert(*sizes[i])
                if r is None:
                    leftover.append(i)
                else:
                    placed[i] = r
            if not leftover or side >= max_size:
                break
            side = min(side * 2 if pow2 else int(side * 1.25) + 1, max_size)
        if not placed:
            raise ValueError("unable to pack frames")
        used_w = max(r.right for r in placed.values())
        used_h = max(r.bottom for r in placed.values())
        if pow2:
            used_w, used_h = _next_pow2(used_w), _next_pow2(used_h)
        bin_dims.append((used_w, used_h))
        for i, r in placed.items():
            results[i] = (bin_index, r)
        remaining = leftover

    return [results[i] for i in range(len(sizes))], bin_dims
