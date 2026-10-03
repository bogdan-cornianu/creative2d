"""Fake backends that draw simple shapes, so the pipeline runs without a GPU."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from creative2d.gen.base import ImageRequest, ImageResult, VideoRequest


class FakeImageBackend:
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ImageRequest] = []

    def generate(self, req: ImageRequest, progress=lambda f, m: None) -> ImageResult:
        self.requests.append(req)
        images = []
        for i in range(req.n):
            if req.tiling:
                rng = np.random.default_rng(req.seed + i)
                arr = rng.integers(60, 120, (256, 256, 3), dtype=np.uint8)
                arr[..., 1] += 80
                img = Image.fromarray(arr, "RGB").convert("RGBA")
            else:
                w, h = (req.width // 4, req.height // 4)
                img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
                d = ImageDraw.Draw(img)
                d.ellipse((w * 0.3, h * 0.2, w * 0.7, h * 0.9), fill=(200, 40, 40, 255))
            images.append(img)
            progress((i + 1) / req.n, "fake")
        return ImageResult(images=images, native_tiling=False, cost=0.01)


class FakeVideoBackend:
    name = "fake-video"

    def generate(self, req: VideoRequest, progress=lambda f, m: None):
        frames = []
        w, h = 256, 256
        for i in range(24):
            img = Image.new("RGB", (w, h), (255, 255, 255))
            d = ImageDraw.Draw(img)
            bob = int(10 * np.sin(i / 24 * 2 * np.pi))
            d.ellipse((80 + i, 60 + bob, 176 + i, 230), fill=(40, 40, 200))
            frames.append(img)
        progress(1.0, "fake video")
        return frames, 0.0
