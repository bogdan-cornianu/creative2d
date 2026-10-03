"""OpenRouter image and video backends."""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

from PIL import Image

from .base import Cancelled, ImageRequest, ImageResult, Progress, VideoRequest, no_progress
from .openrouter.client import OpenRouterClient, OpenRouterError, image_to_data_url
from .openrouter.models import image_model_params, video_model

_RATIOS = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "16:9": 16 / 9, "9:16": 9 / 16, "3:2": 1.5, "2:3": 2 / 3, "21:9": 21 / 9}


def closest_ratio(width: int, height: int, allowed: list[str] | None = None) -> str:
    target = width / height
    options = [r for r in (allowed or _RATIOS) if r in _RATIOS] or list(_RATIOS)
    return min(options, key=lambda r: abs(math.log(_RATIOS[r] / target)))


class OpenRouterImageBackend:
    name = "openrouter"

    def __init__(self, client: OpenRouterClient, model: str, is_cancelled=lambda: False) -> None:
        self.client = client
        self.model = model
        self.is_cancelled = is_cancelled

    def build_body(self, req: ImageRequest, params: dict) -> dict:
        """Request body with only the parameters this model advertises."""
        prompt = req.prompt
        if req.negative:
            prompt += f"\nAvoid: {req.negative}"
        body: dict = {"model": self.model, "prompt": prompt, "n": 1}
        if "aspect_ratio" in params:
            allowed = params["aspect_ratio"].get("values") if params["aspect_ratio"].get("type") == "enum" else None
            body["aspect_ratio"] = closest_ratio(req.width, req.height, allowed)
        if "seed" in params:
            body["seed"] = req.seed
        if "output_format" in params and "png" in (params["output_format"].get("values") or ["png"]):
            body["output_format"] = "png"
        if req.transparent and "background" in params and "transparent" in (params["background"].get("values") or []):
            body["background"] = "transparent"
        if "quality" in params and "high" in (params["quality"].get("values") or []):
            body["quality"] = "high"
        if req.references and "input_references" in params:
            limit = int(params["input_references"].get("max") or len(req.references))
            body["input_references"] = [
                {"type": "image_url", "image_url": {"url": image_to_data_url(r)}} for r in req.references[:limit]
            ]
        return body

    def generate(self, req: ImageRequest, progress: Progress = no_progress) -> ImageResult:
        params = image_model_params(self.client, self.model)
        images: list[Image.Image] = []
        cost = 0.0
        for i in range(req.n):
            if self.is_cancelled():
                raise Cancelled()
            progress(i / req.n, f"OpenRouter {self.model}: image {i + 1}/{req.n}")
            body = self.build_body(req, params)
            if "seed" in body:
                body["seed"] = req.seed + i
            imgs, c = self.client.generate_images(body)
            images.extend(im.convert("RGBA") for im in imgs[:1])
            cost += c
        progress(1.0, "images received")
        return ImageResult(images=images, native_tiling=False, cost=cost)


class OpenRouterVideoBackend:
    name = "openrouter-video"

    def __init__(self, client: OpenRouterClient, model: str, is_cancelled=lambda: False, poll_seconds: float = 5.0):
        self.client = client
        self.model = model
        self.is_cancelled = is_cancelled
        self.poll_seconds = poll_seconds

    def generate(self, req: VideoRequest, progress: Progress = no_progress) -> tuple[list[Image.Image], float]:
        info = video_model(self.client, self.model)
        if info is None:
            raise OpenRouterError(f"{self.model} is not an image-to-video model on OpenRouter")
        body: dict = {
            "model": self.model,
            "prompt": req.prompt + (f"\nAvoid: {req.negative}" if req.negative else ""),
            "frame_images": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(req.image)}, "frame_type": "first_frame"}
            ],
        }
        if info["durations"]:
            body["duration"] = min(info["durations"])
        if info["aspect_ratios"]:
            body["aspect_ratio"] = closest_ratio(req.image.width, req.image.height, info["aspect_ratios"])
        if info["resolutions"]:
            body["resolution"] = info["resolutions"][0]
        job = self.client.create_video(body)
        job_id = job["id"]
        started = time.time()
        while True:
            if self.is_cancelled():
                raise Cancelled()
            status = self.client.get_video(job_id)
            state = status.get("status")
            if state == "completed":
                break
            if state == "failed":
                raise OpenRouterError(f"video generation failed: {status.get('error') or status}")
            elapsed = time.time() - started
            progress(min(0.9, elapsed / 180), f"OpenRouter video {state} ({int(elapsed)}s)")
            time.sleep(self.poll_seconds)
        progress(0.95, "downloading video")
        data = self.client.download_video(job_id)
        cost = float(status.get("usage", {}).get("cost") or 0.0)
        return read_video_frames(data), cost


def read_video_frames(data: bytes) -> list[Image.Image]:
    import imageio.v3 as iio

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        path.write_bytes(data)
        return [Image.fromarray(f).convert("RGB") for f in iio.imiter(path)]
