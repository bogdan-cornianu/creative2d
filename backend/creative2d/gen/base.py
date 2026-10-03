"""Backend interfaces for image and video generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from PIL import Image

# progress(fraction 0..1 within the current step, message)
Progress = Callable[[float, str], None]


def no_progress(_: float, __: str) -> None:
    pass


class Cancelled(Exception):
    """Raised inside backends when the job was cancelled."""


@dataclass
class ImageRequest:
    prompt: str
    negative: str = ""
    width: int = 1024
    height: int = 1024
    seed: int = 0
    n: int = 1
    tiling: bool = False  # generate a seamless (wrap-around) texture
    transparent: bool = False  # ask for alpha output where the backend supports it
    style: str | None = None  # local backend: style key for LoRA selection
    references: list[Image.Image] = field(default_factory=list)  # image guidance (cloud backends)


@dataclass
class ImageResult:
    images: list[Image.Image]
    native_tiling: bool = False  # True when the backend produced truly seamless output
    cost: float = 0.0


class ImageBackend(Protocol):
    name: str

    def generate(self, req: ImageRequest, progress: Progress = no_progress) -> ImageResult: ...


@dataclass
class VideoRequest:
    image: Image.Image  # first frame (keyframe on plain background)
    prompt: str
    negative: str = ""
    num_frames: int = 0  # 0 = use the video profile's default
    seed: int = 0


class VideoBackend(Protocol):
    name: str

    def generate(self, req: VideoRequest, progress: Progress = no_progress) -> tuple[list[Image.Image], float]:
        """Return frames and cost (USD, 0 for local)."""
        ...
