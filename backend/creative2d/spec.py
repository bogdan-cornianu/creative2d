"""Job specification shared by the API, CLI and pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

AssetType = Literal["character", "prop", "tile", "background"]
View = Literal["side", "topdown", "iso"]
Backend = Literal["local", "openrouter"]
Action = Literal["idle", "walk", "run", "attack", "jump", "custom"]


class AnimateOptions(BaseModel):
    enabled: bool = False
    action: Action = "idle"
    custom_motion: str = ""
    frames: int = Field(8, ge=2, le=64)
    fps: int = Field(10, ge=1, le=60)
    loop: bool = True
    backend: Backend = "local"
    model: str | None = None  # video profile (local) or OpenRouter video model id


class OutputOptions(BaseModel):
    spritesheet: bool = True
    atlas: bool = True
    tilesheet: bool = True
    padding: int = Field(2, ge=0, le=32)
    extrude: int = Field(1, ge=0, le=8)
    max_atlas_size: int = Field(2048, ge=256, le=8192)


class JobSpec(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    name: str = Field("asset", pattern=r"^[A-Za-z0-9_-]{1,48}$")
    asset_type: AssetType = "character"
    style: str = "pixel"
    view: View = "side"
    tile_size: int = Field(32, ge=8, le=512)
    frame_size: int = Field(64, ge=8, le=1024)
    background_width: int = Field(1024, ge=128, le=4096)
    background_height: int = Field(576, ge=128, le=4096)
    parallax_layers: int = Field(1, ge=1, le=4)
    directions: Literal[1, 4, 8] = 1  # character/prop facing directions
    tile_chunk: int | None = Field(None, ge=1, le=8)  # tiles per side per texture; None = auto
    iso_depth: int = Field(0, ge=0, le=512)  # iso tiles: side-face height in px (0 = flat diamond)
    variants: int = Field(1, ge=1, le=16)
    candidates: int = Field(1, ge=1, le=8)  # generated per variant; best kept
    seed: int | None = None

    backend: Backend = "local"
    image_model: str | None = None  # local profile name or OpenRouter model id
    enhance_prompt: bool = False
    text_model: str | None = None  # OpenRouter model for prompt enhancement
    vision_model: str | None = None  # OpenRouter model for candidate QA

    animate: AnimateOptions = AnimateOptions()
    output: OutputOptions = OutputOptions()

    @model_validator(mode="after")
    def _check(self) -> "JobSpec":
        if self.animate.enabled and self.asset_type not in ("character", "prop"):
            raise ValueError("animation is only supported for character and prop assets")
        if self.animate.action == "custom" and self.animate.enabled and not self.animate.custom_motion:
            raise ValueError("custom animation needs custom_motion text")
        if self.enhance_prompt and not self.text_model:
            raise ValueError("enhance_prompt needs text_model")
        return self
