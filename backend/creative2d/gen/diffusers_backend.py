"""Local image generation with diffusers (SDXL / FLUX) on MPS, CUDA or CPU."""

from __future__ import annotations

from contextlib import contextmanager

from PIL import Image

from .. import config
from . import downloads, model_cache
from .base import Cancelled, ImageRequest, ImageResult, Progress, no_progress


def _dtype(device: str, pipeline: str):
    import torch

    if device == "cpu":
        return torch.float32
    if pipeline == "flux":
        return torch.bfloat16
    return torch.float16


def _load(profile: dict, device: str):
    import torch
    from diffusers import AutoencoderKL, FluxPipeline, StableDiffusionXLPipeline

    kind = profile["pipeline"]
    dtype = _dtype(device, kind)
    if kind == "sdxl":
        kw = {}
        if profile.get("vae"):
            kw["vae"] = AutoencoderKL.from_pretrained(profile["vae"], torch_dtype=dtype, local_files_only=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            profile["repo"],
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
            local_files_only=True,
            **kw,
        )
    elif kind == "flux":
        pipe = FluxPipeline.from_pretrained(profile["repo"], torch_dtype=dtype, local_files_only=True)
    else:
        raise ValueError(f"unsupported image pipeline {kind!r}")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if device == "mps":
        pipe.enable_attention_slicing()
    pipe._c2d_lora = None  # currently loaded LoRA key
    return pipe


@contextmanager
def seamless(pipe, enabled: bool):
    """Circular padding on every conv layer makes the output wrap around."""
    if not enabled:
        yield
        return
    import torch

    convs = []
    for model in (getattr(pipe, "unet", None), getattr(pipe, "vae", None)):
        if model is None:
            continue
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                convs.append((m, m.padding_mode))
                m.padding_mode = "circular"
    try:
        yield
    finally:
        for m, mode in convs:
            m.padding_mode = mode


def snap(v: int, multiple: int = 64) -> int:
    return max(multiple, int(round(v / multiple)) * multiple)


def native_size(width: int, height: int, native: int) -> tuple[int, int]:
    """Scale requested size to about native^2 pixels, keeping aspect ratio."""
    scale = (native * native / (width * height)) ** 0.5
    return snap(width * scale), snap(height * scale)


class DiffusersImageBackend:
    name = "local"

    def __init__(self, profile_name: str | None = None, is_cancelled=lambda: False) -> None:
        cfg = config.load_models_config()
        self.profile_name = profile_name or cfg.get("default_image_profile", "sdxl")
        profiles = cfg.get("image_profiles", {})
        if self.profile_name not in profiles:
            raise ValueError(f"unknown local image profile {self.profile_name!r}; options: {list(profiles)}")
        self.profile = profiles[self.profile_name]
        self.styles = cfg.get("styles", {})
        self.device = config.detect_device()
        self.is_cancelled = is_cancelled

    def _pipe(self):
        downloads.require(f"image:{self.profile_name}")
        key = f"image:{self.profile_name}:{self.device}"
        return model_cache.get(key, lambda: _load(self.profile, self.device))

    def _apply_lora(self, pipe, style: str | None) -> None:
        lora = (self.styles.get(style or "", {}) or {}).get("lora") if self.profile["pipeline"] == "sdxl" else None
        key = f"{lora['repo']}/{lora.get('weight_name', '')}" if lora else None
        if pipe._c2d_lora == key:
            return
        if pipe._c2d_lora:
            pipe.unload_lora_weights()
        if lora:
            pipe.load_lora_weights(
                lora["repo"], weight_name=lora.get("weight_name"), adapter_name="style", local_files_only=True
            )
            pipe.set_adapters(["style"], adapter_weights=[float(lora.get("scale", 1.0))])
        pipe._c2d_lora = key

    def generate(self, req: ImageRequest, progress: Progress = no_progress) -> ImageResult:
        import torch

        progress(0.0, f"loading {self.profile.get('label', self.profile_name)}")
        pipe = self._pipe()
        self._apply_lora(pipe, req.style)
        width, height = native_size(req.width, req.height, int(self.profile.get("native_size", 1024)))
        steps = int(self.profile.get("steps", 30))
        tiling = req.tiling and self.profile["pipeline"] == "sdxl"
        images: list[Image.Image] = []

        for i in range(req.n):
            done = i

            def on_step(_pipe, step, _t, kw, done=done):
                if self.is_cancelled():
                    raise Cancelled()
                progress((done + (step + 1) / steps) / req.n, f"image {done + 1}/{req.n} step {step + 1}/{steps}")
                return kw

            generator = torch.Generator("cpu").manual_seed(req.seed + i)
            kw = dict(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=float(self.profile.get("guidance", 6.0)),
                generator=generator,
                callback_on_step_end=on_step,
            )
            if self.profile["pipeline"] == "sdxl":
                kw["negative_prompt"] = req.negative or None
            else:
                kw["max_sequence_length"] = 256
            with seamless(pipe, tiling):
                out = pipe(**kw).images[0]
            images.append(out.convert("RGBA"))
        return ImageResult(images=images, native_tiling=tiling)
