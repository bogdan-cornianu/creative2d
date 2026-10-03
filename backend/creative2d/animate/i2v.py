"""Local image-to-video (Wan 2.2 TI2V / LTX-Video) via diffusers."""

from __future__ import annotations

from PIL import Image

from .. import config
from ..gen import downloads, model_cache
from ..gen.base import Cancelled, Progress, VideoRequest, no_progress


def _load(profile: dict, device: str):
    import torch
    from diffusers import LTXImageToVideoPipeline, WanImageToVideoPipeline

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    kind = profile["pipeline"]
    if kind == "wan":
        pipe = WanImageToVideoPipeline.from_pretrained(profile["repo"], torch_dtype=dtype, local_files_only=True)
    elif kind == "ltx":
        pipe = LTXImageToVideoPipeline.from_pretrained(profile["repo"], torch_dtype=dtype, local_files_only=True)
    else:
        raise ValueError(f"unsupported video pipeline {kind!r}")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if kind == "ltx":
        pipe.vae.enable_tiling()
    # Wan's VAE tiling raises peak memory on MPS (measured 67 GB vs 53 GB), so it stays off.
    pipe._c2d_offloaded = False
    return pipe


# Components that are idle while the VAE decodes; moved off the GPU to make room.
_DECODE_OFFLOAD = ("transformer", "text_encoder")


def _restore(pipe, device: str) -> None:
    if getattr(pipe, "_c2d_offloaded", False):
        for name in _DECODE_OFFLOAD:
            getattr(pipe, name).to(device)
        pipe._c2d_offloaded = False


def _decode_wan(pipe, latents, device: str) -> list[Image.Image]:
    """Decode Wan latents with the transformer and text encoder off the GPU.

    On 48 GB Apple Silicon the full pipeline peaks above the MPS limit during
    decode; offloading drops the peak from ~53 GB to ~39 GB. Components move
    back lazily at the start of the next Wan job (about 50 s).
    """
    import torch

    for name in _DECODE_OFFLOAD:
        getattr(pipe, name).to("cpu")
    pipe._c2d_offloaded = True
    if device == "mps":
        torch.mps.empty_cache()
    vae = pipe.vae
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(latents.device, vae.dtype)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(latents.device, vae.dtype)
    with torch.no_grad():
        video = vae.decode(latents.to(vae.dtype) / inv_std + mean, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type="pil")[0]


def valid_frame_count(n: int, kind: str) -> int:
    """Wan needs 4k+1 frames, LTX needs 8k+1."""
    step = 8 if kind == "ltx" else 4
    return max(step + 1, (n - 1) // step * step + 1)


class LocalVideoBackend:
    name = "local-video"

    def __init__(self, profile_name: str | None = None, is_cancelled=lambda: False) -> None:
        cfg = config.load_models_config()
        self.profile_name = profile_name or cfg.get("default_video_profile", "ltx-video")
        profiles = cfg.get("video_profiles", {})
        if self.profile_name not in profiles:
            raise ValueError(f"unknown local video profile {self.profile_name!r}; options: {list(profiles)}")
        self.profile = profiles[self.profile_name]
        self.device = config.detect_device()
        self.is_cancelled = is_cancelled

    def generate(self, req: VideoRequest, progress: Progress = no_progress) -> tuple[list[Image.Image], float]:
        import torch

        progress(0.0, f"loading {self.profile.get('label', self.profile_name)}")
        downloads.require(f"video:{self.profile_name}")
        pipe = model_cache.get(f"video:{self.profile_name}:{self.device}", lambda: _load(self.profile, self.device))
        _restore(pipe, self.device)
        kind = self.profile["pipeline"]
        w, h = int(self.profile.get("width", 704)), int(self.profile.get("height", 704))
        # Keep the keyframe's aspect ratio, dims multiple of 32.
        ar = req.image.width / req.image.height
        if ar >= 1:
            h = max(32, int(w / ar) // 32 * 32)
        else:
            w = max(32, int(h * ar) // 32 * 32)
        image = req.image.convert("RGB").resize((w, h), Image.LANCZOS)
        steps = int(self.profile.get("steps", 30))
        num_frames = valid_frame_count(req.num_frames or int(self.profile.get("num_frames", 49)), kind)

        def on_step(_pipe, step, _t, kw):
            if self.is_cancelled():
                raise Cancelled()
            progress((step + 1) / steps, f"video step {step + 1}/{steps}")
            return kw

        kw = dict(
            image=image,
            prompt=req.prompt,
            negative_prompt=req.negative or None,
            width=w,
            height=h,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=float(self.profile.get("guidance", 5.0)),
            generator=torch.Generator("cpu").manual_seed(req.seed),
            callback_on_step_end=on_step,
            output_type="pil",
        )
        if kind == "ltx":
            kw["frame_rate"] = int(self.profile.get("fps", 24))
            frames = pipe(**kw).frames[0]
        else:
            kw["output_type"] = "latent"
            latents = pipe(**kw).frames
            progress(1.0, "decoding video")
            frames = _decode_wan(pipe, latents, self.device)
        return [f.convert("RGB") for f in frames], 0.0
