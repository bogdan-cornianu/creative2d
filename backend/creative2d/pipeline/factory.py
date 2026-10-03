"""Build backends for a job from its spec and the saved settings."""

from __future__ import annotations

from typing import Callable

from .. import config, settings_store
from ..gen.openrouter.client import OpenRouterClient
from ..spec import JobSpec
from .runner import Backends


def build_backends(spec: JobSpec, is_cancelled: Callable[[], bool]) -> tuple[Backends, OpenRouterClient | None]:
    prefs = settings_store.get_prefs()
    needs_or = (
        spec.backend == "openrouter"
        or (spec.animate.enabled and spec.animate.backend == "openrouter")
        or spec.enhance_prompt
        or bool(spec.vision_model)
    )
    client = OpenRouterClient(settings_store.get_api_key()) if needs_or else None

    if spec.backend == "openrouter":
        from ..gen.openrouter_backend import OpenRouterImageBackend

        model = spec.image_model or prefs.get("image_model")
        if not model:
            raise ValueError("choose an OpenRouter image model")
        image = OpenRouterImageBackend(client, model, is_cancelled)
    else:
        if not config.ml_available():
            raise RuntimeError("local backend needs ML deps: uv sync --extra ml")
        from ..gen.diffusers_backend import DiffusersImageBackend

        image = DiffusersImageBackend(spec.image_model, is_cancelled)

    video = None
    if spec.animate.enabled:
        if spec.animate.backend == "openrouter":
            from ..gen.openrouter_backend import OpenRouterVideoBackend

            model = spec.animate.model or prefs.get("video_model")
            if not model:
                raise ValueError("choose an OpenRouter video model")
            video = OpenRouterVideoBackend(client, model, is_cancelled)
        else:
            from ..animate.i2v import LocalVideoBackend

            video = LocalVideoBackend(spec.animate.model, is_cancelled)

    enhance = rank = None
    if spec.enhance_prompt and spec.text_model:
        from ..gen.enhance import enhance_prompt

        text_model = spec.text_model

        def enhance(request, asset_type, style, action):
            return enhance_prompt(client, text_model, request, asset_type, style, action)

    if spec.vision_model and spec.candidates > 1:
        from ..gen.enhance import rank_candidates

        vision_model = spec.vision_model

        def rank(request, images):
            return rank_candidates(client, vision_model, request, images)

    return Backends(image=image, video=video, enhance=enhance, rank=rank), client
