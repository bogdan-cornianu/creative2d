"""Model catalog for the web UI, cached in memory."""

from __future__ import annotations

import threading
import time

from .client import OpenRouterClient

CACHE_TTL = 3600
_cache: dict[str, tuple[float, list[dict]]] = {}
_lock = threading.Lock()


def _price(pricing: dict | None) -> dict:
    """Keep only non-zero per-unit prices (USD strings from OpenRouter)."""
    if not pricing:
        return {}
    return {k: v for k, v in pricing.items() if v not in (None, "0", 0, "")}


def _fetch(client: OpenRouterClient, kind: str) -> list[dict]:
    if kind == "image":
        general = {m["id"]: m for m in client.list_models("image")}
        out = []
        for m in client.list_image_models():
            params = m.get("supported_parameters") or {}
            out.append(
                {
                    "id": m["id"],
                    "name": m.get("name", m["id"]),
                    "description": m.get("description", ""),
                    "input_modalities": m.get("architecture", {}).get("input_modalities", []),
                    "params": params,
                    "pricing": _price(general.get(m["id"], {}).get("pricing")),
                }
            )
        return out
    if kind == "video":
        out = []
        for m in client.list_video_models():
            out.append(
                {
                    "id": m["id"],
                    "name": m.get("name", m["id"]),
                    "description": m.get("description", ""),
                    "frame_images": m.get("supported_frame_images") or [],
                    "durations": m.get("supported_durations") or [],
                    "resolutions": m.get("supported_resolutions") or [],
                    "aspect_ratios": m.get("supported_aspect_ratios") or [],
                    "pricing": m.get("pricing_skus") or {},
                }
            )
        # Only models that can start from our generated keyframe are useful.
        return [m for m in out if "first_frame" in m["frame_images"]]
    if kind in ("text", "vision"):
        out = []
        for m in client.list_models("text"):
            inputs = m.get("architecture", {}).get("input_modalities", [])
            if kind == "vision" and "image" not in inputs:
                continue
            out.append(
                {
                    "id": m["id"],
                    "name": m.get("name", m["id"]),
                    "context_length": m.get("context_length"),
                    "pricing": _price(m.get("pricing")),
                }
            )
        return out
    raise ValueError(f"unknown model kind {kind!r}")


def list_models(client: OpenRouterClient, kind: str, refresh: bool = False) -> list[dict]:
    with _lock:
        hit = _cache.get(kind)
        if hit and not refresh and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
    models = sorted(_fetch(client, kind), key=lambda m: m["name"].lower())
    with _lock:
        _cache[kind] = (time.time(), models)
    return models


def image_model_params(client: OpenRouterClient, model_id: str) -> dict:
    for m in list_models(client, "image"):
        if m["id"] == model_id:
            return m["params"]
    return {}


def video_model(client: OpenRouterClient, model_id: str) -> dict | None:
    return next((m for m in list_models(client, "video") if m["id"] == model_id), None)
