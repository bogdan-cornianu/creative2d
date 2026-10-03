"""LLM helpers on OpenRouter: prompt enhancement and vision QA ranking."""

from __future__ import annotations

import json
import re

from PIL import Image

from .openrouter.client import OpenRouterClient, image_to_data_url

ENHANCE_SYSTEM = """You write subject descriptions for a 2D game asset image generator.
Given a short request, return JSON only: {"subject": "...", "motion": "..."}.
- subject: one concise visual description (max 40 words) of the thing to draw: shape, colors, materials, key details.
  Do not mention camera, background, art style, resolution or format; those are added separately.
- motion: if an animation is given, one sentence describing that movement for this subject
  (performed in place, loopable, camera static); else "".
"""

QA_SYSTEM = """You grade candidate images for a 2D game asset. Return JSON only:
{"scores": [s0, s1, ...]} with one score 0-10 per image, in order.
Reward: exactly one complete subject matching the request, centered, not cropped, plain clean background,
clear readable silhouette, matches the requested style. Penalize: multiple subjects, cut-off parts, text,
busy or textured background, wrong view angle."""


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError(f"model did not return JSON: {text[:200]!r}")
    return json.loads(m.group(0))


def _content(resp: dict) -> str:
    return resp["choices"][0]["message"].get("content") or ""


def enhance_prompt(
    client: OpenRouterClient, model: str, request: str, asset_type: str, style: str, action: str | None
) -> dict:
    user = f"Asset type: {asset_type}\nArt style: {style}\nAnimation: {action or 'none'}\nRequest: {request}"
    resp = client.chat(
        {
            "model": model,
            "messages": [{"role": "system", "content": ENHANCE_SYSTEM}, {"role": "user", "content": user}],
            "temperature": 0.4,
            "max_tokens": 400,
        }
    )
    data = _parse_json(_content(resp))
    return {"subject": str(data.get("subject") or request).strip(), "motion": str(data.get("motion") or "").strip()}


def rank_candidates(client: OpenRouterClient, model: str, request: str, images: list[Image.Image]) -> list[float]:
    """Score each candidate 0-10 with a vision model."""
    content: list[dict] = [{"type": "text", "text": f"Request: {request}\nImages: {len(images)}"}]
    for img in images:
        thumb = img.convert("RGB")
        thumb.thumbnail((512, 512))
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(thumb)}})
    resp = client.chat(
        {
            "model": model,
            "messages": [{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 200,
        }
    )
    scores = [float(s) for s in _parse_json(_content(resp)).get("scores", [])]
    if len(scores) != len(images):
        raise ValueError(f"expected {len(images)} scores, got {scores}")
    return scores
