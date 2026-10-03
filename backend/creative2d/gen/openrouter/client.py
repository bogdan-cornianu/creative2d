"""Thin OpenRouter HTTP client."""

from __future__ import annotations

import base64
import io
import time

import httpx
from PIL import Image

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


_STATUS_HINTS = {
    401: "invalid or missing OpenRouter API key",
    402: "insufficient OpenRouter credits",
    403: "request blocked by OpenRouter (moderation or key limits)",
    408: "OpenRouter request timed out",
    429: "OpenRouter rate limit reached",
}


def _raise_for(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        detail = resp.json().get("error", {}).get("message") or resp.text
    except ValueError:
        detail = resp.text
    hint = _STATUS_HINTS.get(resp.status_code, "OpenRouter request failed")
    raise OpenRouterError(f"{hint} ({resp.status_code}): {detail[:300]}", resp.status_code)


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def decode_image(b64_or_data_url: str) -> Image.Image:
    data = b64_or_data_url.split(",", 1)[1] if b64_or_data_url.startswith("data:") else b64_or_data_url
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    img.load()
    return img


class OpenRouterClient:
    def __init__(self, api_key: str | None, base_url: str = BASE_URL, transport: httpx.BaseTransport | None = None):
        self.api_key = api_key
        headers = {"HTTP-Referer": "http://localhost", "X-Title": "creative2d"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=300, transport=transport)

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, url: str, retries: int = 2, **kw) -> httpx.Response:
        for attempt in range(retries + 1):
            try:
                resp = self._http.request(method, url, **kw)
            except httpx.TransportError as e:
                if attempt == retries:
                    raise OpenRouterError(f"network error talking to OpenRouter: {e}") from e
                time.sleep(2**attempt)
                continue
            if resp.status_code in (429, 502, 503) and attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            _raise_for(resp)
            return resp
        raise AssertionError("unreachable")

    def _need_key(self) -> None:
        if not self.api_key:
            raise OpenRouterError("OpenRouter API key not set (Settings page or OPENROUTER_API_KEY)", 401)

    # --- discovery (public, no key needed) ---
    def list_models(self, output_modalities: str = "text") -> list[dict]:
        return self._request("GET", "/models", params={"output_modalities": output_modalities}).json()["data"]

    def list_image_models(self) -> list[dict]:
        return self._request("GET", "/images/models").json()["data"]

    def list_video_models(self) -> list[dict]:
        data = self._request("GET", "/videos/models").json()
        return data.get("data", data) if isinstance(data, dict) else data

    def key_info(self) -> dict:
        self._need_key()
        return self._request("GET", "/key", retries=0).json().get("data", {})

    def balance(self) -> dict:
        """Credits left, in USD.

        source "account": account balance from /credits (needs a management key).
        source "key": what is left under this key's own spending limit.
        source "usage": neither is readable; only what this key has spent is known.
        """
        self._need_key()
        try:
            d = self._request("GET", "/credits", retries=0).json()["data"]
            total, used = float(d["total_credits"]), float(d["total_usage"])
            return {"source": "account", "remaining": round(total - used, 6), "total": total, "used": used}
        except OpenRouterError as e:
            if e.status != 403:
                raise
        info = self.key_info()
        used = float(info.get("usage") or 0.0)
        if info.get("limit_remaining") is not None:
            return {"source": "key", "remaining": info["limit_remaining"], "total": info.get("limit"), "used": used}
        return {"source": "usage", "remaining": None, "total": None, "used": used}

    # --- generation ---
    def generate_images(self, body: dict) -> tuple[list[Image.Image], float]:
        """POST /images. Returns decoded images and reported cost (USD)."""
        self._need_key()
        data = self._request("POST", "/images", json=body).json()
        images = [decode_image(item["b64_json"]) for item in data.get("data", []) if item.get("b64_json")]
        if not images:
            raise OpenRouterError(f"model {body.get('model')} returned no image")
        return images, float(data.get("usage", {}).get("cost") or 0.0)

    def chat(self, body: dict) -> dict:
        self._need_key()
        return self._request("POST", "/chat/completions", json=body).json()

    def create_video(self, body: dict) -> dict:
        self._need_key()
        return self._request("POST", "/videos", json=body).json()

    def get_video(self, job_id: str) -> dict:
        return self._request("GET", f"/videos/{job_id}").json()

    def download_video(self, job_id: str, index: int = 0) -> bytes:
        return self._request("GET", f"/videos/{job_id}/content", params={"index": index}).content
