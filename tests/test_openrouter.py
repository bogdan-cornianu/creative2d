import base64
import io
import json

import httpx
import pytest
from PIL import Image

from creative2d.gen.base import ImageRequest, VideoRequest
from creative2d.gen.enhance import enhance_prompt, rank_candidates
from creative2d.gen.openrouter import models as or_models
from creative2d.gen.openrouter.client import OpenRouterClient, OpenRouterError
from creative2d.gen.openrouter_backend import OpenRouterImageBackend, OpenRouterVideoBackend, closest_ratio


def _png_b64(color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


IMAGE_MODELS = {
    "data": [
        {
            "id": "acme/img",
            "name": "Acme Img",
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["image"]},
            "supported_parameters": {
                "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9", "9:16"]},
                "seed": {"type": "boolean"},
                "background": {"type": "enum", "values": ["auto", "transparent", "opaque"]},
                "input_references": {"type": "range", "min": 1, "max": 2},
            },
        }
    ]
}
GENERAL = {"data": [{"id": "acme/img", "name": "Acme Img", "pricing": {"image_output": "0.00004", "prompt": "0"}}]}
VIDEO_MODELS = {
    "data": [
        {"id": "acme/vid", "name": "Acme Vid", "supported_frame_images": ["first_frame"], "supported_durations": [4, 8],
         "supported_aspect_ratios": ["1:1", "16:9"], "supported_resolutions": ["480p"], "pricing_skus": {}},
        {"id": "acme/edit", "name": "Edit only", "supported_frame_images": None},
    ]
}


@pytest.fixture(autouse=True)
def clear_cache():
    or_models._cache.clear()


def make_client(handler, key="sk-test"):
    return OpenRouterClient(key, transport=httpx.MockTransport(handler))


def test_image_backend_body_and_decode():
    seen = []

    def handler(req: httpx.Request):
        if req.url.path.endswith("/images/models"):
            return httpx.Response(200, json=IMAGE_MODELS)
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=GENERAL)
        if req.url.path.endswith("/images"):
            assert req.headers["authorization"] == "Bearer sk-test"
            seen.append(json.loads(req.content))
            return httpx.Response(200, json={"data": [{"b64_json": _png_b64()}], "usage": {"cost": 0.02}})
        return httpx.Response(404)

    backend = OpenRouterImageBackend(make_client(handler), "acme/img")
    ref = Image.new("RGB", (4, 4))
    res = backend.generate(ImageRequest(prompt="knight", negative="blur", width=1920, height=1080, seed=7, n=2,
                                        transparent=True, references=[ref]))
    assert len(res.images) == 2 and res.images[0].mode == "RGBA"
    assert res.cost == pytest.approx(0.04)
    body = seen[0]
    assert body["aspect_ratio"] == "16:9" and body["seed"] == 7 and seen[1]["seed"] == 8
    assert body["background"] == "transparent"
    assert body["input_references"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Avoid: blur" in body["prompt"]
    assert "quality" not in body  # not advertised by the model


def test_error_mapping():
    def handler(req):
        if req.url.path.endswith("/images/models"):
            return httpx.Response(200, json=IMAGE_MODELS)
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=GENERAL)
        return httpx.Response(402, json={"error": {"message": "no credits"}})

    backend = OpenRouterImageBackend(make_client(handler), "acme/img")
    with pytest.raises(OpenRouterError) as e:
        backend.generate(ImageRequest(prompt="x"))
    assert e.value.status == 402 and "insufficient" in str(e.value)


def test_missing_key():
    client = make_client(lambda r: httpx.Response(200, json={}), key=None)
    with pytest.raises(OpenRouterError):
        client.generate_images({"model": "a"})


def test_model_catalog_kinds():
    def handler(req):
        path, q = req.url.path, req.url.params.get("output_modalities")
        if path.endswith("/images/models"):
            return httpx.Response(200, json=IMAGE_MODELS)
        if path.endswith("/videos/models"):
            return httpx.Response(200, json=VIDEO_MODELS)
        if path.endswith("/models") and q == "image":
            return httpx.Response(200, json=GENERAL)
        if path.endswith("/models") and q == "text":
            return httpx.Response(200, json={"data": [
                {"id": "t/text", "name": "Text", "architecture": {"input_modalities": ["text"]}, "pricing": {}},
                {"id": "t/vis", "name": "Vis", "architecture": {"input_modalities": ["text", "image"]}, "pricing": {}},
            ]})
        return httpx.Response(404)

    c = make_client(handler)
    img = or_models.list_models(c, "image")
    assert img[0]["pricing"] == {"image_output": "0.00004"}
    assert [m["id"] for m in or_models.list_models(c, "video")] == ["acme/vid"]
    assert [m["id"] for m in or_models.list_models(c, "text")] == ["t/text", "t/vis"]
    assert [m["id"] for m in or_models.list_models(c, "vision")] == ["t/vis"]


def test_video_backend_polls(monkeypatch):
    import imageio.v3 as iio
    import numpy as np

    frames = np.zeros((5, 16, 16, 3), np.uint8)
    video_bytes = iio.imwrite("<bytes>", frames, extension=".mp4", fps=5)
    polls = {"n": 0}
    created = {}

    def handler(req):
        p = req.url.path
        if p.endswith("/videos/models"):
            return httpx.Response(200, json=VIDEO_MODELS)
        if p.endswith("/videos") and req.method == "POST":
            created.update(json.loads(req.content))
            return httpx.Response(200, json={"id": "job1", "status": "pending"})
        if p.endswith("/videos/job1/content"):
            return httpx.Response(200, content=video_bytes)
        if p.endswith("/videos/job1"):
            polls["n"] += 1
            status = "completed" if polls["n"] >= 2 else "in_progress"
            return httpx.Response(200, json={"id": "job1", "status": status, "usage": {"cost": 0.5}})
        return httpx.Response(404)

    backend = OpenRouterVideoBackend(make_client(handler), "acme/vid", poll_seconds=0)
    out, cost = backend.generate(VideoRequest(image=Image.new("RGB", (64, 64)), prompt="walk"))
    assert len(out) >= 4 and cost == 0.5
    assert created["duration"] == 4 and created["aspect_ratio"] == "1:1"
    assert created["frame_images"][0]["frame_type"] == "first_frame"


def test_enhance_and_rank_parse_json():
    def handler(req):
        body = json.loads(req.content)
        if isinstance(body["messages"][1]["content"], list):
            text = 'Sure! {"scores": [3, 8]}'
        else:
            text = '```json\n{"subject": "tiny green frog", "motion": "hops"}\n```'
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    c = make_client(handler)
    assert enhance_prompt(c, "m", "frog", "character", "pixel", "jump") == {"subject": "tiny green frog",
                                                                            "motion": "hops"}
    assert rank_candidates(c, "m", "frog", [Image.new("RGB", (8, 8))] * 2) == [3.0, 8.0]


def test_closest_ratio():
    assert closest_ratio(1024, 1024) == "1:1"
    assert closest_ratio(1920, 1080, ["1:1", "16:9"]) == "16:9"
    assert closest_ratio(1000, 3000, ["1:1", "16:9"]) == "1:1"


def _balance_client(routes):
    def handler(req):
        status, body = routes[req.url.path.removeprefix("/api/v1")]
        return httpx.Response(status, json=body)

    return OpenRouterClient("k", transport=httpx.MockTransport(handler))


def test_balance_from_account_credits():
    c = _balance_client({"/credits": (200, {"data": {"total_credits": 10.0, "total_usage": 2.5}})})
    assert c.balance() == {"source": "account", "remaining": 7.5, "total": 10.0, "used": 2.5}


def test_balance_falls_back_to_key_limit():
    # Inference keys may not read /credits; the key's own cap is the next best answer.
    c = _balance_client({
        "/credits": (403, {"error": {"code": 403, "message": "Only management keys can perform this operation"}}),
        "/key": (200, {"data": {"limit": 20, "limit_remaining": 14.5, "usage": 5.5}}),
    })
    assert c.balance() == {"source": "key", "remaining": 14.5, "total": 20, "used": 5.5}


def test_balance_unknown_without_key_limit():
    c = _balance_client({
        "/credits": (403, {"error": {"message": "nope"}}),
        "/key": (200, {"data": {"limit": None, "limit_remaining": None, "usage": 3.0}}),
    })
    assert c.balance() == {"source": "usage", "remaining": None, "total": None, "used": 3.0}


def test_balance_bad_key_raises():
    c = _balance_client({"/credits": (401, {"error": {"message": "bad"}})})
    with pytest.raises(OpenRouterError) as e:
        c.balance()
    assert e.value.status == 401
