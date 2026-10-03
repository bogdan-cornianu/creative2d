import threading
import time

import pytest
from fastapi.testclient import TestClient

from creative2d import config
from creative2d.api.app import create_app
from creative2d.gen import downloads
from creative2d.pipeline.runner import Backends

from fakes import FakeImageBackend

MODELS_YAML = """
image_profiles:
  sdxl:
    label: "SDXL"
    pipeline: sdxl
    repo: org/sdxl
    vae: org/vae
    size_gb: 7
  flux:
    label: "FLUX"
    pipeline: flux
    repo: org/flux
    gated: true
default_image_profile: sdxl
styles:
  pixel:
    lora: {repo: org/pixel-lora, weight_name: pixel.safetensors}
  painted: {}
video_profiles:
  ltx:
    label: "LTX"
    pipeline: ltx
    repo: org/ltx
default_video_profile: ltx
matting:
  repo: org/matting
"""


@pytest.fixture
def models_env(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML)
    monkeypatch.setattr(config, "MODELS_FILE", path)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config, "detect_device", lambda: "mps")
    config.load_models_config.cache_clear()
    yield
    config.load_models_config.cache_clear()


def _wait_status(m, entry_id, statuses, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        row = next(r for r in m.list() if r["id"] == entry_id)
        if row["status"] in statuses:
            return row
        time.sleep(0.02)
    raise TimeoutError(row)


def test_catalog(models_env):
    entries = {e.id: e for e in downloads.catalog()}
    assert set(entries) == {"image:sdxl", "image:flux", "matting", "video:ltx"}
    sdxl = entries["image:sdxl"]
    assert sdxl.default and sdxl.size_gb == 7
    assert [p.repo for p in sdxl.parts] == ["org/sdxl", "org/vae", "org/pixel-lora"]
    assert sdxl.parts[0].variant == "fp16"
    assert sdxl.parts[2].filename == "pixel.safetensors"
    # LoRAs are SDXL-only.
    assert [p.repo for p in entries["image:flux"].parts] == ["org/flux"]
    assert entries["image:flux"].gated and not entries["image:flux"].default
    assert entries["matting"].default and entries["video:ltx"].default


def test_download_marks_ready(models_env):
    fetched = []
    gate = threading.Event()

    def fetch(part):
        gate.wait(5)
        fetched.append(part.repo)

    m = downloads.DownloadManager(fetch)
    with pytest.raises(downloads.ModelNotDownloaded, match="SDXL is not downloaded"):
        m.require("image:sdxl")
    m.enqueue(["image:sdxl"])
    _wait_status(m, "image:sdxl", {"downloading"})
    gate.set()
    _wait_status(m, "image:sdxl", {"ready"})
    assert fetched == ["org/sdxl", "org/vae", "org/pixel-lora"]
    m.require("image:sdxl")
    # The record survives a new manager (server restart).
    assert downloads.DownloadManager(fetch).is_ready(next(e for e in downloads.catalog() if e.id == "image:sdxl"))


def test_download_error_and_gated_message(models_env):
    class Resp:
        status_code = 401

    def fetch(part):
        e = RuntimeError("nope")
        e.response = Resp()
        raise e

    m = downloads.DownloadManager(fetch)
    m.enqueue(["image:flux"])
    row = _wait_status(m, "image:flux", {"error"})
    assert "huggingface-cli login" in row["error"]
    with pytest.raises(downloads.ModelNotDownloaded):
        m.require("image:flux")


def test_changed_profile_needs_new_download(models_env, tmp_path):
    m = downloads.DownloadManager(lambda part: None)
    m.enqueue(["video:ltx"])
    _wait_status(m, "video:ltx", {"ready"})
    (tmp_path / "models.yaml").write_text(MODELS_YAML.replace("org/ltx", "org/ltx2"))
    config.load_models_config.cache_clear()
    assert next(r for r in m.list() if r["id"] == "video:ltx")["status"] == "missing"


def test_unknown_id(models_env):
    with pytest.raises(KeyError):
        downloads.DownloadManager(lambda part: None).enqueue(["image:nope"])


@pytest.fixture
def api(models_env, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "ml_available", lambda: True)
    fetched = []
    app = create_app(
        factory=lambda spec, c: (Backends(image=FakeImageBackend()), None),
        db_path=tmp_path / "data" / "jobs.sqlite",
        downloader=downloads.DownloadManager(lambda part: fetched.append(part.repo)),
    )
    with TestClient(app) as c:
        yield c, fetched
    downloads.install(downloads.DownloadManager())


def test_api_local_models(api):
    client, fetched = api
    body = client.get("/api/local-models").json()
    assert body["ml_available"] is True
    assert {m["id"]: m["status"] for m in body["models"]} == {
        "image:sdxl": "missing",
        "image:flux": "missing",
        "matting": "missing",
        "video:ltx": "missing",
    }
    assert client.post("/api/local-models/download", json={"ids": ["matting"]}).status_code == 200
    end = time.time() + 5
    while time.time() < end:
        rows = {m["id"]: m for m in client.get("/api/local-models").json()["models"]}
        if rows["matting"]["status"] == "ready":
            break
        time.sleep(0.02)
    assert rows["matting"]["status"] == "ready"
    assert fetched == ["org/matting"]
    assert client.post("/api/local-models/download", json={"ids": ["image:nope"]}).status_code == 404


def test_api_local_models_without_ml(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(config, "ml_available", lambda: False)
    assert client.get("/api/local-models").json() == {"ml_available": False, "models": []}
    assert client.post("/api/local-models/download", json={"ids": ["matting"]}).status_code == 409
