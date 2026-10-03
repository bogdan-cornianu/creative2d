import json
import time

import pytest
from fastapi.testclient import TestClient

from creative2d import config
from creative2d.api.app import create_app
from creative2d.pipeline.runner import Backends

from fakes import FakeImageBackend, FakeVideoBackend


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config, "ml_available", lambda: False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def factory(spec, is_cancelled):
        return Backends(image=FakeImageBackend(), video=FakeVideoBackend()), None

    app = create_app(factory=factory, db_path=tmp_path / "data" / "jobs.sqlite")
    with TestClient(app) as c:
        yield c


def _wait(client, job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise TimeoutError(job)


def test_job_lifecycle(client):
    r = client.post("/api/jobs", json={"prompt": "slime", "name": "slime", "frame_size": 32, "seed": 1})
    assert r.status_code == 200
    job_id = r.json()["id"]
    job = _wait(client, job_id)
    assert job["status"] == "done", job
    assert job["result"]["result"]["frames"] == ["slime"]
    assert client.get(f"/outputs/{job_id}/assets/slime.png").status_code == 200
    z = client.get(f"/api/jobs/{job_id}/download")
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
    row = next(j for j in client.get("/api/jobs").json() if j["id"] == job_id)
    assert row["thumb"] == "frames/slime.png"
    assert client.get(f"/outputs/{job_id}/assets/{row['thumb']}").status_code == 200
    # SSE on a finished job returns the final snapshot and closes.
    with client.stream("GET", f"/api/jobs/{job_id}/events") as s:
        lines = [l for l in s.iter_lines() if l.startswith("data:")]
    assert json.loads(lines[0][5:])["status"] == "done"
    assert client.delete(f"/api/jobs/{job_id}").json() == {"deleted": True}
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_invalid_spec_rejected(client):
    r = client.post("/api/jobs", json={"prompt": "", "name": "x"})
    assert r.status_code == 422


def test_custom_animation_job(client):
    r = client.post("/api/jobs", json={"prompt": "x", "asset_type": "prop", "style": "painted",
                                       "animate": {"enabled": True, "action": "custom", "custom_motion": "spin"}})
    job = _wait(client, r.json()["id"])
    assert job["status"] == "done"
    assert job["result"]["result"]["anims"][0]["key"] == "asset_custom"


def test_failed_job_reports_error(client):
    def broken(spec, is_cancelled):
        raise ValueError("choose an OpenRouter image model")

    client.app.state.worker.factory = broken
    job = _wait(client, client.post("/api/jobs", json={"prompt": "x"}).json()["id"])
    assert job["status"] == "failed" and "OpenRouter image model" in job["error"]


def test_settings_key_never_returned(client):
    secret = "sk-or-v1-abcdefghijklmnopqrstuvwxyz"
    r = client.put("/api/settings", json={"openrouter_api_key": secret, "prefs": {"image_model": "a/b", "evil": 1}})
    body = r.json()
    assert body["openrouter_key_set"] is True
    assert secret not in json.dumps(body)
    assert body["prefs"] == {"image_model": "a/b"}
    path = config.CONFIG_DIR / "secrets.env"
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert secret not in json.dumps(client.get("/api/settings").json())
    client.put("/api/settings", json={"openrouter_api_key": ""})
    assert client.get("/api/settings").json()["openrouter_key_set"] is False


def test_options(client):
    o = client.get("/api/options").json()
    assert "pixel" in o["styles"] and "sdxl" in o["image_profiles"]
    assert o["defaults"]["frame_size"] == 64
