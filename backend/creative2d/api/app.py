"""FastAPI app: job API, settings, local model downloads, OpenRouter catalog, static outputs and web UI."""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import config, settings_store
from ..gen import downloads, model_cache
from ..gen.openrouter.client import OpenRouterClient, OpenRouterError
from ..gen.openrouter.models import list_models
from ..jobs.store import FINAL, JobStore
from ..jobs.worker import BackendFactory, EventHub, Worker
from ..pipeline.factory import build_backends
from ..spec import JobSpec


class SettingsUpdate(BaseModel):
    openrouter_api_key: str | None = None  # "" clears the stored key; None leaves it unchanged
    prefs: dict | None = None


class DownloadRequest(BaseModel):
    ids: list[str]


def create_app(
    factory: BackendFactory | None = None,
    db_path: Path | None = None,
    downloader: downloads.DownloadManager | None = None,
) -> FastAPI:
    config.ensure_dirs()
    app = FastAPI(title="creative2d")
    store = JobStore(db_path or config.DATA_DIR / "jobs.sqlite")
    hub = EventHub()
    worker = Worker(store, hub, factory or build_backends)
    app.state.store, app.state.worker = store, worker
    if downloader is not None:
        downloads.install(downloader)
    dl = downloads.manager()

    # ------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        return {
            "device": config.detect_device(),
            "ml_available": config.ml_available(),
            "loaded_models": model_cache.loaded(),
            "current_job": worker.current,
            "output_dir": str(config.OUTPUT_DIR.resolve()),
        }

    @app.get("/api/options")
    def options():
        cfg = config.load_models_config()
        return {
            "asset_types": ["character", "prop", "tile", "background"],
            "views": ["side", "topdown", "iso"],
            "actions": ["idle", "walk", "run", "attack", "jump", "custom"],
            "styles": {k: v.get("label", k) for k, v in cfg.get("styles", {}).items()},
            "image_profiles": {k: v.get("label", k) for k, v in cfg.get("image_profiles", {}).items()},
            "default_image_profile": cfg.get("default_image_profile"),
            "video_profiles": {k: v.get("label", k) for k, v in cfg.get("video_profiles", {}).items()},
            "default_video_profile": cfg.get("default_video_profile"),
            "ml_available": config.ml_available(),
            "defaults": JobSpec(prompt="x").model_dump(exclude={"prompt"}),
        }

    # ------------------------------------------------------------ settings
    @app.get("/api/settings")
    def get_settings():
        return settings_store.public_settings()

    @app.put("/api/settings")
    def put_settings(body: SettingsUpdate):
        if body.openrouter_api_key is not None:
            settings_store.set_api_key(body.openrouter_api_key or None)
        if body.prefs is not None:
            settings_store.set_prefs(body.prefs)
        return settings_store.public_settings()

    @app.post("/api/settings/test")
    def test_settings():
        client = OpenRouterClient(settings_store.get_api_key())
        try:
            info = client.key_info()
        except OpenRouterError as e:
            raise HTTPException(e.status or 502, str(e))
        finally:
            client.close()
        return {"ok": True, "label": info.get("label"), "limit": info.get("limit"), "usage": info.get("usage")}

    @app.get("/api/openrouter/credits")
    def openrouter_credits():
        if not settings_store.get_api_key():
            raise HTTPException(404, "no OpenRouter key set")
        client = OpenRouterClient(settings_store.get_api_key())
        try:
            return client.balance()
        except OpenRouterError as e:
            raise HTTPException(e.status or 502, str(e))
        finally:
            client.close()

    # ------------------------------------------------------------ local models
    @app.get("/api/local-models")
    def local_models():
        ml = config.ml_available()
        return {"ml_available": ml, "models": dl.list() if ml else []}

    @app.post("/api/local-models/download")
    def download_local_models(body: DownloadRequest):
        if not config.ml_available():
            raise HTTPException(409, "local models need ML deps: uv sync --extra ml")
        try:
            dl.enqueue(body.ids)
        except KeyError as e:
            raise HTTPException(404, f"unknown local model {e.args[0]!r}")
        return {"models": dl.list()}

    @app.get("/api/openrouter/models")
    def openrouter_models(kind: str = Query("image", pattern="^(image|text|vision|video)$"), refresh: bool = False):
        client = OpenRouterClient(settings_store.get_api_key())
        try:
            return {"kind": kind, "models": list_models(client, kind, refresh)}
        except OpenRouterError as e:
            raise HTTPException(502, str(e))
        finally:
            client.close()

    # ------------------------------------------------------------ jobs
    @app.post("/api/jobs")
    def create_job(spec: JobSpec):
        return {"id": worker.submit(spec)}

    @app.get("/api/jobs")
    def list_jobs(limit: int = 100):
        return store.list(limit)

    def _job(job_id: str) -> dict:
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return _job(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        _job(job_id)
        return {"cancelled": worker.cancel(job_id)}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job = _job(job_id)
        if job["status"] not in FINAL:
            raise HTTPException(409, "cancel the job before deleting it")
        out = (config.OUTPUT_DIR / job_id).resolve()
        if out.parent == config.OUTPUT_DIR.resolve() and out.exists():
            shutil.rmtree(out)
        store.delete(job_id)
        return {"deleted": True}

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str):
        job = _job(job_id)
        if job["status"] != "done":
            raise HTTPException(409, "job not finished")
        name = job["spec"]["name"]
        path = config.OUTPUT_DIR / job_id / f"{name}.zip"
        if not path.exists():
            raise HTTPException(404, "zip missing")
        return FileResponse(path, filename=f"{name}.zip", media_type="application/zip")

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str):
        job = _job(job_id)
        q = hub.subscribe(job_id)

        async def stream():
            try:
                snapshot = {k: job[k] for k in ("id", "status", "progress", "stage", "message")}
                yield f"data: {json.dumps(snapshot)}\n\n"
                if job["status"] in FINAL:
                    return
                while True:
                    try:
                        event = await asyncio.to_thread(q.get, True, 15)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in FINAL:
                        return
            finally:
                hub.unsubscribe(job_id, q)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")
    if config.WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=config.WEB_DIST, html=True), name="web")
    return app

