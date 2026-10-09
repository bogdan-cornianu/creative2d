import asyncio
import json

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from creative2d import config
from creative2d.api.app import create_app
from creative2d.mcp_server import create_server
from creative2d.pipeline.runner import Backends

from fakes import FakeImageBackend, FakeVideoBackend


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config, "ml_available", lambda: False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def factory(spec, is_cancelled):
        return Backends(image=FakeImageBackend(), video=FakeVideoBackend()), None

    return create_app(factory=factory, db_path=tmp_path / "data" / "jobs.sqlite")


def _run(app, fn):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            return await fn(create_server(client=http))

    return asyncio.run(go())


def _data(result):
    return result.structured_content or json.loads(result.content[0].text)


def test_tools_listed(app):
    async def fn(mcp):
        return {t.name for t in await mcp.list_tools()}

    assert _run(app, fn) >= {"get_options", "generate_asset", "get_job", "list_jobs", "cancel_job", "get_asset_file"}


def test_generate_waits_and_returns_paths(app):
    async def fn(mcp):
        r = await mcp.call_tool("generate_asset", {"prompt": "slime", "name": "slime", "frame_size": 32, "seed": 1})
        return _data(r)

    out = _run(app, fn)
    assert out["status"] == "done", out
    assert out["kind"] == "sprites"
    assert "slime.png" in out["files"] and "slime.json" in out["files"]
    assert (config.OUTPUT_DIR / out["id"] / "assets" / "slime.png").exists()


def test_no_wait_then_get_job_and_file(app):
    async def fn(mcp):
        queued = _data(await mcp.call_tool("generate_asset", {"prompt": "rock", "name": "rock", "wait": False}))
        for _ in range(200):
            job = _data(await mcp.call_tool("get_job", {"job_id": queued["id"]}))
            if job["status"] in ("done", "failed", "cancelled"):
                break
            await asyncio.sleep(0.05)
        png = await mcp.call_tool("get_asset_file", {"job_id": queued["id"], "path": "rock.png"})
        return queued, job, png

    queued, job, png = _run(app, fn)
    assert queued["status"] == "queued"
    assert job["status"] == "done"
    assert png.content[0].type == "image" and png.content[0].mime_type == "image/png"


def test_invalid_args_readable_error(app):
    async def fn(mcp):
        with pytest.raises(ToolError, match="invalid arguments"):
            await mcp.call_tool("generate_asset", {"prompt": "x", "asset_type": "tile", "animate": {"enabled": True}})

    _run(app, fn)


def test_path_traversal_rejected(app):
    async def fn(mcp):
        with pytest.raises(ToolError, match="relative"):
            await mcp.call_tool("get_asset_file", {"job_id": "x", "path": "../secrets.env"})

    _run(app, fn)


def test_server_down_message():
    async def go():
        async with httpx.AsyncClient(base_url="http://127.0.0.1:9", timeout=2) as http:
            mcp = create_server(base_url="http://127.0.0.1:9", client=http)
            with pytest.raises(ToolError, match="creative2d serve"):
                await mcp.call_tool("get_options", {})

    asyncio.run(go())


def test_default_models_roundtrip(app):
    async def fn(mcp):
        assert _data(await mcp.call_tool("get_default_models", {})) == {}
        saved = _data(await mcp.call_tool("set_default_models", {"image_model": "a/b", "video_model": "c/d"}))
        assert saved == {"image_model": "a/b", "video_model": "c/d"}
        saved = _data(await mcp.call_tool("set_default_models", {"video_model": ""}))
        assert saved["image_model"] == "a/b" and saved["video_model"] == ""
        with pytest.raises(ToolError, match="at least one"):
            await mcp.call_tool("set_default_models", {})

    _run(app, fn)


def test_list_local_models(app):
    async def fn(mcp):
        img = _data(await mcp.call_tool("list_models", {"kind": "image", "backend": "local"}))["result"]
        assert any(m["id"] == "sdxl" for m in img) and sum(m["default"] for m in img) == 1
        with pytest.raises(ToolError, match="only image and video"):
            await mcp.call_tool("list_models", {"kind": "text", "backend": "local"})

    _run(app, fn)
