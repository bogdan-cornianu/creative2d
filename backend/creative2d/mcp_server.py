"""MCP server (stdio): lets AI agents generate 2D game assets through a running `creative2d serve`.

Stdout carries the protocol, so nothing here may print to it.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .spec import JobSpec

DEFAULT_URL = "http://127.0.0.1:8000"
FINAL = ("done", "failed", "cancelled")
POLL_SECONDS = 2.0

INSTRUCTIONS = """\
Generate 2D game assets (sprites, tiles, parallax backgrounds) with creative2d.
Start with get_options. Then call generate_asset and read the returned file paths.
Prompt tips: put the subject first ("red slime, ..."), describe one subject only,
and leave the style to the `style` argument. For tiles use asset_type=tile; the
result is a seamless texture chunked into a tilesheet (+ Tiled .tsj).
Jobs run one at a time, so queue several and poll with get_job.
"""


def _summary(job: dict, output_dir: str | None) -> dict:
    """Compact job view: status plus the files an agent needs, without the full manifest."""
    out: dict[str, Any] = {k: job.get(k) for k in ("id", "status", "progress", "stage", "message", "error")}
    manifest = job.get("result") or {}
    result = manifest.get("result") or {}
    if manifest:
        out.update(seed=manifest.get("seed"), cost_usd=manifest.get("cost_usd"), seconds=manifest.get("seconds"))
        out["kind"] = result.get("kind")
        for key in ("frames", "frame_size", "anims", "tile_width", "tile_height", "orientation", "chunk", "width", "height"):
            if key in result:
                out[key] = result[key]
    if output_dir and job.get("status") == "done":
        base = Path(output_dir) / job["id"]
        assets = base / "assets"
        out["assets_dir"] = str(assets)
        out["files"] = sorted(str(p.relative_to(assets)) for p in assets.rglob("*") if p.is_file()) if assets.is_dir() else []
        out["zip"] = str(base / f"{job['spec']['name']}.zip")
    return {k: v for k, v in out.items() if v is not None}


def create_server(base_url: str | None = None, client: httpx.AsyncClient | None = None) -> MCPServer:
    """Build the MCP server. `client` is injectable for tests."""
    url = (base_url or os.environ.get("CREATIVE2D_URL") or DEFAULT_URL).rstrip("/")
    http = client or httpx.AsyncClient(base_url=url, timeout=30)
    mcp = MCPServer("creative2d", instructions=INSTRUCTIONS)

    async def call(method: str, path: str, **kw) -> Any:
        try:
            r = await http.request(method, path, **kw)
        except httpx.TransportError as e:
            raise ToolError(f"creative2d server not reachable at {url} ({e.__class__.__name__}). Run: uv run creative2d serve")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ToolError(f"{r.status_code}: {detail}")
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.content

    async def output_dir() -> str | None:
        return (await call("GET", "/api/health")).get("output_dir")

    @mcp.tool()
    async def get_options() -> dict:
        """List asset types, views, styles, animation actions, model profiles and JobSpec defaults."""
        return await call("GET", "/api/options")

    @mcp.tool()
    async def health() -> dict:
        """Server status: compute device, whether local ML models are installed, loaded models, running job."""
        return await call("GET", "/api/health")

    @mcp.tool()
    async def list_models(
        kind: Literal["image", "video", "text", "vision"] = "image",
        backend: Literal["openrouter", "local"] = "openrouter",
        query: str | None = None,
        limit: int = 30,
    ) -> list[dict]:
        """List selectable models. Use the returned `id` as image_model / animate.model / a default.

        backend=openrouter: live catalog (needs an API key on the server); `query` filters id/name,
        e.g. "gemini". backend=local: configured profiles (kind image or video only).
        """
        if backend == "local":
            if kind not in ("image", "video"):
                raise ToolError("local backend has only image and video profiles")
            o = await call("GET", "/api/options")
            profiles = o[f"{kind}_profiles"]
            default = o[f"default_{kind}_profile"]
            return [{"id": k, "name": v, "default": k == default} for k, v in profiles.items()]
        models = (await call("GET", "/api/openrouter/models", params={"kind": kind}))["models"]
        if query:
            q = query.lower()
            models = [m for m in models if q in m["id"].lower() or q in m["name"].lower()]
        return models[:limit]

    @mcp.tool()
    async def get_default_models() -> dict:
        """Saved default models (image_model, video_model, text_model, vision_model) used when a job leaves them unset.

        Defaults apply to OpenRouter jobs; local jobs fall back to the default profile in models.yaml.
        """
        return (await call("GET", "/api/settings"))["prefs"]

    @mcp.tool()
    async def set_default_models(
        image_model: str | None = None,
        video_model: str | None = None,
        text_model: str | None = None,
        vision_model: str | None = None,
    ) -> dict:
        """Save default OpenRouter model ids. Omit a field to keep it; pass "" to clear it.

        These are shared with the web UI. Per-job override: generate_asset(image_model=...,
        animate={"model": ...}).
        """
        prefs = {k: v for k, v in dict(image_model=image_model, video_model=video_model,
                                       text_model=text_model, vision_model=vision_model).items() if v is not None}
        if not prefs:
            raise ToolError("pass at least one of image_model, video_model, text_model, vision_model")
        return (await call("PUT", "/api/settings", json={"prefs": prefs}))["prefs"]

    @mcp.tool()
    async def generate_asset(
        prompt: str,
        name: str = "asset",
        asset_type: Literal["character", "prop", "tile", "background"] = "character",
        style: str = "pixel",
        view: Literal["side", "topdown", "iso"] = "side",
        frame_size: int = 64,
        tile_size: int = 32,
        background_width: int = 1024,
        background_height: int = 576,
        parallax_layers: int = 1,
        directions: Literal[1, 4, 8] = 1,
        tile_chunk: int | None = None,
        iso_depth: int = 0,
        variants: int = 1,
        candidates: int = 1,
        seed: int | None = None,
        backend: Literal["local", "openrouter"] = "local",
        image_model: str | None = None,
        enhance_prompt: bool = False,
        text_model: str | None = None,
        vision_model: str | None = None,
        animate: dict | None = None,
        output: dict | None = None,
        wait: bool = True,
        timeout_s: int = 900,
    ) -> dict:
        """Generate a 2D game asset and return its files.

        asset_type: character/prop -> sprite atlas + spritesheet (frame_size px cells);
        tile -> seamless tilesheet (tile_size px) + Tiled .tsj; background -> 1-4 parallax layers.
        animate (character/prop only): {enabled, action: idle|walk|run|attack|jump|custom,
        custom_motion, frames, fps, loop, backend, model}. output: {spritesheet, atlas, tilesheet,
        padding, extrude, max_atlas_size}. backend=openrouter needs image_model (see get_options).
        With wait=true, blocks until the job ends (or timeout_s) and returns file paths under
        assets_dir. With wait=false, returns the job id at once; poll with get_job.
        """
        fields: dict[str, Any] = dict(
            prompt=prompt, name=name, asset_type=asset_type, style=style, view=view, frame_size=frame_size,
            tile_size=tile_size, background_width=background_width, background_height=background_height,
            parallax_layers=parallax_layers, directions=directions, tile_chunk=tile_chunk, iso_depth=iso_depth,
            variants=variants, candidates=candidates, seed=seed, backend=backend, image_model=image_model,
            enhance_prompt=enhance_prompt, text_model=text_model, vision_model=vision_model,
        )
        if animate is not None:
            fields["animate"] = animate
        if output is not None:
            fields["output"] = output
        try:
            spec = JobSpec.model_validate(fields)
        except ValueError as e:  # pydantic.ValidationError subclasses ValueError
            raise ToolError(f"invalid arguments: {e}")
        job_id = (await call("POST", "/api/jobs", json=spec.model_dump(mode="json")))["id"]
        if not wait:
            return {"id": job_id, "status": "queued"}
        deadline = time.monotonic() + timeout_s
        while True:
            job = await call("GET", f"/api/jobs/{job_id}")
            if job["status"] in FINAL or time.monotonic() >= deadline:
                break
            await asyncio.sleep(POLL_SECONDS)
        return _summary(job, await output_dir())

    @mcp.tool()
    async def get_job(job_id: str) -> dict:
        """Status, progress and (when done) output file paths of one job."""
        return _summary(await call("GET", f"/api/jobs/{job_id}"), await output_dir())

    @mcp.tool()
    async def list_jobs(limit: int = 20) -> list[dict]:
        """Recent jobs, newest first."""
        rows = await call("GET", "/api/jobs", params={"limit": limit})
        keys = ("id", "status", "progress", "stage", "error", "thumb")
        return [{k: r.get(k) for k in keys if r.get(k) is not None} | {"name": r.get("spec", {}).get("name")} for r in rows]

    @mcp.tool()
    async def cancel_job(job_id: str) -> dict:
        """Cancel a queued or running job."""
        return await call("POST", f"/api/jobs/{job_id}/cancel")

    @mcp.tool()
    async def get_asset_file(job_id: str, path: str) -> Image | str:
        """Fetch one output file by path relative to the job's assets dir (see `files` in the job result).

        PNG files come back as an image so you can inspect them; json/tsj/js come back as text.
        """
        if ".." in Path(path).parts or path.startswith("/"):
            raise ToolError("path must be relative to the assets dir")
        data = await call("GET", f"/outputs/{job_id}/assets/{path}")
        if path.lower().endswith(".png"):
            return Image(data=data, format="png")
        return data.decode("utf-8", errors="replace")

    return mcp


def serve(base_url: str | None = None) -> None:
    create_server(base_url).run("stdio")
