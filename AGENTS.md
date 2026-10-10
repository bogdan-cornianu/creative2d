# AGENTS.md

Guide for AI coding agents working on creative2d. Human-facing docs are in `README.md`; read it for setup and user features.

## Git rules (mandatory)

- **Never commit or push to `main`.** Create a branch first (`git checkout -b <short-kebab-name>`), commit there, push the branch, and open a PR into `main`. Check `git branch --show-current` before every commit.
- Past PRs: `ui-fixes-and-ssl-guard`, `openrouter-credits`, `mcp-server`. Match that style: one branch per change, descriptive branch name.
- Commit only when asked. Do not amend or force-push unless asked. Do not commit `outputs/`, `data/`, `.env`, `secrets.env` or `web/dist/` (all gitignored). `uv.lock` is committed; keep it in sync with `pyproject.toml` (`uv lock`).
- Commit messages: short imperative subject line (e.g. "Show remaining OpenRouter credits in the top bar"), body explains why.

## What this is

Local pipeline and web app that generates 2D game assets for Phaser: characters, props, tiles, backgrounds. It generates an image (local diffusers or OpenRouter), post-processes it locally (matting, trim, pixel-art, seamless, iso), optionally animates it (image-to-video), then packs atlases, spritesheets, tilesheets and Tiled `.tsj`. It is exposed as a web app, a CLI and an MCP server.

## Commands

```sh
uv sync --extra ml                  # dev env; omit --extra ml for OpenRouter-only
uv run pytest                       # whole suite, no GPU needed, ~5 s
uv run pytest tests/test_mcp.py -x  # one file
uv run creative2d serve             # API + web UI on http://127.0.0.1:8000
uv run creative2d gen "red slime" --type character --size 64    # in-process, no server
uv run creative2d mcp               # stdio MCP server; needs `serve` running
cd web && npm install && npm run build   # build UI into web/dist (served by `serve`)
cd web && npm run dev               # Vite on :5173, proxies /api to :8000
```

Python is pinned to 3.12 (`>=3.12,<3.13`), managed with `uv`. Run Python through `uv run` (never bare `python`). There is no linter or formatter configured; match surrounding style (type hints, `from __future__ import annotations`, short docstrings, ~120 columns).

## Layout and where to change things

```
backend/creative2d/
  spec.py            JobSpec (pydantic): THE job schema shared by API, CLI, MCP, pipeline
  config.py          paths (OUTPUT_DIR, DATA_DIR, CONFIG_DIR, MODELS_FILE), device detection, models.yaml loader
  settings_store.py  OpenRouter key (secrets.env, 0600) + default-model prefs (settings.json)
  cli.py             argparse: serve | gen | mcp
  mcp_server.py      MCP tools; thin async HTTP client over the API
  api/app.py         FastAPI routes (jobs, settings, models, SSE events, /outputs static, web UI)
  jobs/store.py      SQLite JobStore; jobs/worker.py one-thread FIFO Worker + EventHub
  pipeline/          factory.py (build_backends), runner.py (run_sprites/tiles/background), prompts.py
  gen/               base.py protocols, diffusers_backend.py, openrouter_backend.py, openrouter/ client+catalog, downloads.py, enhance.py
  post/              matting, pixelate, seamless, isometric, trim
  animate/           i2v.py (local video), frames.py (sampling/stabilizing)
  pack/              atlas (MaxRects), spritesheet, tilesheet, export (Phaser/Tiled/zip)
web/src/             React UI (JobForm, JobList, AssetView, SettingsDialog, ModelPicker, PhaserPreview)
models.yaml          local model profiles, style presets, video profiles
tests/               pytest; fakes.py has FakeImageBackend / FakeVideoBackend
```

Adding a job option usually touches, in order: `spec.py` (field + validation) -> `pipeline/runner.py` (use it) -> `web/src/JobForm.tsx` and `web/src/api.ts` (UI) -> `mcp_server.py` `generate_asset` (add the parameter and pass it into `fields`) -> README. `GET /api/options` returns `JobSpec` defaults, so the UI picks up new defaults automatically.

Adding a style: edit `models.yaml` only; it shows up in the UI and in `get_options`.

## Architecture notes

- **Job flow:** `POST /api/jobs` -> `Worker.submit` -> thread calls `factory(spec, is_cancelled)` (`pipeline.factory.build_backends`) -> `runner.run(JobContext, Backends)` -> manifest stored in the job row. Statuses: `queued | running | done | failed | cancelled`. Cancellation is cooperative (raises `Cancelled` at the next progress call). One job at a time, by design (single GPU).
- **Outputs:** `outputs/<job_id>/assets/` (game-ready: `<name>.png` + `.json` atlas, `_sheet.png`, `_anims.json`, `_tiles.png`, `.tsj`, `frames/`, `phaser-loader.js`, `manifest.json`), `raw/`, and `<name>.zip`. Job id format `YYYYmmdd-HHMMSS-<6hex>`.
- **Backends:** local = `DiffusersImageBackend` (SDXL + style LoRA, or FLUX; MPS/CUDA/CPU); cloud = OpenRouter. Models never auto-download. Missing models fail the job with a message naming the model.
- **Default models:** `settings.json` prefs (`image_model`, `video_model`, `text_model`, `vision_model`) apply to OpenRouter jobs only, via `build_backends`. Local jobs use the spec's profile or `default_image_profile` in `models.yaml`.
- **Startup sweep:** `JobStore` marks queued/running jobs as failed ("server restarted"). Never run two processes against the same `data/jobs.sqlite`; this is why the MCP server is an HTTP client and not a second worker.
- **MCP server:** stdio, built on `mcp` 2.x (`from mcp.server.mcpserver import MCPServer, Image`; v1's `FastMCP` does not exist). **Never print to stdout** in `mcp_server.py` or anything it imports; stdout is the protocol. Log to stderr. Tool functions return dicts/lists (lists arrive wrapped as `{"result": [...]}` in `structured_content`). Raise `ToolError` for user-facing failures. `create_server(base_url, client)` accepts an injected `httpx.AsyncClient` for tests.
- **Config reads at call time:** use `config.OUTPUT_DIR` etc. as attribute access (`from creative2d import config`), not `from config import OUTPUT_DIR`, or test monkeypatching breaks.

## Testing

- Write tests for every behavior change. Tests never need a GPU or network: `tests/test_pipeline.py` and `test_api.py` patch `config.ml_available` to False and use `fakes.py` backends.
- API tests: copy the `client` fixture in `tests/test_api.py` (monkeypatch `config.OUTPUT_DIR/DATA_DIR/CONFIG_DIR` to `tmp_path`, inject a fake factory into `create_app(factory=..., db_path=...)`).
- MCP tests: `tests/test_mcp.py` mounts the FastAPI app behind `httpx.ASGITransport` and calls `mcp.call_tool(...)` directly; no subprocess.
- Frontend has no tests; verify with `npm run build` (runs `tsc -b`) and, for visual changes, run `serve` and look at it in a browser.
- Before saying work is done: `uv run pytest` passes, and if `web/` changed, `npm run build` passes.

## Gotchas learned the hard way

- **SDXL + pixel-art LoRA** draws a whole sheet of creatures for ~half of seeds. `subject_share` check + automatic reseed (max 3) in `runner.py` handles it. Keep it when changing generation.
- **Prompt order:** subject first, style fragments after. Style-first prompts lose color adherence (red slime came out green).
- **Tiles:** 1024 px textures carry ~128 px of detail, so 32 px tiles are cut as a seamless k x k block (`tile_chunk`, auto via `auto_tile_chunk`), not downscaled whole. Blocks are named `name_<variant>_r<row>c<col>`.
- **SDXL often draws three-quarter view** even for `view=side`; cloud models follow view instructions better.
- **Animation:** local default is Wan 2.2 TI2V 5B (~3 min/clip on M5 Pro, ~30 GB download). LTX-Video is faster but barely moves legs. On 48 GB machines the Wan transformer and text encoder move to CPU during VAE decode to avoid MPS OOM; keep that. Needs `protobuf` and `ftfy`.
- **Frontend font:** Schibsted Grotesk's `tnum` feature widens punctuation; apply tabular numbers only to numeric elements.
- **SSL:** `cli._drop_unwritable_keylog` removes an unwritable `SSLKEYLOGFILE` so HTTPS calls do not crash; keep it running before any HTTP client is built.
- **Do not reintroduce LM Studio.** Cloud features go through OpenRouter (images, videos, chat completions).
- Dev machine is Apple Silicon (MPS, 48 GB); there is no CUDA. Local ML jobs are slow, so prefer fakes in tests and OpenRouter for quick manual checks.

## Security

- The OpenRouter key is stored in `~/.config/creative2d/secrets.env` (0600) and must never be returned by the API, logged, or put in test output. `public_settings()` returns only a masked form.
- `get_asset_file` and the delete route resolve paths under `OUTPUT_DIR`; keep rejecting `..` and absolute paths.
- The server binds to 127.0.0.1 and has no auth. Do not bind it to a public interface by default.
