# creative2d

Local pipeline and web app for generating 2D game assets for Phaser: characters, props, terrain tiles and backgrounds, with transparent backgrounds, texture atlases, spritesheets, tile sheets and optional animation.

- **Local generation** on your GPU (Apple Silicon MPS, CUDA or CPU) with diffusers: SDXL plus style LoRAs, or FLUX.
- **Cloud generation** through OpenRouter (Gemini image, GPT Image, FLUX.2, Seedream, Recraft and more). You pick models in the web app.
- **Post-processing always runs locally**: background removal (BiRefNet), trimming, bottom-center anchoring, pixel-art downscale and palette lock, seamless tiles, isometric projection.
- **Animation (optional)**: the keyframe is animated with an image-to-video model (local LTX-Video / Wan 2.2, or an OpenRouter video model), then frames are sampled, matted, stabilized and packed.
- **Phaser-ready export**: JSON-hash atlas (or multiatlas), spritesheet grid, `anims.fromJSON` data, tile sheet with extrusion, Tiled `.tsj` tileset, loader snippet and a zip.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Node 20+.

```sh
uv sync --extra ml          # Python 3.12 venv with torch/diffusers (omit --extra ml for OpenRouter-only use)
cd web && npm install && npm run build && cd ..
uv run creative2d serve     # http://127.0.0.1:8000
```

For UI development run `npm run dev` in `web/` (port 5173, proxies to the API on 8000).

Local models never download on their own. Open **Settings → Local models** in the web app and press **Download models** for the defaults (SDXL with the pixel-art LoRA, BiRefNet and the default video model), or download single models from the list (SDXL about 7 GB, Wan 2.2 and LTX-Video about 30 GB each). A local job that needs a missing model fails with a message saying which one. FLUX.1-schnell is gated: accept its licence on Hugging Face and run `uv run huggingface-cli login` first.

### OpenRouter

Open **Settings** in the web app and paste your key. It is stored in `~/.config/creative2d/secrets.env` (mode 0600) and never sent back to the browser. `OPENROUTER_API_KEY` in the environment also works.

With a key you can:

- generate images with any model from OpenRouter's image catalog,
- animate with OpenRouter video models that accept a first frame,
- expand short prompts with a text model,
- generate several candidates and let a vision model pick the best.

## Command line

```sh
uv run creative2d gen "red slime monster" --type character --style pixel --size 64
uv run creative2d gen "mossy cobblestone floor" --type tile --size 32 --variants 4
uv run creative2d gen "grassy meadow" --type tile --view iso --size 128 --style painted
uv run creative2d gen "knight" --size 96 --animate walk --frames 8 --directions 4
uv run creative2d gen "castle" --backend openrouter --model google/gemini-3.1-flash-image
```

Output lands in `outputs/<job id>/`: `assets/` (what goes into your game), `raw/` (unprocessed generations) and a zip of `assets/`.

## MCP server (for AI agents)

`creative2d mcp` runs a stdio MCP server that drives a running `creative2d serve` over HTTP, so agents share the web UI's job queue, models and history. Start `serve` first.

```sh
claude mcp add creative2d -- uv run --directory /path/to/creative2d creative2d mcp
```

Other clients (Claude Desktop etc.):

```json
{"mcpServers": {"creative2d": {"command": "uv", "args": ["run", "--directory", "/path/to/creative2d", "creative2d", "mcp"]}}}
```

Server URL defaults to `http://127.0.0.1:8000`; override with `--url` or `CREATIVE2D_URL`.

Tools: `get_options`, `health`, `list_models` (OpenRouter catalog or local profiles), `get_default_models` / `set_default_models` (image, video, text, vision; shared with the web UI), `generate_asset` (blocks until done by default; `wait=false` returns a job id), `get_job`, `list_jobs`, `cancel_job`, `get_asset_file` (PNGs come back as images). Results list absolute paths under `assets_dir`, so agents on the same machine read files straight from disk.

## Using the output in Phaser

Each job writes `assets/phaser-loader.js` with the exact calls. Typical sprite job:

```js
preload() {
  this.load.atlas("hero", "assets/hero.png", "assets/hero.json");
  this.load.json("hero_anims", "assets/hero_anims.json");
}
create() {
  this.anims.fromJSON(this.cache.json.get("hero_anims"));
  this.add.sprite(100, 100, "hero").play("hero_walk");
}
```

Tiles: `this.load.image("grass_tiles", "grass_tiles.png")`, then `map.addTilesetImage("grass_tiles", "grass_tiles", tileWidth, tileHeight, margin, spacing)` with the margin and spacing from the snippet (tiles are extruded to prevent seams). `grass_tiles.tsj` imports directly into Tiled.

Small tiles are cut from one larger seamless texture ("Tiles per texture", auto by default: 4×4 for 32 px pixel tiles). Each block of tiles (`name_<variant>_r<row>c<col>`) repeats seamlessly as a unit, so paint it as a stamp or pattern in Tiled. Set "Tiles per texture" to 1×1 for single tiles that repeat on their own, with less detail.

## Configuration

`models.yaml` holds local model profiles, style presets (prompt fragments, LoRA, pixelation, palette size) and video profiles. Add a style there and it appears in the web app.

## Layout

```
backend/creative2d/
  api/        FastAPI app (jobs, SSE progress, settings, OpenRouter model catalog)
  jobs/       SQLite job store and single GPU worker
  pipeline/   prompt templates, backend factory, orchestration
  gen/        diffusers and OpenRouter backends, prompt enhancer, vision ranking
  post/       matting, pixel art, seamless tiles, isometric projection, trimming
  animate/    local image-to-video, frame sampling and stabilization
  pack/       MaxRects atlas, spritesheet, tile sheet, Phaser/Tiled export
web/          Vite + React + Phaser preview
tests/        pytest suite (runs without a GPU using fake backends)
```

Run the tests with `uv run pytest`.

## Notes and limits

- SDXL with the pixel-art LoRA sometimes draws a whole sheet of creatures. Each image goes through a single-subject check, and the job retries with new seeds (up to 3 times) when the check fails.
- Multi-direction characters reuse the seed across directions, and cloud models also get the first direction as a reference image. Consistency is best-effort.
- Local animation defaults to Wan 2.2 TI2V 5B, which takes about 3 minutes per clip on an M5 Pro (the first run downloads about 30 GB). LTX-Video is faster, but in testing it moved the prop and not the legs. On 48 GB machines Wan runs at 512 px, and its transformer and text encoder move to the CPU while the video decodes, to stay under the MPS memory limit.
- SDXL often draws characters in three-quarter view even when side view is requested. Cloud image models follow view instructions more reliably.
