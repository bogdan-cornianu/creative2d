"""Command line: `creative2d serve` or `creative2d gen "prompt" [options]`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import config


def _gen(args: argparse.Namespace) -> int:
    from .pipeline.factory import build_backends
    from .pipeline.runner import JobContext, run
    from .spec import AnimateOptions, JobSpec

    spec = JobSpec(
        prompt=args.prompt,
        name=args.name,
        asset_type=args.type,
        style=args.style,
        view=args.view,
        frame_size=args.size,
        tile_size=args.size,
        variants=args.variants,
        directions=args.directions,
        seed=args.seed,
        backend=args.backend,
        image_model=args.model,
        animate=AnimateOptions(
            enabled=bool(args.animate),
            action=args.animate or "idle",
            frames=args.frames,
            backend=args.video_backend,
            model=args.video_model,
        ),
    )
    backends, client = build_backends(spec, lambda: False)
    job_id = time.strftime("cli-%Y%m%d-%H%M%S")
    last = [""]

    def emit(frac: float, stage: str, msg: str) -> None:
        line = f"[{frac * 100:5.1f}%] {stage}: {msg}"
        if line != last[0]:
            print(line, flush=True)
            last[0] = line

    try:
        manifest = run(JobContext(job_id, spec, config.OUTPUT_DIR / job_id, emit), backends)
    finally:
        if client:
            client.close()
    print(json.dumps({k: manifest[k] for k in ("job_id", "seed", "cost_usd", "seconds")}, indent=1))
    print(f"output: {config.OUTPUT_DIR / job_id}")
    return 0


def _drop_unwritable_keylog() -> None:
    """ssl.create_default_context() raises if SSLKEYLOGFILE can not be opened, breaking every HTTPS call."""
    path = os.environ.get("SSLKEYLOGFILE")
    if not path:
        return
    try:
        with open(path, "a"):
            pass
    except OSError as e:
        del os.environ["SSLKEYLOGFILE"]
        print(f"warning: ignoring SSLKEYLOGFILE ({e})", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    _drop_unwritable_keylog()
    p = argparse.ArgumentParser(prog="creative2d")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    g = sub.add_parser("gen", help="generate an asset from the command line")
    g.add_argument("prompt")
    g.add_argument("--name", default="asset")
    g.add_argument("--type", default="character", choices=["character", "prop", "tile", "background"])
    g.add_argument("--style", default="pixel")
    g.add_argument("--view", default="side", choices=["side", "topdown", "iso"])
    g.add_argument("--size", type=int, default=64, help="frame size (sprites) or tile size (tiles)")
    g.add_argument("--variants", type=int, default=1)
    g.add_argument("--directions", type=int, default=1, choices=[1, 4, 8])
    g.add_argument("--seed", type=int)
    g.add_argument("--backend", default="local", choices=["local", "openrouter"])
    g.add_argument("--model", help="local profile name or OpenRouter model id")
    g.add_argument("--animate", choices=["idle", "walk", "run", "attack", "jump"])
    g.add_argument("--frames", type=int, default=8)
    g.add_argument("--video-backend", default="local", choices=["local", "openrouter"])
    g.add_argument("--video-model")

    args = p.parse_args(argv)
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("creative2d.api.app:create_app", factory=True, host=args.host, port=args.port)
        return 0
    return _gen(args)


if __name__ == "__main__":
    sys.exit(main())
