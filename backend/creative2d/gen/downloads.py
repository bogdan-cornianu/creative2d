"""Explicit downloads of local models from Hugging Face.

Nothing downloads implicitly: loaders call `require()` and pass
`local_files_only=True`. The web app's Settings dialog queues downloads here.
A model counts as downloaded once a download finished for its exact set of
parts; that record lives in CONFIG_DIR/models.json.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .. import config

log = logging.getLogger("creative2d.downloads")

# Patterns for plain repo snapshots (VAE, BiRefNet): skip duplicate .bin/.onnx weights.
SNAPSHOT_PATTERNS = ["*.json", "*.py", "*.txt", "*.safetensors"]


@dataclass(frozen=True)
class Part:
    kind: str  # "pipeline" (diffusers pipeline repo), "snapshot" or "file"
    repo: str
    variant: str | None = None
    filename: str | None = None


@dataclass
class Entry:
    id: str
    label: str
    kind: str  # "image", "video" or "matting"
    parts: list[Part]
    size_gb: float | None = None
    default: bool = False
    gated: bool = False
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        self.signature = json.dumps([asdict(p) for p in self.parts], sort_keys=True)


class ModelNotDownloaded(RuntimeError):
    pass


def catalog() -> list[Entry]:
    cfg = config.load_models_config()
    styles = cfg.get("styles", {}) or {}
    entries: list[Entry] = []

    for name, p in (cfg.get("image_profiles") or {}).items():
        kind = p.get("pipeline")
        # Matches DiffusersImageBackend: fp16 weights for SDXL off the CPU.
        variant = "fp16" if kind == "sdxl" and config.detect_device() != "cpu" else None
        parts = [Part("pipeline", p["repo"], variant=variant)]
        if p.get("vae"):
            parts.append(Part("snapshot", p["vae"]))
        if kind == "sdxl":
            for style in styles.values():
                lora = (style or {}).get("lora")
                if lora:
                    part = Part("file", lora["repo"], filename=lora.get("weight_name"))
                    if part not in parts:
                        parts.append(part)
        entries.append(_entry(f"image:{name}", "image", p, name, parts, name == cfg.get("default_image_profile")))

    matting = cfg.get("matting") or {}
    entries.append(
        Entry(
            id="matting",
            label=matting.get("label", "BiRefNet background removal"),
            kind="matting",
            parts=[Part("snapshot", matting.get("repo", "ZhengPeng7/BiRefNet"))],
            size_gb=matting.get("size_gb"),
            default=True,
        )
    )

    for name, p in (cfg.get("video_profiles") or {}).items():
        parts = [Part("pipeline", p["repo"])]
        entries.append(_entry(f"video:{name}", "video", p, name, parts, name == cfg.get("default_video_profile")))
    return entries


def _entry(id: str, kind: str, p: dict, name: str, parts: list[Part], default: bool) -> Entry:
    return Entry(
        id=id,
        label=p.get("label", name),
        kind=kind,
        parts=parts,
        size_gb=p.get("size_gb"),
        default=default,
        gated=bool(p.get("gated")),
    )


def hf_fetch(part: Part) -> None:
    """Download one part into the Hugging Face cache (no-op when cached)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if part.kind == "pipeline":
        from diffusers import DiffusionPipeline

        # Fetches only the components listed in model_index.json, so the
        # single-file checkpoints at the root of SDXL/LTX repos are skipped.
        DiffusionPipeline.download(part.repo, variant=part.variant, use_safetensors=True)
    elif part.kind == "snapshot":
        snapshot_download(part.repo, allow_patterns=SNAPSHOT_PATTERNS)
    elif part.kind == "file":
        hf_hub_download(part.repo, part.filename)
    else:
        raise ValueError(f"unknown part kind {part.kind!r}")


def _describe_error(e: Exception, entry: Entry) -> str:
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status in (401, 403) or type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError"):
        return (
            f"Hugging Face refused access ({status or type(e).__name__}). "
            "Accept the model licence on huggingface.co, then run `uv run huggingface-cli login`."
        )
    return str(e) or type(e).__name__


def _cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def _cache_bytes() -> int:
    """Bytes of regular files in the HF cache, partial downloads included.

    Newer huggingface_hub keeps content in a shared store (hub/blobs/ee/<hash>)
    behind per-repo symlinks, so progress is measured as cache growth.
    """
    total = 0
    for f in _cache_dir().rglob("*"):
        try:
            if not f.is_symlink() and f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


class DownloadManager:
    """One background thread, one download at a time."""

    def __init__(self, fetch: Callable[[Part], None] | None = None) -> None:
        self._fetch = fetch or hf_fetch
        self._lock = threading.Lock()
        self._state: dict[str, dict] = {}  # id -> {"status": queued|downloading|error, "error": str}
        self._start_bytes = 0  # cache size when the current download started
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ records
    def _record_path(self) -> Path:
        return config.CONFIG_DIR / "models.json"

    def _records(self) -> dict:
        path = self._record_path()
        try:
            return json.loads(path.read_text()) if path.exists() else {}
        except ValueError:
            return {}

    def _mark_ready(self, entry: Entry) -> None:
        records = self._records()
        records[entry.id] = {"signature": entry.signature, "at": time.time()}
        config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._record_path().write_text(json.dumps(records, indent=1))

    # ------------------------------------------------------------ queries
    def _find(self, entry_id: str) -> Entry | None:
        return next((e for e in catalog() if e.id == entry_id), None)

    def is_ready(self, entry: Entry) -> bool:
        rec = self._records().get(entry.id)
        return bool(rec) and rec.get("signature") == entry.signature

    def list(self) -> list[dict]:
        records = self._records()
        out = []
        with self._lock:
            state = {k: dict(v) for k, v in self._state.items()}
        for e in catalog():
            s = state.get(e.id)
            ready = records.get(e.id, {}).get("signature") == e.signature
            status = s["status"] if s else ("ready" if ready else "missing")
            out.append(
                {
                    "id": e.id,
                    "label": e.label,
                    "kind": e.kind,
                    "size_gb": e.size_gb,
                    "default": e.default,
                    "gated": e.gated,
                    "status": status,
                    "error": s.get("error") if s else None,
                    "downloaded_bytes": max(0, _cache_bytes() - self._start_bytes) if status == "downloading" else None,
                }
            )
        return out

    def require(self, entry_id: str) -> None:
        entry = self._find(entry_id)
        if entry is None:
            raise ValueError(f"unknown local model {entry_id!r}")
        if not self.is_ready(entry):
            raise ModelNotDownloaded(
                f"{entry.label} is not downloaded. Open Settings → Local models and download it."
            )

    # ------------------------------------------------------------ downloads
    def enqueue(self, entry_ids: list[str]) -> None:
        known = {e.id for e in catalog()}
        unknown = [i for i in entry_ids if i not in known]
        if unknown:
            raise KeyError(unknown[0])
        with self._lock:
            for i in entry_ids:
                if self._state.get(i, {}).get("status") in ("queued", "downloading"):
                    continue
                self._state[i] = {"status": "queued", "error": None}
                self._queue.put(i)
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="creative2d-downloads", daemon=True)
                self._thread.start()

    def _loop(self) -> None:
        while True:
            entry_id = self._queue.get()
            entry = self._find(entry_id)
            if entry is None:
                with self._lock:
                    self._state.pop(entry_id, None)
                continue
            self._start_bytes = _cache_bytes()
            with self._lock:
                self._state[entry_id] = {"status": "downloading", "error": None}
            try:
                for part in entry.parts:
                    log.info("downloading %s: %s", entry_id, part.repo)
                    self._fetch(part)
                self._mark_ready(entry)
                with self._lock:
                    self._state.pop(entry_id, None)
            except Exception as e:
                log.error("download %s failed: %s", entry_id, e)
                with self._lock:
                    self._state[entry_id] = {"status": "error", "error": _describe_error(e, entry)}


_manager = DownloadManager()


def manager() -> DownloadManager:
    return _manager


def install(m: DownloadManager) -> None:
    global _manager
    _manager = m


def require(entry_id: str) -> None:
    _manager.require(entry_id)


def is_ready(entry_id: str) -> bool:
    entry = _manager._find(entry_id)
    return entry is not None and _manager.is_ready(entry)
