"""Server-side settings: OpenRouter API key and default model choices.

The key lives in CONFIG_DIR/secrets.env (0600) and is never returned by the
API; only a masked form is exposed.
"""

from __future__ import annotations

import json
import os
import threading

from . import config

_lock = threading.Lock()
KEY_ENV = "OPENROUTER_API_KEY"


def _secrets_path():
    return config.CONFIG_DIR / "secrets.env"


def _prefs_path():
    return config.CONFIG_DIR / "settings.json"


def get_api_key() -> str | None:
    path = _secrets_path()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(f"{KEY_ENV}="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    return os.environ.get(KEY_ENV) or None


def set_api_key(key: str | None) -> None:
    with _lock:
        config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path = _secrets_path()
        if not key:
            path.unlink(missing_ok=True)
            return
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"{KEY_ENV}={key.strip()}\n")
        os.chmod(path, 0o600)


def mask(key: str | None) -> str | None:
    if not key:
        return None
    return key[:6] + "…" + key[-4:] if len(key) > 12 else "…" * 3


def get_prefs() -> dict:
    path = _prefs_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except ValueError:
            return {}
    return {}


def set_prefs(prefs: dict) -> dict:
    allowed = {"image_model", "text_model", "vision_model", "video_model"}
    with _lock:
        current = get_prefs()
        current.update({k: v for k, v in prefs.items() if k in allowed})
        config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _prefs_path().write_text(json.dumps(current, indent=1))
    return current


def public_settings() -> dict:
    key = get_api_key()
    return {
        "openrouter_key_set": bool(key),
        "openrouter_key_masked": mask(key),
        "openrouter_key_source": "file" if _secrets_path().exists() else ("env" if key else None),
        "prefs": get_prefs(),
    }
