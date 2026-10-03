"""Keeps at most a few heavy pipelines in memory; evicts least recently used.

Unified memory on Apple Silicon is shared with the OS, so by default only
one large pipeline (image or video) stays loaded, plus the small matting model.
"""

from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from typing import Any, Callable

_lock = threading.RLock()
_heavy: "OrderedDict[str, Any]" = OrderedDict()
_light: dict[str, Any] = {}
MAX_HEAVY = 1


def _free_device_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get(key: str, loader: Callable[[], Any], heavy: bool = True) -> Any:
    with _lock:
        store = _heavy if heavy else _light
        if key in store:
            if heavy:
                _heavy.move_to_end(key)
            return store[key]
        if heavy:
            while len(_heavy) >= MAX_HEAVY:
                _heavy.popitem(last=False)
                _free_device_memory()
        obj = loader()
        store[key] = obj
        return obj


def clear() -> None:
    with _lock:
        _heavy.clear()
        _light.clear()
        _free_device_memory()


def loaded() -> list[str]:
    with _lock:
        return list(_heavy) + list(_light)
