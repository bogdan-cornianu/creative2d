"""Single background worker: jobs run one at a time on the GPU."""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from typing import Callable

from .. import config
from ..gen.base import Cancelled
from ..pipeline.runner import Backends, JobContext, run
from ..spec import JobSpec
from .store import FINAL, JobStore

log = logging.getLogger("creative2d.worker")

BackendFactory = Callable[[JobSpec, Callable[[], bool]], tuple[Backends, object]]


class EventHub:
    """Fan-out of job events to SSE subscribers."""

    def __init__(self) -> None:
        self._subs: dict[str, list[queue.Queue]] = {}
        self._lock = threading.Lock()

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subs.get(job_id, [])
            if q in subs:
                subs.remove(q)

    def publish(self, job_id: str, event: dict) -> None:
        with self._lock:
            subs = list(self._subs.get(job_id, [])) + list(self._subs.get("*", []))
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


class Worker:
    def __init__(self, store: JobStore, hub: EventHub, factory: BackendFactory) -> None:
        self.store = store
        self.hub = hub
        self.factory = factory
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancelled: set[str] = set()
        self._current: str | None = None
        self._thread = threading.Thread(target=self._loop, name="creative2d-worker", daemon=True)
        self._thread.start()

    def submit(self, spec: JobSpec) -> str:
        job_id = self.store.create(spec.model_dump())
        self._queue.put(job_id)
        self._publish(job_id, status="queued", progress=0.0, stage="", message="queued")
        return job_id

    def cancel(self, job_id: str) -> bool:
        job = self.store.get(job_id)
        if not job or job["status"] in FINAL:
            return False
        self._cancelled.add(job_id)
        if job["status"] == "queued":
            self.store.update(job_id, status="cancelled", message="cancelled")
            self._publish(job_id, status="cancelled", message="cancelled")
        return True

    @property
    def current(self) -> str | None:
        return self._current

    def _publish(self, job_id: str, **event) -> None:
        self.hub.publish(job_id, {"id": job_id, **event})

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id in self._cancelled:
                continue
            self._current = job_id
            try:
                self._run(job_id)
            finally:
                self._current = None
                self._cancelled.discard(job_id)

    def _run(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if not job:
            return
        spec = JobSpec.model_validate(job["spec"])
        self.store.update(job_id, status="running", message="starting")
        self._publish(job_id, status="running", progress=0.0, stage="start", message="starting")
        last = [0.0]

        def is_cancelled() -> bool:
            return job_id in self._cancelled

        def emit(frac: float, stage: str, msg: str) -> None:
            self._publish(job_id, status="running", progress=round(frac, 4), stage=stage, message=msg)
            now = time.time()
            if now - last[0] > 1.0:  # throttle DB writes
                last[0] = now
                self.store.update(job_id, progress=frac, stage=stage, message=msg)

        client = None
        try:
            backends, client = self.factory(spec, is_cancelled)
            ctx = JobContext(job_id, spec, config.OUTPUT_DIR / job_id, emit, is_cancelled)
            manifest = run(ctx, backends)
            self.store.update(job_id, status="done", progress=1.0, stage="done", message="done", result=manifest)
            self._publish(job_id, status="done", progress=1.0, stage="done", message="done")
        except Cancelled:
            self.store.update(job_id, status="cancelled", message="cancelled")
            self._publish(job_id, status="cancelled", message="cancelled")
        except Exception as e:
            log.error("job %s failed: %s\n%s", job_id, e, traceback.format_exc())
            self.store.update(job_id, status="failed", error=str(e), message="failed")
            self._publish(job_id, status="failed", message=str(e))
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()
