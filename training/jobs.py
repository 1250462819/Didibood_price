"""Training runs in the background, one at a time.

A training parses tens of thousands of listings and fits a model; on this
server that is a minute or more of both cores. It must not hold an HTTP
request open, and two at once would starve the API that serves estimates.
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from domain.model_key import ModelKey
from training.versioned import train_version

logger = logging.getLogger(__name__)

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()
#: Enough history for the panel to show what happened recently.
_MAX_JOBS = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_jobs() -> list[dict[str, Any]]:
    with _JOBS_LOCK:
        return sorted(_JOBS.values(), key=lambda job: job["created_at"], reverse=True)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def running_job() -> dict[str, Any] | None:
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if job["status"] in ("queued", "running"):
                return dict(job)
    return None


def _update(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(fields)


def start_training(key: ModelKey, *, window_days: int | None) -> dict[str, Any]:
    """Queue a training. Refuses while another one is queued or running."""
    busy = running_job()
    if busy:
        raise RuntimeError(f"A training is already in progress ({busy['model_key']})")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "model_key": key.slug(),
        "city_slug": key.city_slug,
        "purpose": key.purpose,
        "window_days": window_days,
        "status": "queued",
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "version": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        if len(_JOBS) > _MAX_JOBS:
            oldest = sorted(_JOBS.values(), key=lambda item: item["created_at"])
            for stale in oldest[: len(_JOBS) - _MAX_JOBS]:
                _JOBS.pop(stale["id"], None)

    def _run() -> None:
        with _RUN_LOCK:
            _update(job_id, status="running", started_at=_now())
            try:
                meta = train_version(key, window_days=window_days)
                _update(
                    job_id,
                    status="succeeded",
                    finished_at=_now(),
                    version=meta["version"],
                )
            except Exception as exc:  # the panel shows the reason; the log keeps the trace
                logger.exception("training failed key=%s", key.slug())
                _update(job_id, status="failed", finished_at=_now(), error=str(exc)[:500])

    threading.Thread(target=_run, name=f"train-{job_id}", daemon=True).start()
    return dict(job)
