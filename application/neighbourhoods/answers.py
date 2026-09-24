"""Finished answers on disk, so the page does not recompute them per visit.

The neighbourhood endpoints aggregate tens of thousands of listings per request
— about a second and a half for one city's overview, and the page asks for
several as a visitor moves the sliders. None of it changes during the day: the
crawl lands overnight, so the same request returns the same numbers until the
next crawl.

So the answers are computed once, at night, and written here as the exact
payload the endpoint returns. A request that matches a stored answer is a file
read; one that does not is computed as before, which keeps every filter working
whether or not it was precomputed.

Staleness is a safety net, not a schedule: if the nightly job stops running, the
answers are ignored after `NEIGHBOURHOOD_ANSWER_MAX_AGE_HOURS` and the page
falls back to live figures rather than serving last week's market forever.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

#: Bump when a payload's shape changes, so yesterday's answers are not served
#: to today's code.
SCHEMA_VERSION = 2


def answers_dir() -> Path:
    return settings.DATA_CACHE_DIR / "answers"


@dataclass(frozen=True)
class AnswerKey:
    """What a stored answer is an answer *to*."""

    endpoint: str
    city: str
    purpose: str
    property_type: str
    filters: dict[str, Any]
    neighbourhood: str | None = None

    def digest(self) -> str:
        body = json.dumps(
            {
                "v": SCHEMA_VERSION,
                "endpoint": self.endpoint,
                "city": self.city,
                "purpose": self.purpose,
                "property_type": self.property_type,
                # Sorted, so two equal filter sets cannot produce two files.
                "filters": {key: self.filters[key] for key in sorted(self.filters)},
                "neighbourhood": self.neighbourhood,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(body.encode("utf-8")).hexdigest()

    def path(self) -> Path:
        return answers_dir() / f"{self.endpoint}__{self.city}__{self.digest()}.json.gz"


def _max_age() -> timedelta:
    return timedelta(hours=settings.NEIGHBOURHOOD_ANSWER_MAX_AGE_HOURS)


def store(key: AnswerKey, payload: dict[str, Any]) -> Path:
    """Write one answer. Written beside the target and moved into place: a
    reader must never see half a file, and the job runs while requests are served."""
    path = key.path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as handle:
        handle.write(body)
    tmp.replace(path)
    return path


def load(key: AnswerKey) -> dict[str, Any] | None:
    """The stored answer, or None when there is none, it is too old, or it is unreadable."""
    path = key.path()
    try:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None
    if datetime.now(tz=timezone.utc) - stamp > _max_age():
        logger.info("answer expired endpoint=%s city=%s", key.endpoint, key.city)
        return None
    try:
        with gzip.open(path, "rb") as handle:
            return json.loads(handle.read().decode("utf-8"))
    except Exception:
        logger.warning("answer unreadable path=%s", path, exc_info=True)
        return None


def built_at() -> datetime | None:
    """When the newest stored answer was written — what /health reports."""
    try:
        stamps = [entry.stat().st_mtime for entry in answers_dir().glob("*.json.gz")]
    except OSError:
        return None
    return datetime.fromtimestamp(max(stamps), tz=timezone.utc) if stamps else None


def count() -> int:
    try:
        return sum(1 for _ in answers_dir().glob("*.json.gz"))
    except OSError:
        return 0
