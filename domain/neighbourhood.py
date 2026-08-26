"""Neighbourhood string helpers."""
from __future__ import annotations

import re

_PERSIAN_RE = re.compile(r"[\u0600-\u06FF]")
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def is_persian_neighbourhood(value: str) -> bool:
    return bool(_PERSIAN_RE.search(value))


def is_slug_like(value: str) -> bool:
    return bool(_SLUG_RE.match(value.strip().lower()))
