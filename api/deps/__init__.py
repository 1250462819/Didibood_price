"""FastAPI dependencies."""
from __future__ import annotations

from db.session import get_db

__all__ = ["get_db"]
