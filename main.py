"""Didibood Price — sale + rent on one service."""
from __future__ import annotations

from app_factory import create_app
from config.settings import settings

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
