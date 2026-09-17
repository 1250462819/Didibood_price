"""Single FastAPI app — sale + rent endpoints."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.market_stats import router as market_stats_router
from api.v1.neighborhoods import router as neighborhoods_router
from api.v1.rent.router import router as rent_router
from api.v1.sale.router import router as sale_router
from config.settings import settings
from schemas.common import HealthResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description=(
            "CatBoost property pricing: separate sale and rent predict/train endpoints."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> dict[str, str | dict[str, str]]:
        return {
            "service": settings.APP_NAME,
            "version": settings.VERSION,
            "docs": "/docs",
            "endpoints": {
                "sale_predict": "/api/v1/sale/predict",
                "sale_train": "/api/v1/sale/train",
                "rent_predict": "/api/v1/rent/predict",
                "rent_train": "/api/v1/rent/train",
                "market_stats": "/api/v1/market-stats",
                "neighborhoods": "/api/v1/neighborhoods",
            },
        }

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=settings.APP_NAME,
            version=settings.VERSION,
        )

    app.include_router(sale_router, prefix="/api/v1")
    app.include_router(rent_router, prefix="/api/v1")
    app.include_router(market_stats_router, prefix="/api/v1")
    app.include_router(neighborhoods_router, prefix="/api/v1")
    logging.basicConfig(level=logging.DEBUG if settings.DEBUG else logging.INFO)
    return app
