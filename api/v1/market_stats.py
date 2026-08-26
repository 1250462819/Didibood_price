"""Market stats routes."""
from __future__ import annotations

from fastapi import APIRouter, Query

from application.market_stats.read import get_market_stats
from schemas.price import MarketStatsResponse, PropertyType, Purpose, MarketStatsQuery

router = APIRouter(prefix="/market-stats", tags=["Market Stats"])


@router.get("", response_model=MarketStatsResponse)
def market_stats(
    city: str = Query(..., min_length=1, max_length=128, description="City slug: tehran, mashhad, isfahan"),
    purpose: Purpose = Query("sale"),
    property_type: PropertyType = Query("apartment"),
    neighborhood: str | None = Query(None, max_length=128),
    months: int = Query(6, ge=3, le=12),
) -> MarketStatsResponse:
    """Return monthly average price/m² from Divar crawl data."""
    return get_market_stats(
        None,
        MarketStatsQuery(
            city_slug=city,
            purpose=purpose,
            property_type=property_type,
            neighbourhood=neighborhood,
            months=months,
        ),
    )
