"""API schemas for the neighbourhood analysis page.

Prices are Toman throughout, as everywhere else in this service. `metric` on the
envelope says what the bare `median` fields mean: price per square metre for
sale, deposit-equivalent for rent.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.price import PropertyType, Purpose


class MonthlyPoint(BaseModel):
    period: str = Field(..., description="YYYY-MM of market entry (Tehran calendar)")
    median: float | None = Field(None, description="None when the month is too thin to publish")
    sample_size: int


class NeighbourhoodRow(BaseModel):
    title: str
    sample_size: int
    low_sample: bool
    median: float | None
    p25: float | None
    p75: float | None
    median_budget_toman: float | None = Field(
        None, description="Sale: asking price. Rent: deposit-equivalent."
    )
    median_area: float | None
    median_rooms: int | None
    median_building_age: int | None
    lat: float | None
    lon: float | None
    trend_pct: float | None
    rank: int
    percentile: float | None


class CitySummary(BaseModel):
    sample_size: int
    neighbourhood_count: int
    median: float | None
    p25: float | None
    p75: float | None
    median_budget_toman: float | None
    median_area: float | None
    trend_pct: float | None
    monthly: list[MonthlyPoint]


class AnalyticsMeta(BaseModel):
    city: str
    purpose: Purpose
    property_type: PropertyType
    months: int
    metric: str
    filters: dict[str, object]
    data_built_at: str
    coverage_start: str | None
    total_listings: int


class NeighbourhoodOverviewResponse(AnalyticsMeta):
    summary: CitySummary
    neighbourhoods: list[NeighbourhoodRow]


class BandRow(BaseModel):
    label: str
    sample_size: int
    median: float | None
    rooms: int | None = None


class AmenityRow(BaseModel):
    key: str
    label: str
    with_sample: int
    without_sample: int
    median_with: float | None
    median_without: float | None
    premium_pct: float | None


class HistogramBucket(BaseModel):
    from_: float | None = Field(None, alias="from")
    to: float | None
    count: int

    model_config = {"populate_by_name": True}


class RentReturn(BaseModel):
    sample_size: int
    monthly_rent_per_sqm_toman: float | None
    median_deposit_toman: float | None
    median_monthly_rent_toman: float | None
    gross_yield_pct: float | None


class RelatedNeighbourhood(BaseModel):
    title: str
    median: float | None
    sample_size: int
    diff_pct: float | None
    rank: int | None = None
    distance_km: float | None = None


class NeighbourhoodPrice(BaseModel):
    median: float | None
    p10: float | None
    p25: float | None
    p75: float | None
    p90: float | None
    median_budget_toman: float | None
    median_area: float | None


class NeighbourhoodDetail(BaseModel):
    title: str
    sample_size: int
    low_sample: bool
    price: NeighbourhoodPrice
    rank: int | None
    percentile: float | None
    trend_pct: float | None
    monthly: list[MonthlyPoint]
    histogram: list[HistogramBucket]
    by_rooms: list[BandRow]
    by_area: list[BandRow]
    by_age: list[BandRow]
    amenities: list[AmenityRow]
    rent: RentReturn | None
    similar: list[RelatedNeighbourhood]
    nearby: list[RelatedNeighbourhood]


class NeighbourhoodDetailResponse(AnalyticsMeta):
    neighbourhood: NeighbourhoodDetail


class NeighbourhoodTitlesResponse(BaseModel):
    city: str
    purpose: Purpose
    property_type: PropertyType
    titles: list[str]
