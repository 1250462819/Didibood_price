"""API schemas for the neighbourhood analysis page.

Prices are Toman throughout, as everywhere else in this service. `metric` on the
envelope says what the bare `median` fields mean: price per square metre for
sale, deposit-equivalent for rent.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.price import PropertyType, Purpose


class AmenitySplit(BaseModel):
    """Listings with an amenity against those without, on the trimmed mean.

    `premium_pct` is withheld below five listings on either side."""

    with_share: float | None = None
    with_sample: int = 0
    without_sample: int = 0
    mean_with: float | None = None
    mean_without: float | None = None
    premium_pct: float | None = None


class CityAmenitySplit(AmenitySplit):
    like_for_like_pct: float | None = Field(
        None,
        description="Listing-weighted average of the within-neighbourhood premiums — "
        "what the amenity adds inside one neighbourhood, not where it is common.",
    )


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
    mean: float | None = Field(None, description="10–90% trimmed mean of the metric")
    mean_area: float | None = None
    mean_budget_toman: float | None = None
    parking: AmenitySplit | None = None
    elevator: AmenitySplit | None = None


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
    mean: float | None = Field(None, description="10–90% trimmed mean of the metric")
    mean_area: float | None = None
    mean_budget_toman: float | None = None
    parking: CityAmenitySplit | None = None
    elevator: CityAmenitySplit | None = None


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
    mean: float | None = None


class AmenityRow(BaseModel):
    key: str
    label: str
    with_sample: int
    without_sample: int
    median_with: float | None
    median_without: float | None
    premium_pct: float | None
    mean_with: float | None = None
    mean_without: float | None = None
    mean_premium_pct: float | None = None


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
    mean: float | None = Field(None, description="10–90% trimmed mean of the metric")
    mean_area: float | None = None
    mean_budget_toman: float | None = None


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


class CityRow(BaseModel):
    city: str
    label: str
    sample_size: int
    neighbourhood_count: int
    median: float | None
    mean: float | None = None
    p25: float | None
    p75: float | None
    median_budget_toman: float | None
    trend_pct: float | None
    lat: float | None
    lon: float | None


class CitiesTotal(CitySummary):
    """Every covered city pooled — the headline above a country-wide map."""

    city_count: int
    spread_ratio: float | None
    dearest_city: str | None
    cheapest_city: str | None


class NeighbourhoodCitiesResponse(BaseModel):
    """The map's first level: every city the crawl covers, dearest first."""

    purpose: Purpose
    property_type: PropertyType
    months: int
    metric: str
    cities: list[CityRow]
    total: CitiesTotal | None = None
