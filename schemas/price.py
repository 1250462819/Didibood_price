"""API schemas — Didibood property_details aligned."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Purpose = Literal["sale", "rent"]
PropertyType = Literal["apartment"]


class PredictRequest(BaseModel):
    city_slug: str = Field(..., examples=["tehran"])
    property_type: PropertyType = "apartment"
    purpose: Purpose = "sale"
    neighbourhood: str | None = Field(None, description="Didibood neighbourhood (محله)")
    area: float = Field(..., gt=0, description="m²")
    rooms: int | None = Field(None, ge=0)
    year_built: int | None = Field(None, ge=1300, le=1500, description="Jalali year")
    floor_number: int | None = None
    total_floors: int | None = Field(None, ge=0)
    has_parking: bool | None = None
    has_elevator: bool | None = None
    has_storage: bool | None = None
    has_balcony: bool | None = None
    number_of_bathrooms: int | None = Field(None, ge=0)
    location_lat: float | None = Field(None, ge=-90, le=90)
    location_long: float | None = Field(None, ge=-180, le=180)
    deposit_amount_toman: int | None = Field(
        None,
        ge=0,
        description="Rent only: preferred deposit on conversion line",
    )
    monthly_rent_toman: int | None = Field(
        None,
        ge=0,
        description="Rent only: preferred monthly rent on conversion line",
    )


class ComparablesOut(BaseModel):
    mean_price_per_sqm_toman: int | None = None
    min_price_per_sqm_toman: int | None = None
    max_price_per_sqm_toman: int | None = None
    median_price_per_sqm_toman: int | None = None
    q1_price_per_sqm_toman: int | None = None
    q3_price_per_sqm_toman: int | None = None
    mean_equivalent_deposit_toman: int | None = None
    min_equivalent_deposit_toman: int | None = None
    max_equivalent_deposit_toman: int | None = None
    median_equivalent_deposit_toman: int | None = None
    sample_size: int
    filters_applied: dict[str, Any] = Field(default_factory=dict)


class SalePredictedRangeOut(BaseModel):
    low_price_per_sqm_toman: int
    high_price_per_sqm_toman: int
    low_total_price_toman: int | None
    high_total_price_toman: int | None
    margin_method: str
    margin_value: float


class RentPredictedRangeOut(BaseModel):
    low_equivalent_deposit_toman: int
    high_equivalent_deposit_toman: int
    margin_method: str
    margin_value: float


class PredictResponse(BaseModel):
    model_key: str
    purpose: Purpose
    model_trained_at: str | None
    predicted_price_per_sqm_toman: int | None = None
    predicted_total_price_toman: int | None = None
    predicted_range: SalePredictedRangeOut | None = None
    predicted_equivalent_deposit_toman: int | None = None
    predicted_equivalent_deposit_range: RentPredictedRangeOut | None = None
    suggested_deposit_toman: int | None = None
    suggested_monthly_rent_toman: int | None = None
    comparables: ComparablesOut
    training_metrics: dict[str, Any] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    city_slug: str
    property_type: PropertyType = "apartment"
    purpose: Purpose = "sale"
    refresh_data: bool = False


class TrainResponse(BaseModel):
    model_key: str
    n_rows: int
    n_train: int
    n_test: int
    mae: float
    mape: float
    model_path: str
    trained_at: str


class MarketStatsQuery(BaseModel):
    city_slug: str = Field(..., examples=["tehran"])
    purpose: Purpose = "sale"
    property_type: PropertyType = "apartment"
    neighbourhood: str | None = Field(None, max_length=128)
    months: int = Field(6, ge=3, le=12)


class MarketStatsPoint(BaseModel):
    period: str = Field(..., description="Gregorian YYYY-MM calendar month")
    avg_price_per_sqm_toman: int | None = None
    sample_size: int


class MarketStatsResponse(BaseModel):
    city_slug: str
    purpose: Purpose
    property_type: PropertyType
    neighbourhood: str | None = None
    months: int
    points: list[MarketStatsPoint] = Field(default_factory=list)
    sample_size: int = 0
    coverage_start: str | None = Field(
        None,
        description="Earliest YYYY-MM in the window that actually has listings",
    )
    covered_months: int = Field(
        0,
        description=(
            "Months in the window with listings. Lets callers tell a flat market "
            "apart from a crawl history too short to answer the question."
        ),
    )
