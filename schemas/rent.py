"""Rent-specific API schemas."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PropertyType = Literal["apartment"]


class RentPredictRequest(BaseModel):
    city_slug: str = Field(..., examples=["tehran"])
    property_type: PropertyType = "apartment"
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


class RentComparablesOut(BaseModel):
    mean_equivalent_deposit_toman: int | None
    min_equivalent_deposit_toman: int | None
    max_equivalent_deposit_toman: int | None
    median_equivalent_deposit_toman: int | None
    sample_size: int
    filters_applied: dict[str, Any] = Field(default_factory=dict)


class RentPredictedRangeOut(BaseModel):
    low_equivalent_deposit_toman: int
    high_equivalent_deposit_toman: int
    margin_method: str
    margin_value: float


class RentPredictResponse(BaseModel):
    model_key: str
    model_trained_at: str | None
    predicted_equivalent_deposit_toman: int
    predicted_equivalent_deposit_range: RentPredictedRangeOut
    suggested_deposit_toman: int
    suggested_monthly_rent_toman: int
    comparables: RentComparablesOut
    training_metrics: dict[str, Any] = Field(default_factory=dict)


class RentTrainRequest(BaseModel):
    city_slug: str
    property_type: PropertyType = "apartment"
    refresh_data: bool = False


class RentTrainResponse(BaseModel):
    model_key: str
    n_rows: int
    n_train: int
    n_test: int
    mae: float
    mape: float
    model_path: str
    trained_at: str
