"""Neighbourhood analysis routes — the data behind /neighborhoods on the site."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from application.neighbourhoods.filters import NeighbourhoodFilters
from application.neighbourhoods.read import (
    get_cities,
    get_detail,
    get_overview,
    get_titles,
)
from config.settings import settings
from domain.model_key import SUPPORTED_CITIES, ModelKey
from schemas.neighbourhoods import (
    NeighbourhoodCitiesResponse,
    NeighbourhoodDetailResponse,
    NeighbourhoodOverviewResponse,
    NeighbourhoodTitlesResponse,
)
from schemas.price import PropertyType, Purpose

router = APIRouter(prefix="/neighborhoods", tags=["Neighbourhoods"])


def _key(city: str, purpose: str, property_type: str) -> ModelKey:
    slug = city.strip().lower()
    if slug not in SUPPORTED_CITIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported city; allowed: {', '.join(SUPPORTED_CITIES)}",
        )
    return ModelKey(city_slug=slug, property_type=property_type, purpose=purpose)


def _filters(
    months: int,
    min_area: float | None,
    max_area: float | None,
    min_rooms: int | None,
    max_rooms: int | None,
    max_building_age: int | None,
    min_budget: float | None,
    max_budget: float | None,
    has_parking: bool | None,
    has_elevator: bool | None,
    has_storage: bool | None,
    has_balcony: bool | None,
) -> NeighbourhoodFilters:
    return NeighbourhoodFilters(
        months=months,
        min_area=min_area,
        max_area=max_area,
        min_rooms=min_rooms,
        max_rooms=max_rooms,
        max_building_age=max_building_age,
        min_budget_toman=min_budget,
        max_budget_toman=max_budget,
        has_parking=has_parking,
        has_elevator=has_elevator,
        has_storage=has_storage,
        has_balcony=has_balcony,
    )


@router.get("", response_model=NeighbourhoodOverviewResponse)
def neighbourhood_overview(
    city: str = Query("tehran", description="City slug"),
    purpose: Purpose = Query("sale"),
    property_type: PropertyType = Query("apartment"),
    months: int = Query(6, ge=1, le=24),
    min_area: float | None = Query(None, ge=0),
    max_area: float | None = Query(None, ge=0),
    min_rooms: int | None = Query(None, ge=0, le=10),
    max_rooms: int | None = Query(None, ge=0, le=10),
    max_building_age: int | None = Query(None, ge=0, le=100),
    min_budget: float | None = Query(None, ge=0, description="Toman"),
    max_budget: float | None = Query(None, ge=0, description="Toman"),
    has_parking: bool | None = Query(None),
    has_elevator: bool | None = Query(None),
    has_storage: bool | None = Query(None),
    has_balcony: bool | None = Query(None),
) -> NeighbourhoodOverviewResponse:
    """City headline plus one row per neighbourhood: ranking table and map values."""
    months = min(months, settings.NEIGHBOURHOOD_MAX_MONTHS)
    payload = get_overview(
        _key(city, purpose, property_type),
        _filters(
            months,
            min_area,
            max_area,
            min_rooms,
            max_rooms,
            max_building_age,
            min_budget,
            max_budget,
            has_parking,
            has_elevator,
            has_storage,
            has_balcony,
        ),
    )
    return NeighbourhoodOverviewResponse(**payload)


@router.get("/detail", response_model=NeighbourhoodDetailResponse)
def neighbourhood_detail(
    neighbourhood: str = Query(..., min_length=1, max_length=128),
    city: str = Query("tehran"),
    purpose: Purpose = Query("sale"),
    property_type: PropertyType = Query("apartment"),
    months: int = Query(6, ge=1, le=24),
    min_area: float | None = Query(None, ge=0),
    max_area: float | None = Query(None, ge=0),
    min_rooms: int | None = Query(None, ge=0, le=10),
    max_rooms: int | None = Query(None, ge=0, le=10),
    max_building_age: int | None = Query(None, ge=0, le=100),
    min_budget: float | None = Query(None, ge=0),
    max_budget: float | None = Query(None, ge=0),
    has_parking: bool | None = Query(None),
    has_elevator: bool | None = Query(None),
    has_storage: bool | None = Query(None),
    has_balcony: bool | None = Query(None),
) -> NeighbourhoodDetailResponse:
    """Everything known about one neighbourhood under the same filters."""
    months = min(months, settings.NEIGHBOURHOOD_MAX_MONTHS)
    payload = get_detail(
        _key(city, purpose, property_type),
        _filters(
            months,
            min_area,
            max_area,
            min_rooms,
            max_rooms,
            max_building_age,
            min_budget,
            max_budget,
            has_parking,
            has_elevator,
            has_storage,
            has_balcony,
        ),
        neighbourhood=neighbourhood,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="محله‌ای با این نام در این فیلترها پیدا نشد.")
    return NeighbourhoodDetailResponse(**payload)


@router.get("/cities", response_model=NeighbourhoodCitiesResponse)
def neighbourhood_cities(
    purpose: Purpose = Query("sale"),
    property_type: PropertyType = Query("apartment"),
    months: int = Query(6, ge=1, le=24),
    min_area: float | None = Query(None, ge=0),
    max_area: float | None = Query(None, ge=0),
    min_rooms: int | None = Query(None, ge=0, le=10),
    max_rooms: int | None = Query(None, ge=0, le=10),
    max_building_age: int | None = Query(None, ge=0, le=100),
    min_budget: float | None = Query(None, ge=0),
    max_budget: float | None = Query(None, ge=0),
    has_parking: bool | None = Query(None),
    has_elevator: bool | None = Query(None),
    has_storage: bool | None = Query(None),
    has_balcony: bool | None = Query(None),
) -> NeighbourhoodCitiesResponse:
    """Every covered city under the same filters — the map before a city is picked."""
    months = min(months, settings.NEIGHBOURHOOD_MAX_MONTHS)
    payload = get_cities(
        _filters(
            months,
            min_area,
            max_area,
            min_rooms,
            max_rooms,
            max_building_age,
            min_budget,
            max_budget,
            has_parking,
            has_elevator,
            has_storage,
            has_balcony,
        ),
        purpose=purpose,
        property_type=property_type,
    )
    return NeighbourhoodCitiesResponse(**payload)


@router.get("/titles", response_model=NeighbourhoodTitlesResponse)
def neighbourhood_titles(
    city: str = Query("tehran"),
    purpose: Purpose = Query("sale"),
    property_type: PropertyType = Query("apartment"),
) -> NeighbourhoodTitlesResponse:
    """Neighbourhood names with data, most listings first — for the search box."""
    key = _key(city, purpose, property_type)
    return NeighbourhoodTitlesResponse(
        city=key.city_slug,
        purpose=key.purpose,
        property_type=key.property_type,
        titles=get_titles(key),
    )
