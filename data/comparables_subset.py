"""Shared comparables neighbourhood filtering (same rules as predict)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from data.neighbourhood_persian import (
    normalize_persian_neighbourhood,
    resolve_request_neighbourhood,
)


def known_neighbourhood_titles(dataset: pd.DataFrame) -> set[str]:
    if dataset.empty or "neighbourhood" not in dataset.columns:
        return set()
    return {
        str(value).strip()
        for value in dataset["neighbourhood"].dropna().unique()
        if str(value).strip() and str(value).strip().lower() != "unknown"
    }


def neighbourhood_mask(series: pd.Series, title: str) -> pd.Series:
    target = normalize_persian_neighbourhood(title)
    return series.astype(str).map(normalize_persian_neighbourhood) == target


def subset_comparables(
    dataset: pd.DataFrame,
    neighbourhood: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter a comparables extract to the same neighbourhood rules as comparable_stats."""
    neighbourhood = (neighbourhood or "").strip()
    neighbourhood_applied = False
    resolved_neighbourhood: str | None = None

    if dataset.empty:
        return dataset, _filters(
            neighbourhood=neighbourhood or None,
            resolved=resolved_neighbourhood,
            applied=neighbourhood_applied,
        )

    if neighbourhood:
        known = known_neighbourhood_titles(dataset)
        resolved = resolve_request_neighbourhood(neighbourhood, known) or neighbourhood
        resolved_neighbourhood = resolved
        subset = dataset.loc[neighbourhood_mask(dataset["neighbourhood"], resolved)]
        neighbourhood_applied = True
    else:
        subset = dataset

    return subset, _filters(
        neighbourhood=neighbourhood or None,
        resolved=resolved_neighbourhood,
        applied=neighbourhood_applied,
    )


EARTH_RADIUS_KM = 6371.0088
SIMILARITY_LEVELS = ("strict", "wide")


def select_similar_comparables(
    dataset: pd.DataFrame,
    *,
    neighbourhood: str | None,
    area: float | None,
    year_built: int | None,
    lat: float | None,
    lng: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pick the listings an estimate is compared against.

    Tiers, first with COMPARABLE_TIER_MIN_ROWS rows wins:
    neighbourhood strict → wide, nearby strict → wide, then neighbourhood and
    nearby with no size/age filter. Without a neighbourhood the city takes its
    place after the nearby tiers. A neighbourhood with no listings of its own
    is still covered by its surroundings when the request has a location.
    """
    hood_pool, base = subset_comparables(dataset, neighbourhood)
    hood_applied = bool(base["neighbourhood_applied"])

    pools: list[tuple[str, pd.DataFrame]] = []
    if hood_applied:
        pools.append(("neighbourhood", hood_pool))
    center = _search_center(lat, lng, hood_pool)
    if center is not None:
        nearby = _within_radius(dataset, center, settings.COMPARABLE_NEARBY_RADIUS_KM)
        if nearby is not None:
            pools.append(("nearby", nearby))
    if not hood_applied:
        pools.append(("city", dataset))

    year_cap = _max_year(dataset)
    tiers = [(scope, pool, level) for scope, pool in pools for level in SIMILARITY_LEVELS]
    tiers += [(scope, pool, "none") for scope, pool in pools]

    fallback: tuple[pd.DataFrame, dict[str, Any]] | None = None
    for scope, pool, level in tiers:
        subset, bounds = _similar(
            pool, level, area=area, year_built=year_built, year_cap=year_cap
        )
        if subset is None:
            continue
        if len(subset) >= settings.COMPARABLE_TIER_MIN_ROWS:
            return subset, _tier_filters(base, scope, level, bounds)
        # Nothing reaches the minimum: a thin unfiltered pool still beats an
        # empty section, as it did before tiers existed.
        if level == "none" and fallback is None and not subset.empty:
            fallback = subset, _tier_filters(base, scope, level, bounds)

    if fallback is not None:
        return fallback
    empty_scope = "neighbourhood" if hood_applied else "city"
    return dataset.iloc[0:0], _tier_filters(base, empty_scope, "none", {})


def _similar(
    pool: pd.DataFrame,
    level: str,
    *,
    area: float | None,
    year_built: int | None,
    year_cap: int | None,
) -> tuple[pd.DataFrame | None, dict[str, int]]:
    """Rows of pool within the level's size/age band; None when nothing can be matched on."""
    if level == "none":
        return pool, {}
    strict = level == "strict"
    area_tol = (
        settings.COMPARABLE_AREA_TOLERANCE_PCT
        if strict
        else settings.COMPARABLE_AREA_TOLERANCE_WIDE_PCT
    )
    year_tol = (
        settings.COMPARABLE_YEAR_TOLERANCE
        if strict
        else settings.COMPARABLE_YEAR_TOLERANCE_WIDE
    )

    mask = pd.Series(True, index=pool.index)
    bounds: dict[str, int] = {}
    if area and area > 0 and "area" in pool.columns:
        low, high = area * (1 - area_tol), area * (1 + area_tol)
        mask &= pd.to_numeric(pool["area"], errors="coerce").between(low, high)
        # Divar areas are whole metres: report the whole metres that pass.
        bounds["area_min"] = math.ceil(low)
        bounds["area_max"] = math.floor(high)
    if year_built and "year_built" in pool.columns:
        low, high = year_built - year_tol, year_built + year_tol
        # Listings with no build year cannot be shown to be the same age.
        mask &= pd.to_numeric(pool["year_built"], errors="coerce").between(low, high)
        bounds["year_min"] = int(low)
        # "Built 1399–1409" reads as a typo in 1405; cap at the newest listing.
        bounds["year_max"] = int(
            high if year_cap is None else max(year_built, min(high, year_cap))
        )
    if not bounds:
        return None, {}
    return pool.loc[mask], bounds


def _search_center(
    lat: float | None,
    lng: float | None,
    hood_pool: pd.DataFrame,
) -> tuple[float, float] | None:
    if lat is not None and lng is not None:
        return float(lat), float(lng)
    if hood_pool.empty or not {"location_lat", "location_long"} <= set(hood_pool.columns):
        return None
    coords = hood_pool[["location_lat", "location_long"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if coords.empty:
        return None
    return float(coords["location_lat"].median()), float(coords["location_long"].median())


def _within_radius(
    dataset: pd.DataFrame,
    center: tuple[float, float],
    radius_km: float,
) -> pd.DataFrame | None:
    if not {"location_lat", "location_long"} <= set(dataset.columns):
        return None
    lat = np.radians(pd.to_numeric(dataset["location_lat"], errors="coerce").to_numpy(dtype=float))
    lng = np.radians(pd.to_numeric(dataset["location_long"], errors="coerce").to_numpy(dtype=float))
    lat0, lng0 = np.radians(center[0]), np.radians(center[1])
    hav = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lng - lng0) / 2) ** 2
    distance_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(hav))
    return dataset.loc[distance_km <= radius_km]


def _max_year(dataset: pd.DataFrame) -> int | None:
    if "year_built" not in dataset.columns:
        return None
    newest = pd.to_numeric(dataset["year_built"], errors="coerce").max()
    return None if pd.isna(newest) else int(newest)


def _tier_filters(
    base: dict[str, Any],
    scope: str,
    level: str,
    bounds: dict[str, int],
) -> dict[str, Any]:
    return {
        **base,
        "neighbourhood_applied": scope == "neighbourhood",
        "scope": scope,
        "similarity": level,
        "area_min": bounds.get("area_min"),
        "area_max": bounds.get("area_max"),
        "year_min": bounds.get("year_min"),
        "year_max": bounds.get("year_max"),
        "radius_km": settings.COMPARABLE_NEARBY_RADIUS_KM if scope == "nearby" else None,
    }


def _filters(
    *,
    neighbourhood: str | None,
    resolved: str | None,
    applied: bool,
) -> dict[str, Any]:
    return {
        "neighbourhood": neighbourhood,
        "neighbourhood_resolved": resolved,
        "neighbourhood_applied": applied,
        "recent_days": settings.COMPARABLE_RECENT_DAYS,
    }
