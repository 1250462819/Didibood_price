"""Neighbourhood aggregates: the ranking, the map values and one hood's detail.

Medians throughout, never means: asking prices have a long right tail, and one
penthouse must not move a neighbourhood's headline. Percentiles are reported
alongside so the spread is visible rather than implied.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from application.neighbourhoods.filters import (
    LOW_SAMPLE_THRESHOLD,
    NeighbourhoodFilters,
    budget_series,
    month_labels,
    to_period,
)

#: A month needs this many listings before its median is allowed to move a trend
#: line. Below it the month is reported with its count and no price.
MIN_TREND_SAMPLE = 5

#: Bands used for the breakdown tables. Chosen to match how people search rather
#: than to split the data evenly.
AREA_BANDS = ((0, 60), (60, 90), (90, 120), (120, 160), (160, 10_000))
AGE_BANDS = ((0, 5), (5, 10), (10, 20), (20, 200))
AMENITIES = (
    ("has_parking", "پارکینگ"),
    ("has_elevator", "آسانسور"),
    ("has_storage", "انباری"),
    ("has_balcony", "بالکن"),
)


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _i(value: Any) -> int | None:
    number = _f(value)
    return None if number is None else int(round(number))


def _median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return _f(values.median()) if not values.empty else None


def _quantile(series: pd.Series, q: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return _f(values.quantile(q)) if not values.empty else None


def _pct_change(old: float | None, new: float | None) -> float | None:
    if not old or not new or old <= 0:
        return None
    return _f((new - old) / old * 100.0)


def monthly_points(
    frame: pd.DataFrame,
    *,
    target_column: str,
    months: int,
) -> list[dict[str, Any]]:
    """Median price per month of market entry, one entry per month in the window."""
    labels = month_labels(months)
    if frame.empty or "first_seen_at" not in frame.columns:
        return [{"period": p, "median": None, "sample_size": 0} for p in labels]

    working = frame.copy()
    working["period"] = working["first_seen_at"].map(to_period)
    working = working[working["period"].notna()]
    counts = working.groupby("period").size()
    medians = working.groupby("period")[target_column].median()

    points: list[dict[str, Any]] = []
    for label in labels:
        sample = int(counts.get(label, 0))
        median = _f(medians.get(label)) if sample >= MIN_TREND_SAMPLE else None
        points.append({"period": label, "median": median, "sample_size": sample})
    return points


def trend_pct(points: list[dict[str, Any]]) -> float | None:
    """Change between the first and last month that each carry enough listings."""
    priced = [p for p in points if p["median"] is not None]
    if len(priced) < 2:
        return None
    return _pct_change(priced[0]["median"], priced[-1]["median"])


def neighbourhood_rows(
    frame: pd.DataFrame,
    *,
    target_column: str,
    purpose: str,
    months: int,
) -> list[dict[str, Any]]:
    """One row per neighbourhood: the ranking table and the map's colour values."""
    if frame.empty:
        return []

    budget = budget_series(frame, purpose, target_column)
    working = frame.assign(_budget=budget)

    rows: list[dict[str, Any]] = []
    for title, group in working.groupby("neighbourhood", sort=False):
        sample = int(len(group))
        median_target = _median(group[target_column])
        if median_target is None:
            continue
        points = monthly_points(group, target_column=target_column, months=months)
        rows.append(
            {
                "title": str(title),
                "sample_size": sample,
                "low_sample": sample < LOW_SAMPLE_THRESHOLD,
                "median": median_target,
                "p25": _quantile(group[target_column], 0.25),
                "p75": _quantile(group[target_column], 0.75),
                "median_budget_toman": _median(group["_budget"]),
                "median_area": _median(group["area"]),
                "median_rooms": _i(_median(group["rooms"])),
                "median_building_age": _i(_median(group.get("building_age", pd.Series(dtype=float)))),
                "lat": _median(group.get("location_lat", pd.Series(dtype=float))),
                "lon": _median(group.get("location_long", pd.Series(dtype=float))),
                "trend_pct": trend_pct(points),
            }
        )

    rows.sort(key=lambda row: row["median"], reverse=True)
    total = len(rows)
    for index, row in enumerate(rows):
        row["rank"] = index + 1
        # Percentile of the *ranking*, so "cheaper than 80% of neighbourhoods"
        # stays true whatever the price distribution looks like.
        row["percentile"] = _f(round((total - index) / total * 100, 1)) if total else None
    return rows


def city_summary(
    frame: pd.DataFrame,
    *,
    target_column: str,
    purpose: str,
    months: int,
    neighbourhood_count: int,
) -> dict[str, Any]:
    points = monthly_points(frame, target_column=target_column, months=months)
    budget = budget_series(frame, purpose, target_column)
    return {
        "sample_size": int(len(frame)),
        "neighbourhood_count": neighbourhood_count,
        "median": _median(frame[target_column]) if not frame.empty else None,
        "p25": _quantile(frame[target_column], 0.25) if not frame.empty else None,
        "p75": _quantile(frame[target_column], 0.75) if not frame.empty else None,
        "median_budget_toman": _median(budget) if not frame.empty else None,
        "median_area": _median(frame["area"]) if not frame.empty else None,
        "trend_pct": trend_pct(points),
        "monthly": points,
    }


def _band_label(low: float, high: float, unit: str) -> str:
    if high >= 10_000:
        return f"{int(low)}+ {unit}"
    if low == 0:
        return f"تا {int(high)} {unit}"
    return f"{int(low)}–{int(high)} {unit}"


def _band_rows(
    frame: pd.DataFrame,
    *,
    column: str,
    bands: tuple[tuple[float, float], ...],
    unit: str,
    target_column: str,
) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []
    values = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for low, high in bands:
        mask = values.notna() & (values >= low) & (values < high)
        subset = frame[mask]
        if subset.empty:
            continue
        rows.append(
            {
                "label": _band_label(low, high, unit),
                "sample_size": int(len(subset)),
                "median": _median(subset[target_column]),
            }
        )
    return rows


def _rooms_rows(frame: pd.DataFrame, *, target_column: str) -> list[dict[str, Any]]:
    if frame.empty or "rooms" not in frame.columns:
        return []
    rooms = pd.to_numeric(frame["rooms"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for value in sorted({int(v) for v in rooms.dropna().unique() if 0 <= v <= 6}):
        subset = frame[rooms == value]
        if len(subset) < 3:
            continue
        rows.append(
            {
                "label": "بدون اتاق" if value == 0 else f"{value} خوابه",
                "rooms": value,
                "sample_size": int(len(subset)),
                "median": _median(subset[target_column]),
            }
        )
    return rows


def _amenity_rows(frame: pd.DataFrame, *, target_column: str) -> list[dict[str, Any]]:
    """What a feature is worth here — the gap between listings that have it and
    those that do not. Not a causal claim: a building with a lift differs in more
    ways than the lift."""
    rows: list[dict[str, Any]] = []
    for column, label in AMENITIES:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        with_it = frame[values == 1]
        without = frame[values == 0]
        if len(with_it) < 5 or len(without) < 5:
            continue
        median_with = _median(with_it[target_column])
        median_without = _median(without[target_column])
        rows.append(
            {
                "label": label,
                "key": column,
                "with_sample": int(len(with_it)),
                "without_sample": int(len(without)),
                "median_with": median_with,
                "median_without": median_without,
                "premium_pct": _pct_change(median_without, median_with),
            }
        )
    return rows


def price_histogram(frame: pd.DataFrame, *, target_column: str, buckets: int = 10) -> list[dict[str, Any]]:
    """Where this neighbourhood's asking prices actually sit."""
    values = pd.to_numeric(frame[target_column], errors="coerce").dropna()
    if len(values) < 10:
        return []
    low, high = values.quantile(0.02), values.quantile(0.98)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return []
    edges = np.linspace(low, high, buckets + 1)
    counts, _ = np.histogram(values.clip(low, high), bins=edges)
    return [
        {
            "from": _f(edges[i]),
            "to": _f(edges[i + 1]),
            "count": int(counts[i]),
        }
        for i in range(buckets)
    ]


def rent_yield_pct(
    sale_per_sqm: float | None,
    rent_frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Gross annual yield: a year of rent over the purchase price, per square metre.

    Deposit is converted to monthly at the site's own ratio before the division,
    so a deposit-heavy listing and a rent-heavy one are comparable.
    """
    if not sale_per_sqm or rent_frame is None or rent_frame.empty:
        return None

    area = pd.to_numeric(rent_frame.get("area"), errors="coerce")
    monthly = pd.to_numeric(rent_frame.get("monthly_rent_toman"), errors="coerce").fillna(0)
    deposit = pd.to_numeric(rent_frame.get("deposit_toman"), errors="coerce").fillna(0)

    from config.settings import settings

    ratio = settings.RENT_RATIO_MONTHLY_TOMAN / settings.RENT_RATIO_DEPOSIT_TOMAN
    effective_monthly = monthly + deposit * ratio
    per_sqm = (effective_monthly / area).replace([np.inf, -np.inf], np.nan).dropna()
    if per_sqm.empty:
        return None

    monthly_per_sqm = _f(per_sqm.median())
    if not monthly_per_sqm:
        return None
    return {
        "sample_size": int(len(per_sqm)),
        "monthly_rent_per_sqm_toman": monthly_per_sqm,
        "median_deposit_toman": _median(deposit[deposit > 0]),
        "median_monthly_rent_toman": _median(monthly[monthly > 0]),
        "gross_yield_pct": _f(monthly_per_sqm * 12 / sale_per_sqm * 100),
    }


def similar_neighbourhoods(
    rows: list[dict[str, Any]],
    title: str,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Neighbourhoods priced closest to this one — the "instead of here, look at…" list."""
    anchor = next((row for row in rows if row["title"] == title), None)
    if anchor is None or not anchor["median"]:
        return []
    others = [
        row
        for row in rows
        if row["title"] != title and row["median"] and not row["low_sample"]
    ]
    others.sort(key=lambda row: abs(row["median"] - anchor["median"]))
    return [
        {
            "title": row["title"],
            "median": row["median"],
            "sample_size": row["sample_size"],
            "diff_pct": _pct_change(anchor["median"], row["median"]),
            "rank": row["rank"],
        }
        for row in others[:limit]
    ]


def nearby_neighbourhoods(
    rows: list[dict[str, Any]],
    title: str,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Geographic neighbours, by the median position of their listings."""
    anchor = next((row for row in rows if row["title"] == title), None)
    if anchor is None or anchor["lat"] is None or anchor["lon"] is None:
        return []

    def distance_km(row: dict[str, Any]) -> float:
        # Equirectangular at Tehran's latitude — good to a few metres over a city,
        # and it keeps the whole ranking in one cheap vectorised pass.
        lat_km = (row["lat"] - anchor["lat"]) * 111.0
        lon_km = (row["lon"] - anchor["lon"]) * 111.0 * math.cos(math.radians(anchor["lat"]))
        return math.hypot(lat_km, lon_km)

    others = [
        row
        for row in rows
        if row["title"] != title and row["lat"] is not None and row["lon"] is not None
    ]
    others.sort(key=distance_km)
    return [
        {
            "title": row["title"],
            "median": row["median"],
            "sample_size": row["sample_size"],
            "diff_pct": _pct_change(anchor["median"], row["median"]),
            "distance_km": _f(round(distance_km(row), 2)),
        }
        for row in others[:limit]
    ]


def neighbourhood_detail(
    frame: pd.DataFrame,
    *,
    title: str,
    target_column: str,
    purpose: str,
    filters: NeighbourhoodFilters,
    rows: list[dict[str, Any]],
    rent_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Everything the detail panel shows for one neighbourhood."""
    row = next((item for item in rows if item["title"] == title), None)
    points = monthly_points(frame, target_column=target_column, months=filters.months)
    budget = budget_series(frame, purpose, target_column)

    return {
        "title": title,
        "sample_size": int(len(frame)),
        "low_sample": len(frame) < LOW_SAMPLE_THRESHOLD,
        "price": {
            "median": _median(frame[target_column]) if not frame.empty else None,
            "p10": _quantile(frame[target_column], 0.10) if not frame.empty else None,
            "p25": _quantile(frame[target_column], 0.25) if not frame.empty else None,
            "p75": _quantile(frame[target_column], 0.75) if not frame.empty else None,
            "p90": _quantile(frame[target_column], 0.90) if not frame.empty else None,
            "median_budget_toman": _median(budget) if not frame.empty else None,
            "median_area": _median(frame["area"]) if not frame.empty else None,
        },
        "rank": row["rank"] if row else None,
        "percentile": row["percentile"] if row else None,
        "trend_pct": trend_pct(points),
        "monthly": points,
        "histogram": price_histogram(frame, target_column=target_column),
        "by_rooms": _rooms_rows(frame, target_column=target_column),
        "by_area": _band_rows(
            frame, column="area", bands=AREA_BANDS, unit="متر", target_column=target_column
        ),
        "by_age": _band_rows(
            frame, column="building_age", bands=AGE_BANDS, unit="سال", target_column=target_column
        ),
        "amenities": _amenity_rows(frame, target_column=target_column),
        "rent": rent_yield_pct(_median(frame[target_column]) if not frame.empty else None, rent_frame)
        if purpose == "sale"
        else None,
        "similar": similar_neighbourhoods(rows, title),
        "nearby": nearby_neighbourhoods(rows, title),
    }
