"""Read models for the neighbourhood analysis page."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from application.neighbourhoods.dataset import AnalyticsFrame, load_analytics_frame
from application.neighbourhoods.filters import (
    NeighbourhoodFilters,
    apply_filters,
    coverage_start,
    with_derived_columns,
)
from application.neighbourhoods.stats import (
    city_summary,
    neighbourhood_detail,
    neighbourhood_rows,
)
from data.comparables_subset import neighbourhood_mask
from data.neighbourhood_persian import (
    normalize_persian_neighbourhood,
    resolve_request_neighbourhood,
)
from domain.model_key import SUPPORTED_CITIES, ModelKey

logger = logging.getLogger(__name__)


def _loaded(key: ModelKey, filters: NeighbourhoodFilters) -> tuple[AnalyticsFrame, pd.DataFrame]:
    loaded = load_analytics_frame(key)
    filtered = apply_filters(
        loaded.frame,
        filters,
        purpose=key.purpose,
        target_column=loaded.target_column,
    )
    # Derive once here, so the ranking and the city summary share the work.
    filtered = with_derived_columns(
        filtered, purpose=key.purpose, target_column=loaded.target_column
    )
    return loaded, filtered


def _meta(loaded: AnalyticsFrame, filtered: pd.DataFrame, filters: NeighbourhoodFilters) -> dict[str, Any]:
    return {
        "city": loaded.key.city_slug,
        "purpose": loaded.key.purpose,
        "property_type": loaded.key.property_type,
        "months": filters.months,
        "metric": "price_per_sqm_toman"
        if loaded.key.purpose == "sale"
        else "equivalent_deposit_toman",
        "filters": filters.as_dict(),
        "data_built_at": loaded.built_at.isoformat(),
        "coverage_start": coverage_start(filtered),
        "total_listings": int(len(loaded.frame)),
    }


def get_overview(key: ModelKey, filters: NeighbourhoodFilters) -> dict[str, Any]:
    """Map values, ranking table and the city headline — one filtered population."""
    loaded, filtered = _loaded(key, filters)
    rows = neighbourhood_rows(
        filtered,
        target_column=loaded.target_column,
        purpose=key.purpose,
        months=filters.months,
    )
    summary = city_summary(
        filtered,
        target_column=loaded.target_column,
        purpose=key.purpose,
        months=filters.months,
        neighbourhood_count=len(rows),
        rows=rows,
    )
    return {**_meta(loaded, filtered, filters), "summary": summary, "neighbourhoods": rows}


def get_detail(
    key: ModelKey,
    filters: NeighbourhoodFilters,
    *,
    neighbourhood: str,
) -> dict[str, Any] | None:
    """One neighbourhood in depth. Returns None when the name matches nothing."""
    loaded, filtered = _loaded(key, filters)
    if filtered.empty:
        return None

    known = {
        str(value).strip()
        for value in filtered["neighbourhood"].dropna().unique()
        if str(value).strip()
    }
    resolved = resolve_request_neighbourhood(neighbourhood, known) or neighbourhood
    subset = filtered.loc[neighbourhood_mask(filtered["neighbourhood"], resolved)]
    if subset.empty:
        wanted = normalize_persian_neighbourhood(neighbourhood)
        logger.info("neighbourhood detail miss city=%s title=%s", key.city_slug, wanted)
        return None

    rows = neighbourhood_rows(
        filtered,
        target_column=loaded.target_column,
        purpose=key.purpose,
        months=filters.months,
    )

    # A sale page answers "what does renting it out return?", which needs the rent
    # frame for the same neighbourhood. Missing rent data is normal, not an error.
    rent_subset: pd.DataFrame | None = None
    if key.purpose == "sale":
        try:
            rent_loaded = load_analytics_frame(
                ModelKey(
                    city_slug=key.city_slug,
                    property_type=key.property_type,
                    purpose="rent",
                )
            )
            rent_filtered = apply_filters(
                rent_loaded.frame,
                filters,
                purpose="rent",
                target_column=rent_loaded.target_column,
            )
            if not rent_filtered.empty:
                rent_subset = rent_filtered.loc[
                    neighbourhood_mask(rent_filtered["neighbourhood"], resolved)
                ]
        except Exception:
            logger.warning("rent frame unavailable for yield city=%s", key.city_slug, exc_info=True)

    detail = neighbourhood_detail(
        subset,
        title=resolved,
        target_column=loaded.target_column,
        purpose=key.purpose,
        filters=filters,
        rows=rows,
        rent_frame=rent_subset,
    )
    return {**_meta(loaded, filtered, filters), "neighbourhood": detail}


def get_titles(key: ModelKey) -> list[str]:
    """Every neighbourhood the data knows about, for the filter's autocomplete."""
    loaded = load_analytics_frame(key)
    if loaded.empty:
        return []
    counts = loaded.frame.groupby("neighbourhood").size().sort_values(ascending=False)
    return [str(title) for title in counts.index]


#: Persian labels for the cities the crawl covers well enough to publish.
CITY_LABELS_FA = {
    "tehran": "تهران",
    "mashhad": "مشهد",
    "isfahan": "اصفهان",
}


def get_cities(
    filters: NeighbourhoodFilters,
    *,
    purpose: str,
    property_type: str = "apartment",
) -> dict[str, Any]:
    """One row per city — what the map shows before anyone picks a city.

    A city with no built frame is skipped rather than reported as empty: it
    means the analytics have not been built for it yet, which is a deployment
    state, not a market fact.
    """
    cities: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    target_column: str | None = None
    for slug in SUPPORTED_CITIES:
        key = ModelKey(city_slug=slug, property_type=property_type, purpose=purpose)
        try:
            loaded, filtered = _loaded(key, filters)
        except Exception:
            logger.warning("city summary unavailable city=%s", slug, exc_info=True)
            continue
        if filtered.empty:
            continue
        frames.append(filtered)
        target_column = loaded.target_column
        rows = neighbourhood_rows(
            filtered,
            target_column=loaded.target_column,
            purpose=purpose,
            months=filters.months,
        )
        summary = city_summary(
            filtered,
            target_column=loaded.target_column,
            purpose=purpose,
            months=filters.months,
            neighbourhood_count=len(rows),
            rows=rows,
        )
        cities.append(
            {
                "city": slug,
                "label": CITY_LABELS_FA.get(slug, slug),
                "sample_size": summary["sample_size"],
                "neighbourhood_count": summary["neighbourhood_count"],
                "median": summary["median"],
                "mean": summary["mean"],
                "p25": summary["p25"],
                "p75": summary["p75"],
                "median_budget_toman": summary["median_budget_toman"],
                "trend_pct": summary["trend_pct"],
                "lat": _median_of(filtered, "location_lat"),
                "lon": _median_of(filtered, "location_long"),
            }
        )

    cities.sort(key=lambda row: row["median"] or 0, reverse=True)

    # The headline above a country-wide map is the country, not its largest
    # city. Recomputed over the pooled listings rather than averaged from the
    # city medians: a median of medians is not a median.
    total: dict[str, Any] | None = None
    if frames and target_column:
        pooled = pd.concat(frames, ignore_index=True)
        total = city_summary(
            pooled,
            target_column=target_column,
            purpose=purpose,
            months=filters.months,
            neighbourhood_count=sum(row["neighbourhood_count"] for row in cities),
        )
        total["city_count"] = len(cities)
        dearest = cities[0]["median"] if cities else None
        cheapest = cities[-1]["median"] if cities else None
        total["spread_ratio"] = (
            round(dearest / cheapest, 1) if dearest and cheapest and cheapest > 0 else None
        )
        total["dearest_city"] = cities[0]["label"] if cities else None
        total["cheapest_city"] = cities[-1]["label"] if cities else None

    return {
        "purpose": purpose,
        "property_type": property_type,
        "months": filters.months,
        "metric": "price_per_sqm_toman" if purpose == "sale" else "equivalent_deposit_toman",
        "cities": cities,
        "total": total,
    }


def _median_of(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else None
