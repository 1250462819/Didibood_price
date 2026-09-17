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
from domain.model_key import ModelKey

logger = logging.getLogger(__name__)


def _loaded(key: ModelKey, filters: NeighbourhoodFilters) -> tuple[AnalyticsFrame, pd.DataFrame]:
    loaded = load_analytics_frame(key)
    filtered = apply_filters(
        loaded.frame,
        filters,
        purpose=key.purpose,
        target_column=loaded.target_column,
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
