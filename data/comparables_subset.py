"""Shared comparables neighbourhood filtering (same rules as predict)."""
from __future__ import annotations

from typing import Any

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
