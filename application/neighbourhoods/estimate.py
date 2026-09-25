"""Fill a filtered map where a neighbourhood has too few matching listings.

Ask for four-room flats with a balcony and most neighbourhoods have none, or
three. Rather than grey them out, the page borrows the city's answer to "how
much dearer is this kind of flat than the neighbourhood's average?" and applies
it to each neighbourhood's own average:

    estimate = the neighbourhood's all-listings price × the segment's multiplier

The multiplier is measured *inside* neighbourhoods — the segment's average
against the same neighbourhood's overall average, combined across the city —
because comparing the city's four-room flats with its one-room flats mostly
measures where big flats are built. The whole filter combination gets one
multiplier; the page itself says amenity premiums do not add up.

A neighbourhood's own listings count in proportion to how many it has: none →
the estimate alone, six → half and half, `LOW_SAMPLE_THRESHOLD` or more → its
own figures untouched. Rows that lean on the estimate at all say so
(`estimated`, `estimate_share`), and stay `low_sample`, so the page keeps them
out of anything that promises a price.
"""
from __future__ import annotations

import math
from typing import Any

from application.neighbourhoods.filters import (
    LOW_SAMPLE_THRESHOLD,
    NeighbourhoodFilters,
)
from application.neighbourhoods.stats import MIN_AMENITY_SIDE, _split

#: A neighbourhood needs this many matching listings before its own ratio to its
#: overall average is allowed to shape the city's multiplier.
MIN_RATIO_SAMPLE = MIN_AMENITY_SIDE

#: …and the multiplier needs this many such neighbourhoods behind it.
MIN_RATIO_HOODS = 5

#: A neighbourhood with fewer listings than this *overall* has no average worth
#: scaling; it is left out rather than estimated.
MIN_BASE_SAMPLE = MIN_AMENITY_SIDE

#: Figures that scale with the price multiplier, the area one and the budget one.
PRICE_FIELDS = ("mean", "median", "p25", "p75")
AREA_FIELDS = ("mean_area", "median_area")
BUDGET_FIELDS = ("mean_budget_toman", "median_budget_toman")

#: Everything but these describes the whole market, not a population to narrow.
_BASE_ONLY = {"months"}


def narrows_segment(filters: NeighbourhoodFilters) -> bool:
    """True when the filters pick a kind of home, not just a time window."""
    return any(name not in _BASE_ONLY for name in filters.as_dict())


def base_filters(filters: NeighbourhoodFilters) -> NeighbourhoodFilters:
    return NeighbourhoodFilters(months=filters.months)


def _positive(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _avg(row: dict[str, Any], mean_field: str, median_field: str) -> float | None:
    return _positive(row.get(mean_field)) or _positive(row.get(median_field))


def segment_multiplier(
    segment: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    mean_field: str,
    median_field: str,
) -> float | None:
    """Listing-weighted geometric mean of segment ÷ overall, per neighbourhood.

    Geometric, because ratios multiply: +25% in one place and −20% in another
    are the same distance from «no difference».
    """
    weighted = 0.0
    total = 0.0
    hoods = 0
    for title, row in segment.items():
        sample = int(row.get("sample_size") or 0)
        whole = base.get(title)
        if sample < MIN_RATIO_SAMPLE or whole is None:
            continue
        own, overall = _avg(row, mean_field, median_field), _avg(whole, mean_field, median_field)
        if own is None or overall is None:
            continue
        weighted += math.log(own / overall) * sample
        total += sample
        hoods += 1
    if hoods < MIN_RATIO_HOODS or not total:
        return None
    return math.exp(weighted / total)


def _blend(own: Any, overall: Any, multiplier: float | None, weight: float) -> float | None:
    own_value = _positive(own)
    overall_value = _positive(overall)
    estimate = overall_value * multiplier if overall_value is not None and multiplier is not None else None
    if estimate is None:
        return own_value
    if own_value is None:
        return estimate
    return weight * own_value + (1 - weight) * estimate


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: row["median"], reverse=True)
    total = len(rows)
    for index, row in enumerate(rows):
        row["rank"] = index + 1
        row["percentile"] = round((total - index) / total * 100, 1) if total else None
    return rows


def fill_segment_gaps(
    segment_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    filters: NeighbourhoodFilters,
) -> list[dict[str, Any]]:
    """The filtered ranking, with thin and missing neighbourhoods estimated.

    Returns `segment_rows` unchanged (bar the two new flags) when the city has
    too few comparable neighbourhoods to trust a multiplier.
    """
    segment = {row["title"]: row for row in segment_rows}
    base = {row["title"]: row for row in base_rows}
    multipliers = {
        "price": segment_multiplier(segment, base, "mean", "median"),
        "area": segment_multiplier(segment, base, "mean_area", "median_area"),
        "budget": segment_multiplier(segment, base, "mean_budget_toman", "median_budget_toman"),
    }
    if multipliers["price"] is None:
        return [{**row, "estimated": False, "estimate_share": 0.0} for row in segment_rows]

    rows: list[dict[str, Any]] = []
    for title in {*segment, *base}:
        own = segment.get(title)
        whole = base.get(title)
        sample = int(own.get("sample_size") or 0) if own else 0
        weight = min(sample, LOW_SAMPLE_THRESHOLD) / LOW_SAMPLE_THRESHOLD
        usable_base = whole is not None and int(whole.get("sample_size") or 0) >= MIN_BASE_SAMPLE
        if weight >= 1 or not usable_base:
            if own is not None:
                rows.append({**own, "estimated": False, "estimate_share": 0.0})
            continue

        row = dict(own) if own is not None else {
            **whole,
            "sample_size": 0,
            "median_rooms": filters.min_rooms if filters.min_rooms is not None else whole.get("median_rooms"),
            "parking": _split(0, 0, None, None),
            "elevator": _split(0, 0, None, None),
        }
        own_row = own or {}
        for fields, family in ((PRICE_FIELDS, "price"), (AREA_FIELDS, "area"), (BUDGET_FIELDS, "budget")):
            for name in fields:
                row[name] = _blend(own_row.get(name), whole.get(name), multipliers[family], weight)
        if row.get("median") is None:
            continue
        if row.get("trend_pct") is None:
            row["trend_pct"] = whole.get("trend_pct")
        row["sample_size"] = sample
        row["low_sample"] = True
        row["estimated"] = True
        row["estimate_share"] = round(1 - weight, 3)
        rows.append(row)

    return _rank(rows)
