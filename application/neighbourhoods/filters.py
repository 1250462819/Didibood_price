"""What the visitor narrowed the market down to.

One filter set drives every number on the page — map, ranking, charts and the
sample listings — so a reader never sees a headline computed over a different
population than the chart below it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

TEHRAN = ZoneInfo("Asia/Tehran")

#: A neighbourhood thinner than this is still shown — hiding it would silently
#: shrink the map — but it is flagged so a median of four listings is never read
#: as a market rate.
LOW_SAMPLE_THRESHOLD = 12


@dataclass(frozen=True)
class NeighbourhoodFilters:
    months: int = 6
    min_area: float | None = None
    max_area: float | None = None
    min_rooms: int | None = None
    max_rooms: int | None = None
    max_building_age: int | None = None
    min_budget_toman: float | None = None
    max_budget_toman: float | None = None
    has_parking: bool | None = None
    has_elevator: bool | None = None
    has_storage: bool | None = None
    has_balcony: bool | None = None

    def as_dict(self) -> dict[str, object]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def window_start(months: int) -> pd.Timestamp:
    """Start of the month `months - 1` back, on the Tehran calendar."""
    current = pd.Timestamp.now(tz=TEHRAN).tz_localize(None).to_period("M")
    return (current - (int(months) - 1)).to_timestamp(how="start").tz_localize(TEHRAN)


def month_labels(months: int) -> list[str]:
    current = pd.Timestamp.now(tz=TEHRAN).tz_localize(None).to_period("M")
    start = current - (int(months) - 1)
    return [p.strftime("%Y-%m") for p in pd.period_range(start, current, freq="M")]


def to_period(value: object) -> str | None:
    """`YYYY-MM` in Tehran for a crawl timestamp, which is stored in UTC."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(TEHRAN).strftime("%Y-%m")


def budget_series(frame: pd.DataFrame, purpose: str, target_column: str) -> pd.Series:
    """What "budget" means for this purpose, in Toman.

    Sale: the asking price. Rent: the deposit-equivalent, the one number that
    puts a 500M/5M listing and a 2B/0 listing on the same scale — the same figure
    the rent model is trained on.
    """
    if purpose == "rent":
        return pd.to_numeric(frame.get(target_column), errors="coerce")
    if "price_total_toman" in frame.columns:
        price = pd.to_numeric(frame["price_total_toman"], errors="coerce")
        # Divar lists a per-metre price for some sale posts and no total at all;
        # the area is already validated, so the product is the honest fallback.
        area = pd.to_numeric(frame.get("area"), errors="coerce")
        per_sqm = pd.to_numeric(frame.get(target_column), errors="coerce")
        return price.fillna(area * per_sqm)
    return pd.Series(index=frame.index, dtype="float64")


def apply_filters(
    frame: pd.DataFrame,
    filters: NeighbourhoodFilters,
    *,
    purpose: str,
    target_column: str,
) -> pd.DataFrame:
    """Narrow the frame. Rows missing a filtered attribute are dropped, because
    "we don't know its age" is not the same answer as "it is new enough"."""
    if frame.empty:
        return frame

    out = frame
    if "first_seen_at" in out.columns:
        seen = pd.to_datetime(out["first_seen_at"], utc=True, errors="coerce")
        out = out[seen >= window_start(filters.months)]

    numeric_windows = (
        ("area", filters.min_area, filters.max_area),
        ("rooms", filters.min_rooms, filters.max_rooms),
    )
    for column, low, high in numeric_windows:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        if low is not None:
            out = out[values >= low]
            values = values.loc[out.index]
        if high is not None:
            out = out[values <= high]

    if filters.max_building_age is not None and "building_age" in out.columns:
        age = pd.to_numeric(out["building_age"], errors="coerce")
        out = out[age.notna() & (age <= filters.max_building_age)]

    if filters.min_budget_toman is not None or filters.max_budget_toman is not None:
        budget = budget_series(out, purpose, target_column)
        if filters.min_budget_toman is not None:
            out = out[budget >= filters.min_budget_toman]
            budget = budget.loc[out.index]
        if filters.max_budget_toman is not None:
            out = out[budget <= filters.max_budget_toman]

    amenities = (
        ("has_parking", filters.has_parking),
        ("has_elevator", filters.has_elevator),
        ("has_storage", filters.has_storage),
        ("has_balcony", filters.has_balcony),
    )
    for column, wanted in amenities:
        if wanted is None or column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        out = out[values == (1 if wanted else 0)]

    return out


def coverage_start(frame: pd.DataFrame) -> str | None:
    if frame.empty or "first_seen_at" not in frame.columns:
        return None
    seen = pd.to_datetime(frame["first_seen_at"], utc=True, errors="coerce").dropna()
    if seen.empty:
        return None
    return to_period(seen.min())


def jalali_now_year() -> int:
    """Only used for labels; the frame already carries building_age."""
    return int(datetime.now(tz=TEHRAN).year) - 621
