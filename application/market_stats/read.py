"""Monthly market stats — listings grouped by the month they entered the market."""
from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

import pandas as pd

from data.comparables_subset import subset_comparables
from data.extract import extract_market_history_dataframe
from domain.model_key import ModelKey
from domain.target import RENT_TARGET_COLUMN, SALE_TARGET_COLUMN
from schemas.price import MarketStatsPoint, MarketStatsQuery, MarketStatsResponse

logger = logging.getLogger(__name__)

TEHRAN = ZoneInfo("Asia/Tehran")

# Bucketing key: when the listing appeared on the market. last_seen_at is when the
# crawler last touched it, which for live listings is "today" — grouping by it
# measured crawl activity, not price movement.
PERIOD_COLUMN = "first_seen_at"


def _current_month() -> pd.Period:
    """Calendar month in Tehran. Periods carry no tz, so drop it deliberately."""
    return pd.Timestamp.now(tz=TEHRAN).tz_localize(None).to_period("M")


def _month_periods(months: int) -> list[str]:
    end = _current_month()
    start = end - (int(months) - 1)
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def _window_start(months: int) -> pd.Timestamp:
    start = _current_month() - (int(months) - 1)
    return start.to_timestamp(how="start").tz_localize(TEHRAN)


def _period_of(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(TEHRAN).strftime("%Y-%m")


def get_market_stats(_db: object, query: MarketStatsQuery) -> MarketStatsResponse:
    key = ModelKey(
        city_slug=query.city_slug.strip().lower(),
        property_type=query.property_type,
        purpose=query.purpose,
    )
    neighbourhood = (query.neighbourhood or "").strip() or None
    months = int(query.months)
    target_column = (
        RENT_TARGET_COLUMN if query.purpose == "rent" else SALE_TARGET_COLUMN
    )
    period_labels = _month_periods(months)

    points: list[MarketStatsPoint] = []
    total_sample = 0
    covered: list[str] = []

    try:
        dataset = extract_market_history_dataframe(key, since=_window_start(months))
        subset, _filters = subset_comparables(dataset, neighbourhood)

        if not subset.empty and target_column in subset.columns:
            frame = subset.copy()
            frame["period"] = (
                frame[PERIOD_COLUMN].map(_period_of)
                if PERIOD_COLUMN in frame.columns
                else None
            )
            frame = frame[frame["period"].notna()]
            count_by_period = frame.groupby("period").size()

            priced = frame[
                frame[target_column].notna()
                & (frame[target_column].astype(float) > 0)
            ]
            avg_by_period = priced.groupby("period")[target_column].mean()
        else:
            count_by_period = pd.Series(dtype=int)
            avg_by_period = pd.Series(dtype=float)

        for period in period_labels:
            sample = int(count_by_period.get(period, 0))
            avg_val = avg_by_period.get(period)
            points.append(
                MarketStatsPoint(
                    period=period,
                    avg_price_per_sqm_toman=(
                        int(round(float(avg_val)))
                        if sample > 0 and pd.notna(avg_val)
                        else None
                    ),
                    sample_size=sample,
                )
            )
            total_sample += sample
            if sample > 0:
                covered.append(period)
    except Exception:
        logger.exception(
            "market_stats history pipeline failed city=%s neighbourhood=%s",
            key.city_slug,
            neighbourhood,
        )
        points = [
            MarketStatsPoint(period=period, avg_price_per_sqm_toman=None, sample_size=0)
            for period in period_labels
        ]
        total_sample = 0
        covered = []

    return MarketStatsResponse(
        city_slug=key.city_slug,
        purpose=query.purpose,
        property_type=query.property_type,
        neighbourhood=neighbourhood,
        months=months,
        points=points,
        sample_size=total_sample,
        # Crawl history is young, so callers must be able to tell "the market was
        # flat" from "we have not been collecting that long".
        coverage_start=covered[0] if covered else None,
        covered_months=len(covered),
    )
