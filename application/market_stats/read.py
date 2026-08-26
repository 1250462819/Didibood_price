"""Monthly market stats — same comparables extract/filter, grouped by calendar month."""
from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import settings
from data.comparables_subset import subset_comparables
from data.extract import extract_comparables_dataframe
from domain.model_key import ModelKey
from domain.target import RENT_TARGET_COLUMN, SALE_TARGET_COLUMN
from schemas.price import MarketStatsPoint, MarketStatsQuery, MarketStatsResponse

logger = logging.getLogger(__name__)

TEHRAN = ZoneInfo("Asia/Tehran")


def _month_periods(months: int) -> list[str]:
    end = pd.Timestamp.now(tz=TEHRAN).to_period("M")
    start = end - (int(months) - 1)
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def _period_from_last_seen(value: object) -> str | None:
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

    try:
        dataset = extract_comparables_dataframe(
            key,
            recent_days=settings.COMPARABLE_RECENT_DAYS,
        )
        subset, _filters = subset_comparables(dataset, neighbourhood)

        if not subset.empty and target_column in subset.columns:
            period_frame = subset.copy()
            if "last_seen_at" in period_frame.columns:
                period_frame["period"] = period_frame["last_seen_at"].map(
                    _period_from_last_seen
                )
            else:
                period_frame["period"] = None
            period_frame = period_frame[period_frame["period"].notna()]
            count_by_period = period_frame.groupby("period").size()

            price_frame = subset[
                subset[target_column].notna()
                & (subset[target_column].astype(float) > 0)
            ].copy()
            if "last_seen_at" in price_frame.columns:
                price_frame["period"] = price_frame["last_seen_at"].map(
                    _period_from_last_seen
                )
            else:
                price_frame["period"] = None
            price_frame = price_frame[price_frame["period"].notna()]
            avg_by_period = price_frame.groupby("period")[target_column].mean()
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
                        int(round(float(avg_val))) if sample > 0 and pd.notna(avg_val) else None
                    ),
                    sample_size=sample,
                )
            )
            total_sample += sample
    except Exception:
        logger.exception(
            "market_stats comparables pipeline failed city=%s neighbourhood=%s",
            key.city_slug,
            neighbourhood,
        )
        points = [
            MarketStatsPoint(period=period, avg_price_per_sqm_toman=None, sample_size=0)
            for period in period_labels
        ]
        total_sample = 0

    return MarketStatsResponse(
        city_slug=key.city_slug,
        purpose=query.purpose,
        property_type=query.property_type,
        neighbourhood=neighbourhood,
        months=months,
        points=points,
        sample_size=total_sample,
    )
