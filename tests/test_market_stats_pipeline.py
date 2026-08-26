"""Market stats groups listings by the month they entered the market."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from application.market_stats.read import get_market_stats
from schemas.price import MarketStatsQuery

TEHRAN = timezone(timedelta(hours=3, minutes=30))
NOW = datetime(2025, 8, 14, 12, 0, tzinfo=TEHRAN)


def _patch_source(monkeypatch, df: pd.DataFrame):
    monkeypatch.setattr(
        "application.market_stats.read.extract_market_history_dataframe",
        lambda *args, **kwargs: df.copy(),
    )
    monkeypatch.setattr(
        "application.market_stats.read.pd.Timestamp.now",
        lambda tz=None: pd.Timestamp(NOW),
    )


def _query(months: int = 6) -> MarketStatsQuery:
    return MarketStatsQuery(
        city_slug="tehran",
        property_type="apartment",
        purpose="sale",
        neighbourhood="ارم",
        months=months,
    )


def test_every_listing_lands_in_exactly_one_month(monkeypatch):
    rows = [
        {
            "neighbourhood": "ارم",
            "price_per_sqm_toman": 200_000_000 + i,
            "first_seen_at": NOW - timedelta(days=i % 25),
        }
        for i in range(71)
    ]
    _patch_source(monkeypatch, pd.DataFrame(rows))

    result = get_market_stats(None, _query())

    assert sum(p.sample_size for p in result.points) == 71
    assert result.sample_size == 71


def test_buckets_follow_market_entry_not_crawl_activity(monkeypatch):
    # Same listings, all last touched today but first seen across three months.
    rows = [
        {
            "neighbourhood": "ارم",
            "price_per_sqm_toman": price,
            "first_seen_at": datetime(2025, month, 10, tzinfo=TEHRAN),
            "last_seen_at": NOW,
        }
        for month, price in ((6, 100_000_000), (7, 150_000_000), (8, 200_000_000))
    ]
    _patch_source(monkeypatch, pd.DataFrame(rows))

    result = get_market_stats(None, _query())
    by_period = {p.period: p.avg_price_per_sqm_toman for p in result.points}

    assert by_period["2025-06"] == 100_000_000
    assert by_period["2025-07"] == 150_000_000
    assert by_period["2025-08"] == 200_000_000


def test_coverage_reports_how_far_back_data_actually_goes(monkeypatch):
    # A 6-month window over a crawl that only started two months ago.
    rows = [
        {
            "neighbourhood": "ارم",
            "price_per_sqm_toman": 200_000_000,
            "first_seen_at": datetime(2025, month, 10, tzinfo=TEHRAN),
        }
        for month in (7, 8)
    ]
    _patch_source(monkeypatch, pd.DataFrame(rows))

    result = get_market_stats(None, _query(months=6))

    assert len(result.points) == 6
    assert result.covered_months == 2
    assert result.coverage_start == "2025-07"


def test_empty_history_reports_no_coverage(monkeypatch):
    _patch_source(monkeypatch, pd.DataFrame())

    result = get_market_stats(None, _query())

    assert result.sample_size == 0
    assert result.covered_months == 0
    assert result.coverage_start is None
