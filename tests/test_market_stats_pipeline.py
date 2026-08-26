"""Market stats uses the same comparables pool, grouped by month."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from application.market_stats.read import get_market_stats
from schemas.price import MarketStatsQuery


def test_market_stats_monthly_counts_match_comparables_subset(monkeypatch):
    tehran = timezone(timedelta(hours=3, minutes=30))
    now = datetime(2025, 8, 14, 12, 0, tzinfo=tehran)
    rows = []
    for i in range(71):
        rows.append(
            {
                "neighbourhood": "ارم",
                "price_per_sqm_toman": 200_000_000 + i,
                "last_seen_at": now - timedelta(days=i % 25),
            }
        )

    df = pd.DataFrame(rows)

    monkeypatch.setattr(
        "application.market_stats.read.extract_comparables_dataframe",
        lambda *args, **kwargs: df.copy(),
    )
    monkeypatch.setattr(
        "application.market_stats.read.pd.Timestamp.now",
        lambda tz=None: pd.Timestamp(now),
    )

    query = MarketStatsQuery(
        city_slug="tehran",
        property_type="apartment",
        purpose="sale",
        neighbourhood="ارم",
        months=6,
    )
    result = get_market_stats(None, query)

    month_counts = [p.sample_size for p in result.points if p.sample_size > 0]
    assert sum(month_counts) == 71
    assert result.sample_size == 71
