#!/usr/bin/env python3
"""Debug market-stats (comparables pipeline). Run on server in Didibood_Price dir."""
from __future__ import annotations

from application.market_stats.read import get_market_stats
from schemas.price import MarketStatsQuery


def main() -> None:
    for neighbourhood in ("بهار", "ارم", None):
        result = get_market_stats(
            None,
            MarketStatsQuery(
                city_slug="tehran",
                neighbourhood=neighbourhood,
                months=6,
            ),
        )
        label = neighbourhood or "(city-wide)"
        print(f"=== {label} sample_size={result.sample_size} ===")
        for point in result.points:
            if point.sample_size > 0:
                print(
                    point.period,
                    point.sample_size,
                    point.avg_price_per_sqm_toman,
                )


if __name__ == "__main__":
    main()
