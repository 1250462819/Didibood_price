#!/usr/bin/env python3
"""Print comparables + monthly counts for a neighbourhood (same pipeline as market-stats)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from application.market_stats.read import get_market_stats  # noqa: E402
from config.settings import settings  # noqa: E402
from data.comparables_subset import subset_comparables  # noqa: E402
from data.extract import extract_comparables_dataframe  # noqa: E402
from domain.features import PricingFeatures  # noqa: E402
from domain.model_key import ModelKey  # noqa: E402
from inference.predictor import comparable_stats  # noqa: E402
from schemas.price import MarketStatsQuery  # noqa: E402

TEHRAN = ZoneInfo("Asia/Tehran")


def persian_label(ym: str) -> str:
    import jdatetime

    y, m = map(int, ym.split("-"))
    g = jdatetime.date.fromgregorian(year=y, month=m, day=15)
    names = [
        "فروردین",
        "اردیبهشت",
        "خرداد",
        "تیر",
        "مرداد",
        "شهریور",
        "مهر",
        "آبان",
        "آذر",
        "دی",
        "بهمن",
        "اسفند",
    ]
    return f"{names[g.month - 1]} {g.jyear}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("neighbourhood")
    parser.add_argument("--city", default="tehran")
    args = parser.parse_args()

    key = ModelKey(city_slug=args.city, property_type="apartment", purpose="sale")
    neighbourhood = args.neighbourhood.strip()

    dataset = extract_comparables_dataframe(
        key,
        recent_days=settings.COMPARABLE_RECENT_DAYS,
    )
    subset, filters = subset_comparables(dataset, neighbourhood)

    features = PricingFeatures.from_request(
        {
            "city_slug": args.city,
            "neighbourhood": neighbourhood,
            "area": 85,
            "rooms": 2,
        }
    )
    comp = comparable_stats(
        features,
        dataset=dataset,
        target_column="price_per_sqm_toman",
    )

    stats = get_market_stats(
        None,
        MarketStatsQuery(
            city_slug=args.city,
            neighbourhood=neighbourhood,
            months=6,
        ),
    )

    print(f"neighbourhood={neighbourhood!r} resolved={filters.get('neighbourhood_resolved')!r}")
    print(f"recent_days={settings.COMPARABLE_RECENT_DAYS}")
    print(f"total_30d={comp['sample_size']}")
    for point in stats.points:
        if point.sample_size > 0:
            print(
                f"month {point.period} ({persian_label(point.period)}): {point.sample_size}"
            )
    print(f"monthly_sum={stats.sample_size}")


if __name__ == "__main__":
    main()
