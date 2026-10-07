"""Comparable stats tests."""
from __future__ import annotations

import pandas as pd

from domain.features import PricingFeatures
from inference.predictor import comparable_stats


def test_comparable_min_max_ignore_outliers():
    rows = [180_000_000] * 30 + [200_000_000] * 30 + [220_000_000] * 30
    rows += [20_000_000, 900_000_000]  # obvious outliers
    df = pd.DataFrame(
        {
            "neighbourhood": ["جردن"] * len(rows),
            "area": [85.0] * len(rows),
            "rooms": [2] * len(rows),
            "price_per_sqm_toman": rows,
        }
    )
    features = PricingFeatures.from_request(
        {
            "city_slug": "tehran",
            "neighbourhood": "جردن",
            "area": 85,
            "rooms": 2,
        }
    )
    result = comparable_stats(
        features,
        dataset=df,
        target_column="price_per_sqm_toman",
    )

    # Stats describe the trimmed population, so the count must match it too.
    assert result["sample_size"] < len(rows)
    assert result["min_price_per_sqm_toman"] >= 150_000_000
    assert result["max_price_per_sqm_toman"] <= 300_000_000
    assert result["q1_price_per_sqm_toman"] is not None
    assert result["q3_price_per_sqm_toman"] is not None
    assert result["q1_price_per_sqm_toman"] <= result["median_price_per_sqm_toman"] <= result["q3_price_per_sqm_toman"]
    assert result["filters_applied"]["neighbourhood_applied"] is True
    assert result["filters_applied"]["min_max_outliers_removed"] >= 2


def test_max_stays_near_the_median_on_a_fat_right_tail():
    """Regression: Tehran city-wide reported max 1.06B/sqm against a 200M median.

    log-IQR at 1.5x let that through, and because mean/median were computed on
    the untrimmed series while min/max were not, the published stats described
    two different populations.
    """
    rows = [200_000_000 + (i % 2000) * 50_000 for i in range(3000)]
    rows += [1_061_538_462, 980_000_000, 40_000_000]
    df = pd.DataFrame(
        {
            "neighbourhood": ["جردن"] * len(rows),
            "area": [85.0] * len(rows),
            "rooms": [2] * len(rows),
            "price_per_sqm_toman": rows,
        }
    )
    features = PricingFeatures.from_request(
        {"city_slug": "tehran", "neighbourhood": "جردن", "area": 85, "rooms": 2}
    )

    result = comparable_stats(
        features, dataset=df, target_column="price_per_sqm_toman"
    )

    median = result["median_price_per_sqm_toman"]
    assert result["max_price_per_sqm_toman"] < median * 2
    assert result["min_price_per_sqm_toman"] > median * 0.5
    # One population: min <= q1 <= median <= q3 <= max, all from the same rows.
    assert (
        result["min_price_per_sqm_toman"]
        <= result["q1_price_per_sqm_toman"]
        <= median
        <= result["q3_price_per_sqm_toman"]
        <= result["max_price_per_sqm_toman"]
    )


def test_comparable_keeps_small_neighbourhood_sample():
    df = pd.DataFrame(
        {
            "neighbourhood": ["آجودانیه", "آجودانیه", "جردن"] * 10,
            "area": [85.0, 90.0, 85.0] * 10,
            "rooms": [2, 2, 2] * 10,
            "price_per_sqm_toman": [300_000_000, 310_000_000, 200_000_000] * 10,
        }
    )
    features = PricingFeatures.from_request(
        {
            "city_slug": "tehran",
            "neighbourhood": "آجودانیه",
            "area": 85,
            "rooms": 2,
        }
    )
    result = comparable_stats(
        features,
        dataset=df,
        target_column="price_per_sqm_toman",
    )

    assert result["sample_size"] == 20
    assert result["filters_applied"]["neighbourhood_applied"] is True


def test_comparable_falls_back_to_whole_neighbourhood_when_too_few_similar():
    df = pd.DataFrame(
        {
            "neighbourhood": ["بهار"] * 5,
            "area": [50.0, 60.0, 85.0, 100.0, 120.0],
            "rooms": [1, 2, 3, 4, 1],
            "price_per_sqm_toman": [200_000_000] * 5,
        }
    )
    features = PricingFeatures.from_request(
        {
            "city_slug": "tehran",
            "neighbourhood": "بهار",
            "area": 85,
            "rooms": 1,
        }
    )
    result = comparable_stats(
        features,
        dataset=df,
        target_column="price_per_sqm_toman",
    )

    assert result["sample_size"] == 5
    filters = result["filters_applied"]
    assert filters["recent_days"] == 30
    assert filters["scope"] == "neighbourhood"
    assert filters["similarity"] == "none"
    assert filters["area_min"] is None


def _listings(
    neighbourhood: str,
    *,
    count: int,
    area: float,
    year: int | None,
    pps: int,
    lat: float = 32.70,
    lng: float = 51.65,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neighbourhood": [neighbourhood] * count,
            "area": [area] * count,
            "year_built": [year] * count,
            "location_lat": [lat] * count,
            "location_long": [lng] * count,
            "price_per_sqm_toman": [pps] * count,
        }
    )


def test_comparable_matches_size_and_age_within_neighbourhood():
    """A new 170 m² flat is compared with new large flats, not the old stock."""
    df = pd.concat(
        [
            _listings("ملک شهر", count=10, area=165.0, year=1403, pps=125_000_000),
            _listings("ملک شهر", count=40, area=85.0, year=1385, pps=90_000_000),
        ],
        ignore_index=True,
    )
    features = PricingFeatures.from_request(
        {"city_slug": "isfahan", "neighbourhood": "ملک‌شهر", "area": 170, "year_built": 1404}
    )

    result = comparable_stats(features, dataset=df, target_column="price_per_sqm_toman")

    assert result["sample_size"] == 10
    assert result["median_price_per_sqm_toman"] == 125_000_000
    filters = result["filters_applied"]
    assert filters["scope"] == "neighbourhood"
    assert filters["neighbourhood_applied"] is True
    assert filters["similarity"] == "strict"
    assert (filters["area_min"], filters["area_max"]) == (145, 195)
    assert filters["year_min"] == 1399
    # Capped at the newest listing rather than 1409.
    assert filters["year_max"] == 1404


def test_comparable_widens_band_before_leaving_neighbourhood():
    df = pd.concat(
        [
            _listings("ملک شهر", count=3, area=170.0, year=1404, pps=125_000_000),
            _listings("ملک شهر", count=7, area=135.0, year=1396, pps=110_000_000),
            _listings("ملک شهر", count=30, area=70.0, year=1380, pps=80_000_000),
        ],
        ignore_index=True,
    )
    features = PricingFeatures.from_request(
        {"city_slug": "isfahan", "neighbourhood": "ملک شهر", "area": 170, "year_built": 1404}
    )

    result = comparable_stats(features, dataset=df, target_column="price_per_sqm_toman")

    assert result["sample_size"] == 10
    assert result["filters_applied"]["scope"] == "neighbourhood"
    assert result["filters_applied"]["similarity"] == "wide"


def test_comparable_uses_nearby_listings_when_neighbourhood_has_none():
    """A neighbourhood with no listings of its own is covered by its surroundings."""
    df = pd.concat(
        [
            # ~0.6 km away, other neighbourhood
            _listings("مرداویج", count=9, area=120.0, year=1400, pps=110_000_000, lat=32.6255, lng=51.6650),
            # ~11 km away, must not leak in
            _listings("زینبیه", count=30, area=120.0, year=1400, pps=60_000_000, lat=32.7200, lng=51.6650),
        ],
        ignore_index=True,
    )
    features = PricingFeatures.from_request(
        {
            "city_slug": "isfahan",
            "neighbourhood": "محله بدون آگهی",
            "area": 120,
            "year_built": 1401,
            "location_lat": 32.6200,
            "location_long": 51.6650,
        }
    )

    result = comparable_stats(features, dataset=df, target_column="price_per_sqm_toman")

    assert result["sample_size"] == 9
    assert result["median_price_per_sqm_toman"] == 110_000_000
    filters = result["filters_applied"]
    assert filters["scope"] == "nearby"
    assert filters["neighbourhood_applied"] is False
    assert filters["similarity"] == "strict"
    assert filters["radius_km"] == 1.5


def test_comparable_skips_listings_without_build_year_when_matching_age():
    df = pd.concat(
        [
            _listings("ملک شهر", count=8, area=170.0, year=1402, pps=125_000_000),
            _listings("ملک شهر", count=8, area=170.0, year=None, pps=70_000_000),
        ],
        ignore_index=True,
    )
    features = PricingFeatures.from_request(
        {"city_slug": "isfahan", "neighbourhood": "ملک شهر", "area": 170, "year_built": 1404}
    )

    result = comparable_stats(features, dataset=df, target_column="price_per_sqm_toman")

    assert result["sample_size"] == 8
    assert result["median_price_per_sqm_toman"] == 125_000_000


def test_comparable_empty_when_neighbourhood_missing():
    df = pd.DataFrame(
        {
            "neighbourhood": ["جردن"] * 5,
            "area": [85.0] * 5,
            "rooms": [2] * 5,
            "price_per_sqm_toman": [200_000_000] * 5,
        }
    )
    features = PricingFeatures.from_request(
        {
            "city_slug": "tehran",
            "neighbourhood": "آجودانیه",
            "area": 85,
            "rooms": 2,
        }
    )
    result = comparable_stats(
        features,
        dataset=df,
        target_column="price_per_sqm_toman",
    )

    assert result["sample_size"] == 0
    assert result["mean_price_per_sqm_toman"] is None
    assert result["filters_applied"]["neighbourhood_applied"] is True
