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


def test_comparable_ignores_area_and_rooms():
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
    assert result["filters_applied"]["recent_days"] == 30
    assert "area" not in result["filters_applied"]
    assert "rooms" not in result["filters_applied"]


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
