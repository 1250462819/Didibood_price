"""Rent target + feature tests."""
from __future__ import annotations

import pandas as pd

from data.clean import drop_rent_cap_violations, prepare_rent_training_frame
from domain.features import row_from_divar_join
from domain.target import (
    RENT_TARGET_COLUMN,
    equivalent_deposit_toman,
    prediction_bounds_rent,
    split_rent_on_conversion_line,
)


def test_equivalent_deposit_monthly_only():
    y, src = equivalent_deposit_toman(
        0,
        30_000_000,
        ratio_deposit_toman=10_000_000,
        ratio_monthly_toman=300_000,
    )
    assert src == "monthly_only"
    assert abs(y - 1_000_000_000) < 1


def test_equivalent_deposit_mixed():
    y, src = equivalent_deposit_toman(
        200_000_000,
        15_000_000,
        ratio_deposit_toman=10_000_000,
        ratio_monthly_toman=300_000,
    )
    assert src == "deposit_and_monthly"
    assert abs(y - (200_000_000 + 15_000_000 * (10 / 0.3))) < 1


def test_row_from_divar_rent():
    row = row_from_divar_join(
        {
            "post_token": "rent1",
            "category_slug": "apartment-rent",
            "listing_type": "RENT",
            "city_slug": "tehran",
            "district": "جردن",
            "price_total": 200_000_000,
            "price_per_unit": 15_000_000,
            "geo_lat": 35.77,
            "geo_lon": 51.41,
            "attributes": [
                {"key": "متراژ", "value": "85"},
                {"key": "اتاق", "value": "2"},
            ],
        }
    )
    assert row is not None
    assert row["equivalent_deposit_toman"] > 200_000_000
    assert row["deposit_toman"] == 200_000_000
    assert row["monthly_rent_toman"] == 15_000_000


def test_row_from_divar_rent_remapped_deposit_as_monthly():
    row = row_from_divar_join(
        {
            "post_token": "rent2",
            "category_slug": "apartment-rent",
            "listing_type": "RENT",
            "city_slug": "isfahan",
            "district": "malekshahr",
            "price_total": 15_000_000,
            "price_per_unit": 0,
            "geo_lat": 32.65,
            "geo_lon": 51.67,
            "attributes": [
                {"key": "متراژ", "value": "90"},
                {"key": "اتاق", "value": "2"},
            ],
        }
    )
    assert row is not None
    assert row["deposit_toman"] == 0
    assert row["monthly_rent_toman"] == 15_000_000
    assert row["y_source"] == "remapped_deposit_as_monthly"
    assert abs(row["equivalent_deposit_toman"] - 500_000_000) < 1


def test_split_rent_full_deposit_default():
    dep, mon = split_rent_on_conversion_line(
        700_000_000,
        ratio_deposit_toman=10_000_000,
        ratio_monthly_toman=300_000,
    )
    assert dep == 700_000_000
    assert mon == 0


def test_split_rent_with_preferred_deposit():
    dep, mon = split_rent_on_conversion_line(
        700_000_000,
        ratio_deposit_toman=10_000_000,
        ratio_monthly_toman=300_000,
        preferred_deposit_toman=200_000_000,
    )
    assert dep == 200_000_000
    assert mon > 0


def test_prediction_bounds_rent():
    bounds = prediction_bounds_rent(700_000_000, mape=0.2)
    assert bounds["low_equivalent_deposit_toman"] == 560_000_000
    assert bounds["high_equivalent_deposit_toman"] == 840_000_000


def test_drop_rent_cap_violations():
    df = pd.DataFrame(
        [
            {
                RENT_TARGET_COLUMN: 500_000_000,
                "deposit_toman": 200_000_000,
                "monthly_rent_toman": 10_000_000,
                "area": 80,
            },
            {
                RENT_TARGET_COLUMN: 15_000_000_000_000,
                "deposit_toman": 15_000_000_000_000,
                "monthly_rent_toman": 0,
                "area": 90,
            },
            {
                RENT_TARGET_COLUMN: 5_000,
                "deposit_toman": 0,
                "monthly_rent_toman": 150,
                "area": 70,
            },
        ]
    )
    out = drop_rent_cap_violations(df)
    assert len(out) == 1
    assert out.iloc[0][RENT_TARGET_COLUMN] == 500_000_000


def test_prepare_rent_training_frame_caps():
    df = pd.DataFrame(
        [
            {
                RENT_TARGET_COLUMN: 400_000_000,
                "deposit_toman": 100_000_000,
                "monthly_rent_toman": 9_000_000,
                "area": 85,
            },
            {
                RENT_TARGET_COLUMN: 12_000_000_000,
                "deposit_toman": 12_000_000_000,
                "monthly_rent_toman": 0,
                "area": 100,
            },
        ]
    )
    out = prepare_rent_training_frame(df)
    assert len(out) == 1
