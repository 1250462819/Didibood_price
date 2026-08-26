"""Feature + target tests."""
from __future__ import annotations

from unittest.mock import patch

from domain.features import PricingFeatures, row_from_divar_join
from domain.target import prediction_bounds_sale, price_per_sqm_toman, total_price_toman


def test_price_per_sqm_prefers_unit():
    y, src = price_per_sqm_toman(
        price_total=8_500_000_000,
        price_per_unit=141_000_000,
        area_sqm=85,
    )
    assert y == 141_000_000
    assert src == "price_per_unit"


def test_price_per_sqm_computed_fallback():
    y, src = price_per_sqm_toman(
        price_total=8_500_000_000,
        price_per_unit=None,
        area_sqm=110,
    )
    assert src == "computed"
    assert abs(y - 8_500_000_000 / 110) < 1


def test_row_from_divar_join():
    row = row_from_divar_join(
        {
            "post_token": "abc123",
            "category_slug": "apartment-sell",
            "listing_type": "BUY",
            "city_slug": "tehran",
            "district": "جردن",
            "location": "جردن، تهران",
            "price_total": 12_000_000_000,
            "price_per_unit": 141_000_000,
            "geo_lat": 35.77,
            "geo_lon": 51.41,
            "attributes": [
                {"key": "متراژ", "value": "85"},
                {"key": "اتاق", "value": "2"},
                {"key": "ساخت", "value": "1398"},
                {"key": "طبقه", "value": "3 از 5"},
                {"key": "پارکینگ", "value": "دارد"},
            ],
        }
    )
    assert row is not None
    assert row["neighbourhood"] == "جردن"
    assert row["area"] == 85.0
    assert row["price_per_sqm_toman"] == 141_000_000


def test_row_from_divar_join_maps_slug_to_persian():
    with patch(
        "data.neighbourhood_persian.district_to_persian_map",
        return_value={"jordan": "جردن"},
    ):
        row = row_from_divar_join(
            {
                "post_token": "abc124",
                "category_slug": "apartment-sell",
                "listing_type": "BUY",
                "city_slug": "tehran",
                "district": "jordan",
                "location": "ignored",
                "price_total": 12_000_000_000,
                "price_per_unit": 141_000_000,
                "attributes": [{"key": "متراژ", "value": "85"}],
            }
        )
    assert row is not None
    assert row["neighbourhood"] == "جردن"


def test_pricing_features_from_request():
    f = PricingFeatures.from_request(
        {
            "city_slug": "tehran",
            "neighbourhood": "ونک",
            "area": 90,
            "rooms": 2,
            "year_built": 1400,
            "parking": True,
        }
    )
    row = f.to_model_row()
    assert row["neighbourhood"] == "ونک"
    assert row["has_parking"] == 1
    assert total_price_toman(150_000_000, 90) == 13_500_000_000


def test_prediction_bounds_from_mape():
    bounds = prediction_bounds_sale(360_994_913, 85, mape=0.1875)
    assert bounds["low_price_per_sqm_toman"] < 360_994_913
    assert bounds["high_price_per_sqm_toman"] > 360_994_913
    assert bounds["low_total_price_toman"] == total_price_toman(
        bounds["low_price_per_sqm_toman"], 85
    )
    assert bounds["margin_method"] == "test_mape"
