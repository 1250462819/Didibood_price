"""Persian neighbourhood helpers and mapping tests."""
from __future__ import annotations

from unittest.mock import patch

from data.neighbourhood_persian import (
    _is_neighbourhood_label,
    clean_location_label,
    extract_neighbourhood_label,
    normalize_persian_neighbourhood,
    resolve_request_neighbourhood,
    resolve_training_neighbourhood,
)
from domain.neighbourhood import is_persian_neighbourhood, is_slug_like


def test_normalize_persian_neighbourhood():
    assert normalize_persian_neighbourhood("  ونک ") == "ونک"
    assert clean_location_label("جردن، تهران") == "جردن"


def test_is_slug_like():
    assert is_slug_like("jordan")
    assert not is_slug_like("جردن")


def test_is_persian_neighbourhood():
    assert is_persian_neighbourhood("جردن")
    assert not is_persian_neighbourhood("jordan")


def test_resolve_request_neighbourhood_exact():
    known = {"جردن", "ونک"}
    assert resolve_request_neighbourhood("جردن", known) == "جردن"
    assert resolve_request_neighbourhood("  ونک ", known) == "ونک"


def test_extract_neighbourhood_label_from_agency_string():
    assert extract_neighbourhood_label("آژانس املاک اوپال در پونک") == "پونک"


def test_is_neighbourhood_label_rejects_noise():
    assert not _is_neighbourhood_label("دقایقی پیش در تهران")


def test_resolve_training_neighbourhood_uses_persian_district():
    assert (
        resolve_training_neighbourhood(
            city_slug="tehran",
            district="جردن",
            location=None,
        )
        == "جردن"
    )


def test_resolve_training_neighbourhood_maps_slug():
    with patch(
        "data.neighbourhood_persian.district_to_persian_map",
        return_value={"jordan": "جردن"},
    ):
        assert (
            resolve_training_neighbourhood(
                city_slug="tehran",
                district="jordan",
                location=None,
            )
            == "جردن"
        )


def test_resolve_training_neighbourhood_falls_back_to_location():
    assert (
        resolve_training_neighbourhood(
            city_slug="tehran",
            district=None,
            location="آجودانیه، تهران",
        )
        == "آجودانیه"
    )
