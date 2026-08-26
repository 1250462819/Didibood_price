"""Market stats helpers."""
from __future__ import annotations

from data.neighbourhood_persian import district_slugs_for_neighbourhood


def test_district_slugs_for_persian_title(monkeypatch):
    monkeypatch.setattr(
        "data.neighbourhood_persian.district_to_persian_map",
        lambda city_slug: {"bahar": "بهار", "baharestan": "بهارستان"},
    )
    slugs = district_slugs_for_neighbourhood("tehran", "بهار")
    assert slugs == ["bahar"]


def test_district_slugs_for_slug_input():
    slugs = district_slugs_for_neighbourhood("tehran", "bahar")
    assert slugs == ["bahar"]
