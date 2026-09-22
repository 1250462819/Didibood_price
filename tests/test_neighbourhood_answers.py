"""The nightly answer store: what it serves, and when it refuses to.

The store exists so the city page reads a file instead of aggregating the
market on every visit. These tests hold it to the two promises that makes: a
stored answer is returned untouched for the request it answers, and it stops
being served once it is too old to trust.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from application.neighbourhoods import answers, read
from application.neighbourhoods.filters import NeighbourhoodFilters
from application.neighbourhoods.precompute import (
    PAGE_MONTHS,
    ROOMS_PLUS,
    page_filter_grid,
)
from application.neighbourhoods.read import cities_key, detail_key, overview_key
from config.settings import settings
from domain.model_key import ModelKey

TEHRAN = ModelKey(city_slug="tehran", property_type="apartment", purpose="sale")


@pytest.fixture(autouse=True)
def store_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the real artifacts directory from a test."""
    monkeypatch.setattr(settings, "DATA_CACHE_DIR", tmp_path)
    return tmp_path


def test_stored_answer_comes_back_whole() -> None:
    key = overview_key(TEHRAN, NeighbourhoodFilters(months=PAGE_MONTHS))
    payload = {"summary": {"mean": 123.5}, "neighbourhoods": [{"title": "پونک"}]}

    answers.store(key, payload)

    assert answers.load(key) == payload


def test_the_same_filters_in_any_order_are_one_answer() -> None:
    first = overview_key(TEHRAN, NeighbourhoodFilters(months=PAGE_MONTHS, has_parking=True, has_elevator=True))
    second = overview_key(TEHRAN, NeighbourhoodFilters(months=PAGE_MONTHS, has_elevator=True, has_parking=True))

    assert first.digest() == second.digest()


def test_a_different_filter_is_a_different_answer() -> None:
    plain = overview_key(TEHRAN, NeighbourhoodFilters(months=PAGE_MONTHS))
    parking = overview_key(TEHRAN, NeighbourhoodFilters(months=PAGE_MONTHS, has_parking=True))
    rooms = overview_key(TEHRAN, NeighbourhoodFilters(months=PAGE_MONTHS, min_rooms=2, max_rooms=2))

    assert len({plain.digest(), parking.digest(), rooms.digest()}) == 3


def test_cities_do_not_share_answers() -> None:
    mashhad = ModelKey(city_slug="mashhad", property_type="apartment", purpose="sale")
    filters = NeighbourhoodFilters(months=PAGE_MONTHS)

    assert overview_key(TEHRAN, filters).digest() != overview_key(mashhad, filters).digest()


def test_a_neighbourhood_is_found_however_its_name_is_spelled() -> None:
    """The map and the crawl disagree about ya and kaf; the reader should not notice."""
    filters = NeighbourhoodFilters(months=PAGE_MONTHS)

    assert detail_key(TEHRAN, filters, "پونك").digest() == detail_key(TEHRAN, filters, "پونک").digest()


def test_an_answer_older_than_the_limit_is_not_served(store_dir: Path) -> None:
    key = cities_key(NeighbourhoodFilters(months=PAGE_MONTHS), purpose="sale", property_type="apartment")
    path = answers.store(key, {"cities": []})

    stale = datetime.now(tz=timezone.utc) - timedelta(
        hours=settings.NEIGHBOURHOOD_ANSWER_MAX_AGE_HOURS + 1
    )
    os.utime(path, (stale.timestamp(), stale.timestamp()))

    assert answers.load(key) is None


def test_a_missing_answer_is_simply_absent() -> None:
    assert answers.load(overview_key(TEHRAN, NeighbourhoodFilters(months=PAGE_MONTHS))) is None


def test_a_corrupt_answer_is_ignored_rather_than_raised(store_dir: Path) -> None:
    key = overview_key(TEHRAN, NeighbourhoodFilters(months=PAGE_MONTHS))
    path = key.path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not gzip")

    assert answers.load(key) is None


def test_the_grid_is_exactly_what_the_page_can_ask() -> None:
    """Four toggles, five room choices — every combination, and nothing else."""
    grid = list(page_filter_grid())

    assert len(grid) == 16 * 5
    assert len({overview_key(TEHRAN, filters).digest() for filters in grid}) == len(grid)
    assert all(filters.months == PAGE_MONTHS for filters in grid)

    plus = [filters for filters in grid if filters.min_rooms == ROOMS_PLUS]
    assert plus and all(filters.max_rooms is None for filters in plus), "«۴+» must stay open-ended"

    exact = [filters for filters in grid if filters.min_rooms == 2]
    assert exact and all(filters.max_rooms == 2 for filters in exact)


def test_the_endpoint_serves_the_stored_answer_without_computing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filters = NeighbourhoodFilters(months=PAGE_MONTHS)
    answers.store(overview_key(TEHRAN, filters), {"marker": "last night"})

    def explode(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("a stored answer must not be recomputed")

    monkeypatch.setattr(read, "compute_overview", explode)

    assert read.get_overview(TEHRAN, filters) == {"marker": "last night"}


def test_a_filter_nobody_precomputed_is_still_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The store is a shortcut, never a gate: an unusual request computes live."""
    unusual = NeighbourhoodFilters(months=PAGE_MONTHS, max_building_age=5)
    monkeypatch.setattr(read, "compute_overview", lambda *args, **kwargs: {"marker": "computed"})

    assert read.get_overview(TEHRAN, unusual) == {"marker": "computed"}
