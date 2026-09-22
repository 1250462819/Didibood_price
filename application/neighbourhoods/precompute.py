"""The nightly job that answers the city page's questions in advance.

The page asks a small, known set of things: one overview per combination of the
filters its panel offers, one detail per neighbourhood, and the city list. Each
of those costs a second or more to compute from the listing frame, and none of
them changes until the next crawl — so they are computed once here, after the
frames are rebuilt, and written to the answer store (`answers.py`).

The grid is deliberately exactly what the page can ask for. Anything else — a
filter only the API exposes, a rent query, an unusual month window — is not
precomputed and is still computed per request.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Iterator

from application.neighbourhoods import answers
from application.neighbourhoods.dataset import refresh_analytics_frame
from application.neighbourhoods.filters import NeighbourhoodFilters
from application.neighbourhoods.read import (
    _loaded,
    cities_key,
    compute_cities,
    compute_detail,
    compute_overview,
    detail_key,
    overview_key,
)
from application.neighbourhoods.stats import neighbourhood_rows
from domain.model_key import SUPPORTED_CITIES, ModelKey

logger = logging.getLogger(__name__)

#: What the page's own requests carry. The site never asks for another window.
PAGE_MONTHS = 6

#: The four toggles under the sliders, in the page's order.
AMENITIES = ("has_parking", "has_elevator", "has_balcony", "has_storage")

#: «همه» plus the four room buttons; the last is open-ended («۴+»).
ROOM_CHOICES = (None, 1, 2, 3, 4)
ROOMS_PLUS = 4


def page_filter_grid() -> Iterator[NeighbourhoodFilters]:
    """Every combination the panel can produce: amenities × rooms."""
    subsets = [subset for size in range(len(AMENITIES) + 1) for subset in combinations(AMENITIES, size)]
    for subset, rooms in product(subsets, ROOM_CHOICES):
        wanted = {name: True for name in subset}
        yield NeighbourhoodFilters(
            months=PAGE_MONTHS,
            min_rooms=rooms,
            max_rooms=rooms if rooms is not None and rooms < ROOMS_PLUS else None,
            **wanted,
        )


@dataclass
class PrecomputeReport:
    overviews: int = 0
    details: int = 0
    cities: int = 0
    failures: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def line(self) -> str:
        return (
            f"overviews={self.overviews} details={self.details} cities={self.cities} "
            f"failures={len(self.failures)} in {self.seconds:.0f}s"
        )


def _precompute_city(
    city: str,
    *,
    purpose: str,
    property_type: str,
    with_details: bool,
    report: PrecomputeReport,
) -> None:
    key = ModelKey(city_slug=city, property_type=property_type, purpose=purpose)

    for filters in page_filter_grid():
        try:
            answers.store(overview_key(key, filters), compute_overview(key, filters))
            report.overviews += 1
        except Exception as exc:  # one filter must not stop the night's work
            report.failures.append(f"overview {city} {filters.as_dict()}: {exc}")
            logger.exception("precompute overview failed city=%s", city)

    if not with_details:
        return

    # Every detail shares one filtered frame and one ranking; computing them per
    # neighbourhood is what makes the live endpoint slow.
    base = NeighbourhoodFilters(months=PAGE_MONTHS)
    try:
        prepared = _loaded(key, base)
        loaded, filtered = prepared
        if filtered.empty:
            return
        rows = neighbourhood_rows(
            filtered,
            target_column=loaded.target_column,
            purpose=purpose,
            months=base.months,
        )
    except Exception as exc:
        report.failures.append(f"detail frame {city}: {exc}")
        logger.exception("precompute detail frame failed city=%s", city)
        return

    for row in rows:
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        try:
            payload = compute_detail(
                key, base, neighbourhood=title, prepared=prepared, city_rows=rows
            )
            if payload is None:
                continue
            answers.store(detail_key(key, base, title), payload)
            report.details += 1
        except Exception as exc:
            report.failures.append(f"detail {city} {title}: {exc}")
            logger.exception("precompute detail failed city=%s title=%s", city, title)


def run_precompute(
    cities: list[str] | None = None,
    *,
    purpose: str = "sale",
    property_type: str = "apartment",
    with_details: bool = True,
    refresh_frames: bool = False,
) -> PrecomputeReport:
    """Fill the answer store. Returns what was written, for the log line.

    `refresh_frames` re-extracts the listing frames from the crawl database
    first; the nightly unit already does that in its own step, so the default
    is off.
    """
    started = time.monotonic()
    report = PrecomputeReport()
    wanted = cities or list(SUPPORTED_CITIES)

    for city in wanted:
        if refresh_frames:
            try:
                refresh_analytics_frame(
                    ModelKey(city_slug=city, property_type=property_type, purpose=purpose)
                )
            except Exception as exc:
                report.failures.append(f"refresh {city}: {exc}")
                logger.exception("precompute refresh failed city=%s", city)
        _precompute_city(
            city,
            purpose=purpose,
            property_type=property_type,
            with_details=with_details,
            report=report,
        )

    # The country-wide list the map shows before a city is picked.
    try:
        base = NeighbourhoodFilters(months=PAGE_MONTHS)
        answers.store(
            cities_key(base, purpose=purpose, property_type=property_type),
            compute_cities(base, purpose=purpose, property_type=property_type),
        )
        report.cities += 1
    except Exception as exc:
        report.failures.append(f"cities: {exc}")
        logger.exception("precompute cities failed")

    report.seconds = time.monotonic() - started
    logger.info("precompute done %s", report.line())
    return report
