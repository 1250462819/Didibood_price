"""The numbers behind the neighbourhood analysis page.

Every assertion here is about a claim the page makes to a reader: this
neighbourhood is dearer than that one, this month moved, this filter narrowed
the population. A synthetic frame keeps those claims checkable — the real crawl
frame changes every day.
"""
from __future__ import annotations

import pandas as pd
import pytest

from application.neighbourhoods.filters import (
    NeighbourhoodFilters,
    apply_filters,
    month_labels,
)
from application.neighbourhoods.stats import (
    city_summary,
    neighbourhood_detail,
    neighbourhood_rows,
    rent_yield_pct,
    similar_neighbourhoods,
    trend_pct,
)
from domain.target import SALE_TARGET_COLUMN

TARGET = SALE_TARGET_COLUMN


def _rows(spec: list[tuple[str, float, float, int, int]]) -> pd.DataFrame:
    """(neighbourhood, price/m², area, months_ago, rooms) → a frame like the real one."""
    now = pd.Timestamp.now(tz="Asia/Tehran")
    records = []
    for index, (title, price, area, months_ago, rooms) in enumerate(spec):
        seen = (now.tz_localize(None).to_period("M") - months_ago).to_timestamp(how="start")
        records.append(
            {
                "neighbourhood": title,
                TARGET: price,
                "area": area,
                "rooms": rooms,
                "building_age": 8,
                "has_parking": 1,
                "has_elevator": 1,
                "has_storage": 0,
                "has_balcony": 0,
                "location_lat": 35.7 + index * 0.001,
                "location_long": 51.4 + index * 0.001,
                "price_total_toman": price * area,
                "post_token": f"t{index}",
                "first_seen_at": seen.tz_localize("Asia/Tehran").tz_convert("UTC"),
                "last_seen_at": now.tz_convert("UTC"),
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def frame() -> pd.DataFrame:
    return _rows(
        [("الهیه", 500_000_000, 120, 1, 3)] * 20
        + [("پونک", 100_000_000, 80, 1, 2)] * 20
        + [("تجریش", 300_000_000, 100, 1, 2)] * 3
    )


def test_neighbourhoods_are_ranked_dearest_first(frame):
    rows = neighbourhood_rows(frame, target_column=TARGET, purpose="sale", months=6)
    assert [row["title"] for row in rows] == ["الهیه", "تجریش", "پونک"]
    assert [row["rank"] for row in rows] == [1, 2, 3]


def test_a_thin_neighbourhood_is_published_but_flagged(frame):
    """Hiding it would shrink the map; publishing it unmarked would mislead."""
    rows = {row["title"]: row for row in neighbourhood_rows(frame, target_column=TARGET, purpose="sale", months=6)}
    assert rows["تجریش"]["sample_size"] == 3
    assert rows["تجریش"]["low_sample"] is True
    assert rows["الهیه"]["low_sample"] is False


def test_the_median_is_not_dragged_by_one_penthouse():
    frame = _rows([("پونک", 100_000_000, 80, 1, 2)] * 20 + [("پونک", 5_000_000_000, 400, 1, 5)])
    rows = neighbourhood_rows(frame, target_column=TARGET, purpose="sale", months=6)
    assert rows[0]["median"] == 100_000_000
    assert rows[0]["p75"] == 100_000_000


def test_percentile_says_how_many_neighbourhoods_are_cheaper(frame):
    rows = neighbourhood_rows(frame, target_column=TARGET, purpose="sale", months=6)
    assert rows[0]["percentile"] == 100.0
    assert rows[-1]["percentile"] == pytest.approx(33.3, abs=0.1)


def test_a_month_too_thin_to_publish_keeps_its_count_but_not_a_price():
    frame = _rows([("پونک", 100_000_000, 80, 1, 2)] * 2)
    summary = city_summary(frame, target_column=TARGET, purpose="sale", months=3, neighbourhood_count=1)
    thin = next(point for point in summary["monthly"] if point["sample_size"] == 2)
    assert thin["median"] is None


def test_the_trend_compares_the_first_and_last_month_with_enough_listings():
    frame = _rows(
        [("پونک", 100_000_000, 80, 2, 2)] * 6 + [("پونک", 120_000_000, 80, 0, 2)] * 6
    )
    summary = city_summary(frame, target_column=TARGET, purpose="sale", months=3, neighbourhood_count=1)
    assert summary["trend_pct"] == pytest.approx(20.0)


def test_a_flat_market_reports_zero_not_none():
    frame = _rows([("پونک", 100_000_000, 80, 2, 2)] * 6 + [("پونک", 100_000_000, 80, 0, 2)] * 6)
    assert trend_pct(
        city_summary(frame, target_column=TARGET, purpose="sale", months=3, neighbourhood_count=1)["monthly"]
    ) == 0.0


def test_the_month_window_matches_what_was_asked_for():
    assert len(month_labels(6)) == 6
    assert len(month_labels(12)) == 12


def test_an_area_filter_narrows_every_number(frame):
    filtered = apply_filters(
        frame, NeighbourhoodFilters(months=6, min_area=100), purpose="sale", target_column=TARGET
    )
    assert set(filtered["neighbourhood"]) == {"الهیه", "تجریش"}


def test_a_budget_filter_is_about_the_whole_price_not_the_metre(frame):
    """A 120m flat at 500M/m² is out of a 12B budget; an 80m at 100M/m² is not."""
    filtered = apply_filters(
        frame,
        NeighbourhoodFilters(months=6, max_budget_toman=12_000_000_000),
        purpose="sale",
        target_column=TARGET,
    )
    assert set(filtered["neighbourhood"]) == {"پونک"}


def test_a_listing_with_no_age_is_dropped_by_an_age_filter():
    frame = _rows([("پونک", 100_000_000, 80, 1, 2)] * 4)
    frame.loc[0, "building_age"] = None
    filtered = apply_filters(
        frame, NeighbourhoodFilters(months=6, max_building_age=10), purpose="sale", target_column=TARGET
    )
    assert len(filtered) == 3


def test_listings_older_than_the_window_are_left_out():
    frame = _rows([("پونک", 100_000_000, 80, 0, 2)] * 3 + [("پونک", 100_000_000, 80, 8, 2)] * 3)
    filtered = apply_filters(frame, NeighbourhoodFilters(months=3), purpose="sale", target_column=TARGET)
    assert len(filtered) == 3


def test_detail_answers_with_the_breakdowns_the_panel_shows(frame):
    rows = neighbourhood_rows(frame, target_column=TARGET, purpose="sale", months=6)
    detail = neighbourhood_detail(
        frame[frame["neighbourhood"] == "الهیه"],
        title="الهیه",
        target_column=TARGET,
        purpose="sale",
        filters=NeighbourhoodFilters(months=6),
        rows=rows,
    )
    assert detail["rank"] == 1
    assert detail["price"]["median"] == 500_000_000
    assert detail["by_rooms"][0]["rooms"] == 3
    assert detail["by_area"]
    assert [item["title"] for item in detail["similar"]] == ["پونک"]


def test_similar_leaves_out_the_neighbourhood_itself_and_thin_ones(frame):
    rows = neighbourhood_rows(frame, target_column=TARGET, purpose="sale", months=6)
    titles = [item["title"] for item in similar_neighbourhoods(rows, "الهیه")]
    assert "الهیه" not in titles
    assert "تجریش" not in titles  # three listings — too thin to recommend


def test_gross_yield_turns_a_deposit_into_rent_before_dividing():
    rent = pd.DataFrame(
        [
            {"area": 100.0, "monthly_rent_toman": 0.0, "deposit_toman": 1_000_000_000.0},
            {"area": 100.0, "monthly_rent_toman": 0.0, "deposit_toman": 1_000_000_000.0},
        ]
    )
    # 1B deposit → 30M/month at the site ratio → 300K per m² per month.
    result = rent_yield_pct(100_000_000, rent)
    assert result["monthly_rent_per_sqm_toman"] == pytest.approx(300_000)
    assert result["gross_yield_pct"] == pytest.approx(3.6)


def test_yield_is_absent_rather_than_guessed_when_there_is_no_rent_data():
    assert rent_yield_pct(100_000_000, None) is None
    assert rent_yield_pct(100_000_000, pd.DataFrame()) is None


def test_a_half_crawled_month_is_counted_but_not_priced():
    """The crawl began mid-July; six days of listings against a full August read
    as a fall in prices that never happened."""
    frame = _rows(
        [("پونک", 150_000_000, 80, 2, 2)] * 3  # the stub month
        + [("پونک", 100_000_000, 80, 1, 2)] * 60
        + [("پونک", 100_000_000, 80, 0, 2)] * 60
    )
    summary = city_summary(frame, target_column=TARGET, purpose="sale", months=3, neighbourhood_count=1)
    stub = summary["monthly"][0]
    assert stub["sample_size"] == 3
    assert stub["median"] is None
    assert summary["trend_pct"] == 0.0


def test_a_full_month_still_counts_even_if_smaller_than_its_neighbour():
    """Normal month-to-month variation must not be mistaken for a partial crawl."""
    frame = _rows(
        [("پونک", 100_000_000, 80, 1, 2)] * 30 + [("پونک", 110_000_000, 80, 0, 2)] * 60
    )
    summary = city_summary(frame, target_column=TARGET, purpose="sale", months=2, neighbourhood_count=1)
    assert [point["median"] for point in summary["monthly"]] == [100_000_000, 110_000_000]
    assert summary["trend_pct"] == pytest.approx(10.0)


def test_a_city_row_carries_what_the_first_map_level_needs():
    """Before a city is picked the map has one mark per city, not per neighbourhood."""
    from application.neighbourhoods.read import CITY_LABELS_FA

    assert CITY_LABELS_FA["tehran"] == "تهران"
    for slug in ("tehran", "mashhad", "isfahan"):
        assert slug in CITY_LABELS_FA


def test_the_country_headline_is_pooled_not_averaged(monkeypatch):
    """Three cities' medians averaged is not the median of the three cities."""
    import pandas as pd

    from application.neighbourhoods import read as read_module

    cheap = _rows([("محلهٔ ارزان", 100_000_000, 80, 1, 2)] * 30)
    dear = _rows([("محلهٔ گران", 500_000_000, 80, 1, 2)] * 10)

    frames = {"tehran": dear, "mashhad": cheap, "isfahan": cheap}

    class _Loaded:
        def __init__(self, frame):
            self.frame = frame
            self.target_column = TARGET
            self.built_at = pd.Timestamp.now('UTC').to_pydatetime()
            self.stale = False

    def fake_loaded(key, _filters):
        frame = frames[key.city_slug]
        return _Loaded(frame), frame

    monkeypatch.setattr(read_module, "_loaded", fake_loaded)
    payload = read_module.get_cities(NeighbourhoodFilters(months=6), purpose="sale")

    total = payload["total"]
    # 70 cheap listings against 10 dear ones: the pooled median is the cheap one,
    # while the mean of the three city medians would be far above it.
    assert total["median"] == 100_000_000
    assert total["sample_size"] == 70
    assert total["city_count"] == 3
    assert total["spread_ratio"] == 5.0


# --- «میانگین»: trimmed means and what an amenity is worth -------------------

from application.neighbourhoods.stats import (  # noqa: E402
    amenity_split,
    grouped_trimmed_mean,
    like_for_like_premium,
    trimmed_mean,
)
from schemas.neighbourhoods import (  # noqa: E402
    NeighbourhoodDetailResponse,
    NeighbourhoodOverviewResponse,
)


def test_the_average_ignores_the_extreme_tenth_at_each_end():
    """The page says «میانگین», so it is a mean — but one penthouse must not move it."""
    values = pd.Series([100.0] * 9 + [10_000.0])
    assert trimmed_mean(values) == pytest.approx(100.0)


def test_a_trimmed_mean_of_nothing_is_none():
    assert trimmed_mean(pd.Series([], dtype="float64")) is None


def test_grouped_trimmed_mean_is_per_neighbourhood():
    frame = pd.DataFrame(
        {"neighbourhood": ["a"] * 10 + ["b"] * 10, "v": [10.0] * 9 + [1000.0] + [20.0] * 10}
    )
    means = grouped_trimmed_mean(frame, "neighbourhood", "v")
    assert means["a"] == pytest.approx(10.0)
    assert means["b"] == pytest.approx(20.0)


def _amenity_frame(with_price: float, without_price: float, n_with: int, n_without: int, title: str = "x"):
    rows = [{"neighbourhood": title, TARGET: with_price, "has_parking": 1}] * n_with
    rows += [{"neighbourhood": title, TARGET: without_price, "has_parking": 0}] * n_without
    return pd.DataFrame(rows)


def test_amenity_split_reports_both_sides_and_the_premium():
    split = amenity_split(_amenity_frame(120.0, 100.0, 8, 6), "has_parking", TARGET)
    assert split["with_sample"] == 8
    assert split["without_sample"] == 6
    assert split["mean_with"] == pytest.approx(120.0)
    assert split["mean_without"] == pytest.approx(100.0)
    assert split["premium_pct"] == pytest.approx(20.0)
    assert split["with_share"] == pytest.approx(8 / 14)


def test_amenity_split_withholds_a_premium_below_five_on_either_side():
    split = amenity_split(_amenity_frame(120.0, 100.0, 8, 4), "has_parking", TARGET)
    assert split["premium_pct"] is None
    assert split["mean_without"] == pytest.approx(100.0)


def test_like_for_like_premium_compares_inside_each_neighbourhood():
    """Parking is common in the dear area and rare in the cheap one. Pooled, that
    reads as parking doubling the price; inside each area it adds 10%."""
    frame = pd.concat(
        [
            _amenity_frame(330.0, 300.0, 20, 5, title="dear"),
            _amenity_frame(110.0, 100.0, 5, 20, title="cheap"),
        ],
        ignore_index=True,
    )
    assert amenity_split(frame, "has_parking", TARGET)["premium_pct"] > 90
    assert like_for_like_premium(frame, "has_parking", TARGET) == pytest.approx(10.0)


def test_overview_rows_carry_means_and_amenity_splits(frame):
    rows = {
        row["title"]: row
        for row in neighbourhood_rows(frame, target_column=TARGET, purpose="sale", months=6)
    }
    elahieh = rows["الهیه"]
    assert elahieh["mean"] == pytest.approx(500_000_000)
    assert elahieh["mean_area"] == pytest.approx(120)
    assert elahieh["mean_budget_toman"] == pytest.approx(500_000_000 * 120)
    assert set(elahieh["parking"]) >= {"with_share", "mean_with", "mean_without", "premium_pct"}
    assert elahieh["parking"]["with_share"] == pytest.approx(1.0)
    # Every listing in the fixture has parking: nothing to compare against.
    assert elahieh["parking"]["premium_pct"] is None


def test_city_summary_carries_means_and_a_like_for_like_parking_premium(frame):
    summary = city_summary(frame, target_column=TARGET, purpose="sale", months=6, neighbourhood_count=3)
    assert summary["mean"] is not None
    assert summary["mean_area"] is not None
    assert summary["mean_budget_toman"] is not None
    assert "like_for_like_pct" in summary["parking"]
    assert "like_for_like_pct" in summary["elevator"]


def test_detail_carries_means_on_price_bands_and_amenities(frame):
    subset = frame[frame["neighbourhood"] == "الهیه"]
    rows = neighbourhood_rows(frame, target_column=TARGET, purpose="sale", months=6)
    detail = neighbourhood_detail(
        subset,
        title="الهیه",
        target_column=TARGET,
        purpose="sale",
        filters=NeighbourhoodFilters(months=6),
        rows=rows,
    )
    assert detail["price"]["mean"] == pytest.approx(500_000_000)
    assert detail["price"]["mean_area"] == pytest.approx(120)
    assert detail["price"]["mean_budget_toman"] == pytest.approx(500_000_000 * 120)
    assert detail["by_area"] and all("mean" in band for band in detail["by_area"])
    assert detail["by_rooms"] and all("mean" in band for band in detail["by_rooms"])


def test_amenity_rows_carry_mean_based_premiums():
    frame = _rows([("پونک", 120_000_000, 80, 1, 2)] * 8 + [("پونک", 100_000_000, 80, 1, 2)] * 6)
    frame.loc[8:, "has_parking"] = 0
    rows = neighbourhood_rows(frame, target_column=TARGET, purpose="sale", months=6)
    detail = neighbourhood_detail(
        frame,
        title="پونک",
        target_column=TARGET,
        purpose="sale",
        filters=NeighbourhoodFilters(months=6),
        rows=rows,
    )
    parking = next(row for row in detail["amenities"] if row["key"] == "has_parking")
    assert parking["mean_with"] == pytest.approx(120_000_000)
    assert parking["mean_without"] == pytest.approx(100_000_000)
    assert parking["mean_premium_pct"] == pytest.approx(20.0)


def test_the_response_models_keep_the_new_fields(frame):
    """FastAPI drops any field a response model does not declare — silently."""
    rows = neighbourhood_rows(frame, target_column=TARGET, purpose="sale", months=6)
    summary = city_summary(frame, target_column=TARGET, purpose="sale", months=6, neighbourhood_count=3)
    meta = {
        "city": "tehran",
        "purpose": "sale",
        "property_type": "apartment",
        "months": 6,
        "metric": "price_per_sqm_toman",
        "filters": {},
        "data_built_at": "2026-09-21T00:00:00+00:00",
        "coverage_start": "2026-07",
        "total_listings": len(frame),
    }
    dumped = NeighbourhoodOverviewResponse(**meta, summary=summary, neighbourhoods=rows).model_dump(
        by_alias=True
    )
    assert dumped["neighbourhoods"][0]["mean"] is not None
    assert dumped["neighbourhoods"][0]["parking"]["with_share"] is not None
    assert "like_for_like_pct" in dumped["summary"]["parking"]

    detail = neighbourhood_detail(
        frame[frame["neighbourhood"] == "الهیه"],
        title="الهیه",
        target_column=TARGET,
        purpose="sale",
        filters=NeighbourhoodFilters(months=6),
        rows=rows,
    )
    dumped_detail = NeighbourhoodDetailResponse(**meta, neighbourhood=detail).model_dump(by_alias=True)
    assert dumped_detail["neighbourhood"]["price"]["mean"] is not None
    assert "mean" in dumped_detail["neighbourhood"]["by_area"][0]


def test_a_city_row_carries_its_average(monkeypatch):
    from application.neighbourhoods import read as read_module

    frames = {
        "tehran": _rows([("الف", 300_000_000, 80, 1, 2)] * 12),
        "mashhad": _rows([("ب", 100_000_000, 80, 1, 2)] * 12),
        "isfahan": _rows([("ج", 150_000_000, 80, 1, 2)] * 12),
    }

    class _Loaded:
        def __init__(self, frame):
            self.frame = frame
            self.target_column = TARGET

    monkeypatch.setattr(read_module, "_loaded", lambda key, _f: (_Loaded(frames[key.city_slug]), frames[key.city_slug]))
    payload = read_module.get_cities(NeighbourhoodFilters(months=6), purpose="sale")
    by_city = {row["city"]: row for row in payload["cities"]}
    assert by_city["tehran"]["mean"] == pytest.approx(300_000_000)
    assert by_city["mashhad"]["mean"] == pytest.approx(100_000_000)
