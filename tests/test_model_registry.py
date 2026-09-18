"""Model versions: training never replaces what serves, and admins choose.

The claims under test are the ones the admin panel makes: a new training is
kept alongside the old one, nothing goes live on its own, "better than the
current model" means better on the same listings, and going back is always
possible.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from domain.model_key import ModelKey
from domain.target import SALE_TARGET_COLUMN
from training import registry
from training.versioned import HOLDOUT_DAYS, train_version

KEY = ModelKey(city_slug="tehran", property_type="apartment", purpose="sale")
NOW = datetime(2026, 9, 19, 12, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_models(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(settings, "MIN_TRAINING_ROWS", 40)
    registry._MODEL_CACHE.clear()
    yield tmp_path
    registry._MODEL_CACHE.clear()


def _frame(n: int = 400, *, days: int = 60, seed: int = 1) -> pd.DataFrame:
    """Listings whose price per metre follows area and neighbourhood cleanly."""
    rng = np.random.default_rng(seed)
    hoods = np.array(["الهیه", "پونک", "نارمک", "ونک"])
    base = {"الهیه": 600e6, "پونک": 250e6, "نارمک": 150e6, "ونک": 400e6}
    rows = []
    for index in range(n):
        hood = hoods[index % len(hoods)]
        area = float(rng.integers(50, 200))
        price = base[hood] * (1 + (area - 100) / 1000) * rng.normal(1, 0.03)
        seen = NOW - timedelta(days=float(rng.uniform(0, days)))
        rows.append(
            {
                "neighbourhood": hood,
                "area": area,
                "rooms": int(area // 50),
                "year_built": 1395,
                "building_age": 9,
                "floor_number": 2,
                "total_floors": 5,
                "has_parking": 1,
                "has_elevator": 1,
                "has_storage": 1,
                "has_balcony": 0,
                "number_of_bathrooms": 1,
                "location_lat": 35.7,
                "location_long": 51.4,
                SALE_TARGET_COLUMN: price,
                "price_total_toman": price * area,
                "post_token": f"t{index}",
                "first_seen_at": seen,
            }
        )
    return pd.DataFrame(rows)


def test_a_training_is_saved_as_a_version_and_the_first_one_goes_live():
    meta = train_version(KEY, now=NOW, frame=_frame())
    assert registry.active_version(KEY) == meta["version"]
    assert [v["version"] for v in registry.list_versions(KEY)] == [meta["version"]]


def test_a_second_training_does_not_replace_what_serves():
    first = train_version(KEY, now=NOW, frame=_frame(seed=1))
    second = train_version(KEY, now=NOW + timedelta(hours=1), frame=_frame(seed=2))
    assert registry.active_version(KEY) == first["version"]
    assert {v["version"] for v in registry.list_versions(KEY)} == {
        first["version"],
        second["version"],
    }


def test_the_new_model_is_graded_on_the_same_listings_as_the_active_one():
    first = train_version(KEY, now=NOW, frame=_frame(seed=1))
    second = train_version(KEY, now=NOW + timedelta(hours=1), frame=_frame(seed=2))
    assert second["baseline"]["version"] == first["version"]
    assert second["better_than_active"] == (second["mape"] < second["baseline"]["mape"])


def test_the_exam_is_the_most_recent_listings():
    meta = train_version(KEY, now=NOW, frame=_frame(n=600, days=60))
    assert meta["eval_method"] == "recent"
    assert meta["holdout_days"] == HOLDOUT_DAYS
    # Roughly a quarter of a 60-day spread falls in the last 14 days.
    assert 80 < meta["n_test"] < 250


def test_a_window_keeps_only_that_many_days_of_listings():
    meta = train_version(KEY, now=NOW, window_days=30, frame=_frame(n=600, days=90))
    assert meta["window_days"] == 30
    assert datetime.fromisoformat(meta["data_from"]) >= NOW - timedelta(days=30)
    assert meta["n_rows"] < 600


def test_too_few_recent_listings_falls_back_to_a_random_split_and_says_so():
    meta = train_version(KEY, now=NOW + timedelta(days=40), frame=_frame(n=300, days=20))
    assert meta["eval_method"] == "random"
    assert meta["holdout_days"] is None


def test_activating_an_older_version_is_a_rollback():
    first = train_version(KEY, now=NOW, frame=_frame(seed=1))
    second = train_version(KEY, now=NOW + timedelta(hours=1), frame=_frame(seed=2))
    registry.activate(KEY, second["version"])
    assert registry.active_version(KEY) == second["version"]
    registry.activate(KEY, first["version"])
    assert registry.active_version(KEY) == first["version"]


def test_predictions_follow_the_active_version():
    first = train_version(KEY, now=NOW, frame=_frame(seed=1))
    second = train_version(KEY, now=NOW + timedelta(hours=1), frame=_frame(seed=2))
    assert registry.load_active(KEY)[1]["version"] == first["version"]
    registry.activate(KEY, second["version"])
    assert registry.load_active(KEY)[1]["version"] == second["version"]


def test_an_unknown_version_cannot_be_activated():
    train_version(KEY, now=NOW, frame=_frame())
    with pytest.raises(FileNotFoundError):
        registry.activate(KEY, "19990101T000000Z")


def test_a_legacy_model_is_imported_as_the_active_first_version(isolated_models):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(iterations=5, verbose=False)
    model.fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
    legacy = isolated_models / f"{KEY.slug()}"
    model.save_model(str(legacy.with_suffix(".cbm")))
    legacy.with_suffix(".meta.json").write_text(
        json.dumps({"model_key": KEY.slug(), "trained_at": "2026-08-07T12:52:00+00:00", "mape": 0.18}),
        encoding="utf-8",
    )

    versions = registry.list_versions(KEY)
    assert len(versions) == 1
    assert versions[0]["version"] == "20260807T125200Z"
    assert versions[0]["imported_from_legacy"] is True
    assert registry.active_version(KEY) == "20260807T125200Z"
    # The original file is left where it was, as a backup.
    assert legacy.with_suffix(".cbm").is_file()
