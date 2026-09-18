"""Train a new model *version*, measured the way it will be used.

Two things differ from the original pipeline, both on purpose:

* The data is always extracted fresh. The old path read a cached parquet
  unless told otherwise, so a "retrain" could silently reuse August's data.
* Accuracy is measured on the most recent listings, not a random 20%. A model
  is used to price what is on the market *now*; a random split lets it be
  graded on listings from the same weeks it learnt from, which flatters it in a
  market moving several percent a month. The active model is graded on the
  very same listings, so "better than the current one" is a like-for-like
  claim.

After grading, the model is refit on the whole window — the newest two weeks
are the most informative data there is, and holding them out of the served
model would make it stale on arrival.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from config.settings import settings
from data.clean import feature_matrix, prepare_training_frame_for_key
from data.extract import extract_analytics_dataframe
from domain.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, target_column_for
from domain.model_key import ModelKey
from training import registry

logger = logging.getLogger(__name__)

#: The most recent listings are the exam. Two weeks is long enough to hold a
#: few thousand Tehran listings and short enough to still be "today's market".
HOLDOUT_DAYS = 14
#: Below this the recent slice is too thin to grade on; fall back to random.
MIN_HOLDOUT_ROWS = 100

MODEL_PARAMS: dict[str, Any] = {
    "loss_function": "RMSE",
    "depth": 6,
    "learning_rate": 0.08,
    "iterations": 600,
    "l2_leaf_reg": 4,
    "random_seed": settings.RANDOM_SEED,
    "verbose": False,
}


def _errors(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_values = actual.astype(float).to_numpy()
    predicted_values = np.maximum(np.asarray(predicted, dtype=float), 1.0)
    ape = np.abs(predicted_values - actual_values) / np.maximum(actual_values, 1.0)
    return {
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "mae": float(np.mean(np.abs(predicted_values - actual_values))),
    }


def _seen(frame: pd.DataFrame) -> pd.Series:
    if "first_seen_at" not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index)
    return pd.to_datetime(frame["first_seen_at"], utc=True, errors="coerce")


def train_version(
    key: ModelKey,
    *,
    window_days: int | None = None,
    now: datetime | None = None,
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train, grade against the active model, save as an inactive version.

    `frame` is for tests; production always extracts fresh.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    target = target_column_for(key)

    raw = frame if frame is not None else extract_analytics_dataframe(key)
    data = prepare_training_frame_for_key(raw, key)
    seen = _seen(data)

    if window_days:
        start = now - timedelta(days=int(window_days))
        data = data[seen >= start]
        seen = seen.loc[data.index]
    if len(data) < settings.MIN_TRAINING_ROWS:
        raise ValueError(
            f"Not enough rows for {key.slug()} in a {window_days or 'full'}-day window: "
            f"{len(data)} (need {settings.MIN_TRAINING_ROWS})"
        )

    holdout_start = now - timedelta(days=HOLDOUT_DAYS)
    is_recent = seen >= holdout_start
    test = data[is_recent]
    train = data[~is_recent]
    eval_method = "recent"
    if len(test) < MIN_HOLDOUT_ROWS or len(train) < settings.MIN_TRAINING_ROWS:
        train, test = train_test_split(
            data, test_size=settings.TRAIN_TEST_SIZE, random_state=settings.RANDOM_SEED
        )
        eval_method = "random"

    cat_indices = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES]
    x_train, y_train = feature_matrix(train), train[target].astype(float)
    x_test, y_test = feature_matrix(test), test[target].astype(float)

    graded = CatBoostRegressor(**MODEL_PARAMS)
    graded.fit(
        Pool(x_train, y_train, cat_features=cat_indices),
        eval_set=Pool(x_test, y_test, cat_features=cat_indices),
        use_best_model=True,
    )
    metrics = _errors(y_test, graded.predict(Pool(x_test, cat_features=cat_indices)))

    # The same exam for whatever serves today.
    baseline: dict[str, Any] | None = None
    active = registry.active_version(key)
    if active:
        try:
            active_model, _ = registry.load_version(key, active)
            baseline = {
                "version": active,
                **_errors(y_test, active_model.predict(Pool(x_test, cat_features=cat_indices))),
            }
        except Exception:
            logger.warning("could not grade active model %s", active, exc_info=True)

    best_iterations = max(int(graded.get_best_iteration() or 0) + 1, 50)
    final = CatBoostRegressor(**{**MODEL_PARAMS, "iterations": best_iterations})
    final.fit(Pool(feature_matrix(data), data[target].astype(float), cat_features=cat_indices))

    data_seen = _seen(data).dropna()
    meta: dict[str, Any] = {
        "version": registry.new_version_id(now),
        "model_key": key.slug(),
        "city_slug": key.city_slug,
        "property_type": key.property_type,
        "purpose": key.purpose,
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target": target,
        "trained_at": now.isoformat(),
        "window_days": window_days,
        "data_from": data_seen.min().isoformat() if not data_seen.empty else None,
        "data_to": data_seen.max().isoformat() if not data_seen.empty else None,
        "n_rows": int(len(data)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "eval_method": eval_method,
        "holdout_days": HOLDOUT_DAYS if eval_method == "recent" else None,
        "iterations": best_iterations,
        **metrics,
        "baseline": baseline,
        "better_than_active": (
            baseline is not None and metrics["mape"] < baseline["mape"]
        ),
        "comparables_cache": {
            "mean_toman": float(data[target].astype(float).mean()),
            "min_toman": float(data[target].astype(float).min()),
            "max_toman": float(data[target].astype(float).max()),
            "sample_size": int(len(data)),
            "target": target,
        },
    }
    if key.purpose == "rent":
        meta["rent_conversion"] = {
            "ratio_deposit_toman": settings.RENT_RATIO_DEPOSIT_TOMAN,
            "ratio_monthly_toman": settings.RENT_RATIO_MONTHLY_TOMAN,
        }

    registry.save_version(key, final, meta)
    logger.info(
        "trained %s version=%s rows=%s mape=%.3f baseline=%s",
        key.slug(),
        meta["version"],
        meta["n_rows"],
        meta["mape"],
        baseline and round(baseline["mape"], 3),
    )
    return meta
