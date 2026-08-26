"""Train and persist CatBoost models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from config.settings import settings
from data.clean import feature_matrix, prepare_training_frame_for_key
from data.extract import load_or_extract
from domain.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, target_column_for
from domain.model_key import ModelKey


@dataclass
class TrainResult:
    model_key: str
    n_rows: int
    n_train: int
    n_test: int
    mae: float
    mape: float
    model_path: str
    meta_path: str
    trained_at: str


def model_paths(key: ModelKey) -> tuple[Path, Path]:
    base = settings.MODEL_DIR / key.slug()
    return base.with_suffix(".cbm"), base.with_suffix(".meta.json")


def train_model(key: ModelKey, *, refresh_data: bool = False) -> TrainResult:
    target_column = target_column_for(key)
    df = load_or_extract(key, refresh=refresh_data)
    df = prepare_training_frame_for_key(df, key)
    if len(df) < settings.MIN_TRAINING_ROWS:
        raise ValueError(
            f"Not enough rows for {key.slug()}: {len(df)} "
            f"(need {settings.MIN_TRAINING_ROWS})"
        )

    x = feature_matrix(df)
    y = df[target_column].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=settings.TRAIN_TEST_SIZE,
        random_state=settings.RANDOM_SEED,
    )

    cat_indices = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES]
    train_pool = Pool(x_train, y_train, cat_features=cat_indices)
    test_pool = Pool(x_test, y_test, cat_features=cat_indices)

    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=6,
        learning_rate=0.08,
        iterations=600,
        l2_leaf_reg=4,
        random_seed=settings.RANDOM_SEED,
        verbose=False,
    )
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    preds = model.predict(test_pool)
    mae = float(mean_absolute_error(y_test, preds))
    mape = float(mean_absolute_percentage_error(y_test, np.maximum(preds, 1)))

    model_path, meta_path = model_paths(key)
    model.save_model(str(model_path))

    meta = {
        "model_key": key.slug(),
        "city_slug": key.city_slug,
        "property_type": key.property_type,
        "purpose": key.purpose,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target": target_column,
        "n_rows": len(df),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "mae": mae,
        "mape": mape,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "comparables_cache": _comparables_summary(df, target_column),
    }
    if key.purpose == "rent":
        meta["rent_conversion"] = {
            "ratio_deposit_toman": settings.RENT_RATIO_DEPOSIT_TOMAN,
            "ratio_monthly_toman": settings.RENT_RATIO_MONTHLY_TOMAN,
        }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return TrainResult(
        model_key=key.slug(),
        n_rows=len(df),
        n_train=len(x_train),
        n_test=len(x_test),
        mae=mae,
        mape=mape,
        model_path=str(model_path),
        meta_path=str(meta_path),
        trained_at=meta["trained_at"],
    )


def _comparables_summary(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    y = df[target_column].astype(float)
    return {
        "mean_toman": float(y.mean()),
        "min_toman": float(y.min()),
        "max_toman": float(y.max()),
        "sample_size": int(len(y)),
        "target": target_column,
    }


def load_model(key: ModelKey) -> tuple[CatBoostRegressor, dict[str, Any]]:
    model_path, meta_path = model_paths(key)
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Model not found for {key.slug()}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta
