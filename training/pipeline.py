"""Train and persist CatBoost models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor

from config.settings import settings
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
    """Train a new version. It is saved *inactive*: see `training.registry`.

    `refresh_data` is kept for callers of the old signature; data is now always
    extracted fresh, because the cached dataset is how retrains used to reuse
    stale listings without saying so.
    """
    del refresh_data
    from training import registry
    from training.versioned import train_version

    meta = train_version(key)
    model_path, meta_path = registry.version_paths(key, meta["version"])
    return TrainResult(
        model_key=key.slug(),
        n_rows=meta["n_rows"],
        n_train=meta["n_train"],
        n_test=meta["n_test"],
        mae=meta["mae"],
        mape=meta["mape"],
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
    """The model predictions use: the registry's active version."""
    from training import registry

    return registry.load_active(key)
