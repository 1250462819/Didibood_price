"""Dataset cleaning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import settings
from domain.features import FEATURE_COLUMNS, target_column_for
from domain.model_key import ModelKey
from domain.target import RENT_TARGET_COLUMN


def drop_invalid_rows(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out[out[target_column].notna()]
    out = out[out[target_column] > 0]
    out = out[out["area"].notna()]
    out = out[out["area"] >= 25]
    out = out[out["area"] <= 600]
    return out.reset_index(drop=True)


def drop_rent_cap_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Rent-only: drop parser garbage and unrealistic deposit/monthly/equivalent."""
    if df.empty:
        return df
    out = df.copy()
    target = RENT_TARGET_COLUMN
    out = out[out[target] >= settings.RENT_MIN_EQUIVALENT_TOMAN]
    out = out[out[target] <= settings.RENT_MAX_EQUIVALENT_TOMAN]

    if "deposit_toman" in out.columns:
        dep = out["deposit_toman"].fillna(0).astype(float)
        out = out[dep <= settings.RENT_MAX_DEPOSIT_TOMAN]
    if "monthly_rent_toman" in out.columns:
        mon = out["monthly_rent_toman"].fillna(0).astype(float)
        out = out[mon <= settings.RENT_MAX_MONTHLY_TOMAN]

    return out.reset_index(drop=True)


def price_outlier_mask(series: pd.Series, *, iqr_multiplier: float | None = None) -> pd.Series:
    """IQR mask on log(target)."""
    if len(series) < 20:
        return pd.Series(True, index=series.index)
    log_y = np.log(series.astype(float))
    q1, q3 = log_y.quantile(0.25), log_y.quantile(0.75)
    iqr = q3 - q1
    mult = (
        settings.OUTLIER_IQR_MULTIPLIER
        if iqr_multiplier is None
        else iqr_multiplier
    )
    low, high = q1 - mult * iqr, q3 + mult * iqr
    return (log_y >= low) & (log_y <= high)


def remove_price_outliers(
    df: pd.DataFrame,
    target_column: str,
    *,
    iqr_multiplier: float | None = None,
) -> pd.DataFrame:
    if len(df) < 20:
        return df
    mask = price_outlier_mask(
        df[target_column].astype(float),
        iqr_multiplier=iqr_multiplier,
    )
    return df.loc[mask].reset_index(drop=True)


def prepare_training_frame(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Sale (and generic) — unchanged."""
    out = drop_invalid_rows(df, target_column)
    return remove_price_outliers(out, target_column)


def prepare_rent_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rent-only: caps + tighter IQR."""
    target = RENT_TARGET_COLUMN
    out = drop_invalid_rows(df, target)
    out = drop_rent_cap_violations(out)
    return remove_price_outliers(
        out,
        target,
        iqr_multiplier=settings.RENT_OUTLIER_IQR_MULTIPLIER,
    )


def prepare_training_frame_for_key(df: pd.DataFrame, key: ModelKey) -> pd.DataFrame:
    if key.purpose == "rent":
        return prepare_rent_training_frame(df)
    return prepare_training_frame(df, target_column_for(key))


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[list(FEATURE_COLUMNS)].copy()
    frame["neighbourhood"] = frame["neighbourhood"].fillna("unknown").astype(str)
    return frame
