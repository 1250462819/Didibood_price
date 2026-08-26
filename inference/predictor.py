"""Prediction + comparable market stats."""
from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import Pool

from config.settings import settings
from data.clean import price_outlier_mask
from data.comparables_subset import (
    known_neighbourhood_titles as _known_neighbourhood_titles,
    subset_comparables,
)
from data.extract import extract_comparables_dataframe, load_or_extract
from data.neighbourhood_persian import resolve_request_neighbourhood
from domain.features import FEATURE_COLUMNS, PricingFeatures, target_column_for
from domain.model_key import ModelKey
from domain.target import (
    prediction_bounds_rent,
    prediction_bounds_sale,
    split_rent_on_conversion_line,
    total_price_toman,
)
from training.pipeline import load_model


def predict_sale(features: PricingFeatures) -> dict[str, Any]:
    features.purpose = "sale"
    raw = _predict(features, purpose="sale")
    comp = raw["comparables"]
    return {
        "model_key": raw["model_key"],
        "model_trained_at": raw["model_trained_at"],
        "predicted_price_per_sqm_toman": raw["predicted_price_per_sqm_toman"],
        "predicted_total_price_toman": raw["predicted_total_price_toman"],
        "predicted_range": raw["predicted_range"],
        "comparables": {
            "mean_price_per_sqm_toman": comp["mean_price_per_sqm_toman"],
            "min_price_per_sqm_toman": comp["min_price_per_sqm_toman"],
            "max_price_per_sqm_toman": comp["max_price_per_sqm_toman"],
            "median_price_per_sqm_toman": comp["median_price_per_sqm_toman"],
            "q1_price_per_sqm_toman": comp.get("q1_price_per_sqm_toman"),
            "q3_price_per_sqm_toman": comp.get("q3_price_per_sqm_toman"),
            "sample_size": comp["sample_size"],
            "filters_applied": comp["filters_applied"],
        },
        "training_metrics": raw["training_metrics"],
    }


def predict_rent(features: PricingFeatures) -> dict[str, Any]:
    features.purpose = "rent"
    raw = _predict(features, purpose="rent")
    comp = raw["comparables"]
    return {
        "model_key": raw["model_key"],
        "model_trained_at": raw["model_trained_at"],
        "predicted_equivalent_deposit_toman": raw["predicted_equivalent_deposit_toman"],
        "predicted_equivalent_deposit_range": raw["predicted_equivalent_deposit_range"],
        "suggested_deposit_toman": raw["suggested_deposit_toman"],
        "suggested_monthly_rent_toman": raw["suggested_monthly_rent_toman"],
        "comparables": {
            "mean_equivalent_deposit_toman": comp["mean_equivalent_deposit_toman"],
            "min_equivalent_deposit_toman": comp["min_equivalent_deposit_toman"],
            "max_equivalent_deposit_toman": comp["max_equivalent_deposit_toman"],
            "median_equivalent_deposit_toman": comp["median_equivalent_deposit_toman"],
            "sample_size": comp["sample_size"],
            "filters_applied": comp["filters_applied"],
        },
        "training_metrics": raw["training_metrics"],
    }


def predict_price(features: PricingFeatures) -> dict[str, Any]:
    """Legacy combined entry — prefer predict_sale / predict_rent."""
    if features.purpose == "rent":
        return predict_rent(features)
    return predict_sale(features)


def _predict(features: PricingFeatures, *, purpose: str) -> dict[str, Any]:
    key = ModelKey(
        city_slug=features.city_slug,
        property_type=features.property_type,
        purpose=purpose,
    )
    model, meta = load_model(key)
    dataset = load_cached_dataset(key)
    features = _resolve_features_neighbourhood(features, dataset)
    row = features.to_model_row()
    frame = pd.DataFrame([row])
    cat_indices = [FEATURE_COLUMNS.index(c) for c in meta["categorical_features"]]
    pool = Pool(frame, cat_features=cat_indices)
    predicted = float(model.predict(pool)[0])

    target_column = target_column_for(key)
    comparables_dataset = load_comparables_dataset(key)
    if neighbourhood := (features.neighbourhood or "").strip():
        known = _known_neighbourhood_titles(comparables_dataset)
        resolved = resolve_request_neighbourhood(neighbourhood, known)
        if resolved:
            features.neighbourhood = resolved
    comparables = comparable_stats(
        features,
        dataset=comparables_dataset,
        target_column=target_column,
    )

    base = {
        "model_key": key.slug(),
        "model_trained_at": meta.get("trained_at"),
        "comparables": comparables,
        "training_metrics": {
            "mae": meta.get("mae"),
            "mape": meta.get("mape"),
            "train_rows": meta.get("n_rows"),
            "target": target_column,
        },
    }

    if purpose == "rent":
        return _rent_response(base, features, predicted, meta)
    return _sale_response(base, features, predicted, meta)


def _sale_response(
    base: dict[str, Any],
    features: PricingFeatures,
    predicted_pps: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    area = features.area or 0.0
    predicted_pps_int = int(round(predicted_pps))
    return {
        **base,
        "predicted_price_per_sqm_toman": predicted_pps_int,
        "predicted_total_price_toman": total_price_toman(predicted_pps, area)
        if area > 0
        else None,
        "predicted_range": prediction_bounds_sale(
            predicted_pps,
            area,
            mape=meta.get("mape"),
        ),
    }


def _rent_response(
    base: dict[str, Any],
    features: PricingFeatures,
    predicted_equivalent: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    conversion = meta.get("rent_conversion") or {}
    ratio_dep = float(
        conversion.get("ratio_deposit_toman", settings.RENT_RATIO_DEPOSIT_TOMAN)
    )
    ratio_mon = float(
        conversion.get("ratio_monthly_toman", settings.RENT_RATIO_MONTHLY_TOMAN)
    )
    equivalent_int = int(round(predicted_equivalent))
    deposit, monthly = split_rent_on_conversion_line(
        predicted_equivalent,
        ratio_deposit_toman=ratio_dep,
        ratio_monthly_toman=ratio_mon,
    )
    return {
        **base,
        "predicted_equivalent_deposit_toman": equivalent_int,
        "predicted_equivalent_deposit_range": prediction_bounds_rent(
            predicted_equivalent,
            mape=meta.get("mape"),
        ),
        "suggested_deposit_toman": deposit,
        "suggested_monthly_rent_toman": monthly,
    }


def load_cached_dataset(key: ModelKey) -> pd.DataFrame:
    try:
        return load_or_extract(key, refresh=False)
    except Exception:
        return pd.DataFrame()


def load_comparables_dataset(key: ModelKey) -> pd.DataFrame:
    """Fresh PLP rows from the last COMPARABLE_RECENT_DAYS for neighbourhood ppsqm stats."""
    try:
        return extract_comparables_dataframe(key)
    except Exception:
        return pd.DataFrame()


def _resolve_features_neighbourhood(
    features: PricingFeatures,
    dataset: pd.DataFrame,
) -> PricingFeatures:
    raw = (features.neighbourhood or "").strip()
    if not raw:
        return features
    resolved = resolve_request_neighbourhood(raw, _known_neighbourhood_titles(dataset))
    if resolved and resolved != features.neighbourhood:
        features.neighbourhood = resolved
    return features


def comparable_stats(
    features: PricingFeatures,
    *,
    dataset: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    if dataset.empty or target_column not in dataset.columns:
        return _empty_comparables(features.purpose)

    neighbourhood = (features.neighbourhood or "").strip()
    subset, filters = subset_comparables(dataset, neighbourhood or None)
    neighbourhood_applied = bool(filters.get("neighbourhood_applied"))
    resolved_neighbourhood = filters.get("neighbourhood_resolved")

    if neighbourhood_applied and subset.empty:
        return _empty_comparables(
            features.purpose,
            filters={
                **filters,
                "min_max_outliers_removed": 0,
            },
        )

    y = subset[target_column].astype(float)
    if y.empty:
        return _empty_comparables(
            features.purpose,
            filters={
                **filters,
                "min_max_outliers_removed": 0,
            },
        )

    y_for_bounds = y.loc[price_outlier_mask(y)]
    if y_for_bounds.empty:
        y_for_bounds = y

    filters = {
        **filters,
        "min_max_outliers_removed": int(len(y) - len(y_for_bounds)),
    }

    if features.purpose == "rent":
        return {
            "mean_price_per_sqm_toman": None,
            "min_price_per_sqm_toman": None,
            "max_price_per_sqm_toman": None,
            "median_price_per_sqm_toman": None,
            "mean_equivalent_deposit_toman": int(round(float(y.mean()))),
            "min_equivalent_deposit_toman": int(round(float(y_for_bounds.min()))),
            "max_equivalent_deposit_toman": int(round(float(y_for_bounds.max()))),
            "median_equivalent_deposit_toman": int(round(float(y.median()))),
            "sample_size": int(len(y)),
            "filters_applied": filters,
        }

    return {
        "mean_price_per_sqm_toman": int(round(float(y.mean()))),
        "min_price_per_sqm_toman": int(round(float(y_for_bounds.min()))),
        "max_price_per_sqm_toman": int(round(float(y_for_bounds.max()))),
        "median_price_per_sqm_toman": int(round(float(y.median()))),
        "q1_price_per_sqm_toman": int(round(float(y.quantile(0.25)))),
        "q3_price_per_sqm_toman": int(round(float(y.quantile(0.75)))),
        "mean_equivalent_deposit_toman": None,
        "min_equivalent_deposit_toman": None,
        "max_equivalent_deposit_toman": None,
        "median_equivalent_deposit_toman": None,
        "sample_size": int(len(y)),
        "filters_applied": filters,
    }


def _empty_comparables(
    purpose: str,
    *,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = {
        "mean_price_per_sqm_toman": None,
        "min_price_per_sqm_toman": None,
        "max_price_per_sqm_toman": None,
        "median_price_per_sqm_toman": None,
        "q1_price_per_sqm_toman": None,
        "q3_price_per_sqm_toman": None,
        "mean_equivalent_deposit_toman": None,
        "min_equivalent_deposit_toman": None,
        "max_equivalent_deposit_toman": None,
        "median_equivalent_deposit_toman": None,
        "sample_size": 0,
        "filters_applied": filters or {},
    }
    if purpose == "rent":
        return base
    return base
