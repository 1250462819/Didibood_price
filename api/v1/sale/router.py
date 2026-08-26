"""Sale predict + train routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from domain.features import PricingFeatures
from domain.model_key import ModelKey
from inference.predictor import predict_sale
from schemas.sale import (
    SalePredictRequest,
    SalePredictResponse,
    SaleTrainRequest,
    SaleTrainResponse,
)
from training.pipeline import train_model

router = APIRouter(prefix="/sale", tags=["Sale"])


@router.post("/predict", response_model=SalePredictResponse)
def predict(payload: SalePredictRequest) -> SalePredictResponse:
    features = PricingFeatures.from_request({**payload.model_dump(), "purpose": "sale"})
    if not features.city_slug:
        raise HTTPException(status_code=422, detail="city_slug is required")
    try:
        result = predict_sale(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model not trained: {exc}") from exc
    return SalePredictResponse(**result)


@router.post("/train", response_model=SaleTrainResponse)
def train(payload: SaleTrainRequest) -> SaleTrainResponse:
    key = ModelKey(
        city_slug=payload.city_slug.strip().lower(),
        property_type=payload.property_type,
        purpose="sale",
    )
    try:
        result = train_model(key, refresh_data=payload.refresh_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SaleTrainResponse(
        model_key=result.model_key,
        n_rows=result.n_rows,
        n_train=result.n_train,
        n_test=result.n_test,
        mae=result.mae,
        mape=result.mape,
        model_path=result.model_path,
        trained_at=result.trained_at,
    )
