"""Rent predict + train routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from domain.features import PricingFeatures
from domain.model_key import ModelKey
from inference.predictor import predict_rent
from schemas.rent import (
    RentPredictRequest,
    RentPredictResponse,
    RentTrainRequest,
    RentTrainResponse,
)
from training.pipeline import train_model

router = APIRouter(prefix="/rent", tags=["Rent"])


@router.post("/predict", response_model=RentPredictResponse)
def predict(payload: RentPredictRequest) -> RentPredictResponse:
    features = PricingFeatures.from_request({**payload.model_dump(), "purpose": "rent"})
    if not features.city_slug:
        raise HTTPException(status_code=422, detail="city_slug is required")
    try:
        result = predict_rent(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model not trained: {exc}") from exc
    return RentPredictResponse(**result)


@router.post("/train", response_model=RentTrainResponse)
def train(payload: RentTrainRequest) -> RentTrainResponse:
    key = ModelKey(
        city_slug=payload.city_slug.strip().lower(),
        property_type=payload.property_type,
        purpose="rent",
    )
    try:
        result = train_model(key, refresh_data=payload.refresh_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RentTrainResponse(
        model_key=result.model_key,
        n_rows=result.n_rows,
        n_train=result.n_train,
        n_test=result.n_test,
        mae=result.mae,
        mape=result.mape,
        model_path=result.model_path,
        trained_at=result.trained_at,
    )
