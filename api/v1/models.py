"""Model management for the admin panel: versions, training jobs, activation.

Not routed through the public gateway. The backend calls these on behalf of an
authenticated admin, with `X-Models-Token`.
"""
from __future__ import annotations

import hmac

from fastapi import APIRouter, Header, HTTPException

from config.settings import settings
from domain.model_key import SUPPORTED_CITIES, ModelKey
from schemas.models import (
    ActivateResponse,
    ModelsResponse,
    TrainJob,
    TrainJobsResponse,
    TrainRequest,
)
from training import jobs, registry

router = APIRouter(prefix="/models", tags=["Model management"])

#: Rent is trained and listed, but its data stopped on 13 August; the panel
#: only offers sale for now, and so does this list by default.
DEFAULT_PURPOSES = ("sale",)


def _authorise(token: str | None) -> None:
    expected = settings.MODELS_ADMIN_TOKEN
    if not expected:
        return
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid models token")


def _key(city: str, purpose: str, property_type: str = "apartment") -> ModelKey:
    slug = city.strip().lower()
    if slug not in SUPPORTED_CITIES:
        raise HTTPException(status_code=422, detail=f"Unsupported city: {city}")
    return ModelKey(city_slug=slug, property_type=property_type, purpose=purpose)


@router.get("", response_model=ModelsResponse)
def list_models(
    purpose: str = "sale",
    x_models_token: str | None = Header(None),
) -> ModelsResponse:
    """Every city's versions, newest first, with the active one marked."""
    _authorise(x_models_token)
    keys = []
    for city in SUPPORTED_CITIES:
        key = _key(city, purpose)
        active = registry.active_record(key)
        versions = registry.list_versions(key)
        for version in versions:
            version["is_active"] = bool(active and version["version"] == active["version"])
        keys.append(
            {
                "model_key": key.slug(),
                "city_slug": key.city_slug,
                "purpose": key.purpose,
                "active": active,
                "versions": versions,
            }
        )
    return ModelsResponse(keys=keys, running=jobs.running_job())


@router.post("/train", response_model=TrainJob, status_code=202)
def start_training(
    payload: TrainRequest,
    x_models_token: str | None = Header(None),
) -> TrainJob:
    """Queue a training. The new version is saved inactive."""
    _authorise(x_models_token)
    key = _key(payload.city, payload.purpose)
    try:
        job = jobs.start_training(key, window_days=payload.window_days)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return TrainJob(**job)


@router.get("/jobs", response_model=TrainJobsResponse)
def list_training_jobs(x_models_token: str | None = Header(None)) -> TrainJobsResponse:
    _authorise(x_models_token)
    return TrainJobsResponse(jobs=[TrainJob(**job) for job in jobs.list_jobs()])


@router.get("/jobs/{job_id}", response_model=TrainJob)
def get_training_job(job_id: str, x_models_token: str | None = Header(None)) -> TrainJob:
    _authorise(x_models_token)
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return TrainJob(**job)


@router.post("/{city}/{purpose}/versions/{version}/activate", response_model=ActivateResponse)
def activate_version(
    city: str,
    purpose: str,
    version: str,
    x_models_token: str | None = Header(None),
) -> ActivateResponse:
    """Make `version` the one that answers predictions. Rollback is the same call."""
    _authorise(x_models_token)
    key = _key(city, purpose)
    try:
        record = registry.activate(key, version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ActivateResponse(model_key=key.slug(), **record)
