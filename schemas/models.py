"""Schemas for model management."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    city: str = Field(..., examples=["tehran"])
    purpose: Literal["sale", "rent"] = "sale"
    #: None trains on everything crawled. Otherwise the last N days.
    window_days: int | None = Field(None, ge=14, le=730)


class TrainJob(BaseModel):
    id: str
    model_key: str
    city_slug: str
    purpose: str
    window_days: int | None
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    started_at: str | None
    finished_at: str | None
    version: str | None
    error: str | None


class TrainJobsResponse(BaseModel):
    jobs: list[TrainJob]


class ModelKeyVersions(BaseModel):
    model_key: str
    city_slug: str
    purpose: str
    active: dict[str, Any] | None
    #: Raw version metadata — see `training.versioned` for its fields.
    versions: list[dict[str, Any]]


class ModelsResponse(BaseModel):
    keys: list[ModelKeyVersions]
    running: dict[str, Any] | None = None


class ActivateResponse(BaseModel):
    model_key: str
    version: str
    activated_at: str
    reason: str
