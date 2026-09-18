"""The model-management routes are for the admin panel only."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app_factory import create_app
from config.settings import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(settings, "MODELS_ADMIN_TOKEN", "secret-token")
    return TestClient(create_app())


def test_listing_models_needs_the_token(client):
    assert client.get("/api/v1/models").status_code == 401
    assert client.get("/api/v1/models", headers={"X-Models-Token": "wrong"}).status_code == 401


def test_with_the_token_every_city_is_listed_even_before_any_training(client):
    response = client.get("/api/v1/models", headers={"X-Models-Token": "secret-token"})
    assert response.status_code == 200
    cities = {key["city_slug"] for key in response.json()["keys"]}
    assert cities == {"tehran", "mashhad", "isfahan"}


def test_training_an_unknown_city_is_refused(client):
    response = client.post(
        "/api/v1/models/train",
        json={"city": "atlantis", "purpose": "sale"},
        headers={"X-Models-Token": "secret-token"},
    )
    assert response.status_code == 422


def test_a_window_shorter_than_the_exam_is_refused(client):
    response = client.post(
        "/api/v1/models/train",
        json={"city": "tehran", "purpose": "sale", "window_days": 7},
        headers={"X-Models-Token": "secret-token"},
    )
    assert response.status_code == 422


def test_activating_a_missing_version_is_a_404(client):
    response = client.post(
        "/api/v1/models/tehran/sale/versions/19990101T000000Z/activate",
        headers={"X-Models-Token": "secret-token"},
    )
    assert response.status_code == 404
