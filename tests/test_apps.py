"""FastAPI app smoke tests."""
from __future__ import annotations


def test_unified_app_routes():
    from main import app

    paths = set(app.openapi()["paths"])
    assert "/api/v1/sale/predict" in paths
    assert "/api/v1/sale/train" in paths
    assert "/api/v1/rent/predict" in paths
    assert "/api/v1/rent/train" in paths
    assert "/api/v1/market-stats" in paths
    assert "/health" in paths
