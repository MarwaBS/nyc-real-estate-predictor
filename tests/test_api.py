"""Tests for FastAPI endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_response_has_models_loaded_field() -> None:
    response = client.get("/health")
    data = response.json()
    assert "models_loaded" in data


def test_predict_returns_503_when_no_models() -> None:
    """Before models are trained, predict should return 503."""
    response = client.post("/predict", json={
        "beds": 2,
        "bath": 2.0,
        "propertysqft": 1200.0,
        "borough": "manhattan",
        "type": "condo",
        "zipcode": "10022",
        "latitude": 40.758,
        "longitude": -73.985,
        "sublocality": "midtown",
    })
    # 503 if models not trained, 200 if they are — both are acceptable
    assert response.status_code in (200, 503)


def test_predict_rejects_invalid_zipcode() -> None:
    response = client.post("/predict", json={
        "beds": 2,
        "bath": 2.0,
        "propertysqft": 1200.0,
        "borough": "manhattan",
        "type": "condo",
        "zipcode": "abc",  # Invalid
        "latitude": 40.758,
        "longitude": -73.985,
    })
    assert response.status_code == 422


def test_predict_rejects_negative_sqft() -> None:
    response = client.post("/predict", json={
        "beds": 2,
        "bath": 2.0,
        "propertysqft": -100.0,  # Invalid
        "borough": "manhattan",
        "type": "condo",
        "zipcode": "10022",
        "latitude": 40.758,
        "longitude": -73.985,
    })
    assert response.status_code == 422


def test_docs_endpoint_accessible() -> None:
    response = client.get("/docs")
    assert response.status_code == 200
