"""Tests for FastAPI endpoints."""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "beds": 2,
    "bath": 2.0,
    "propertysqft": 1200.0,
    "borough": "manhattan",
    "type": "condo",
    "zipcode": "10022",
    "latitude": 40.758,
    "longitude": -73.985,
    "sublocality": "midtown east",
}


def _models_present() -> bool:
    from src.config import MODELS_DIR

    return (Path(MODELS_DIR) / "price_zone_best.joblib").exists()


@contextmanager
def reloaded_app(**env: str | None) -> Iterator[object]:
    """Reload ``api.main`` with the given env applied so import-time globals
    (``api_key``, ``daily_rate_limit`` → the limiter decorator) are rebuilt.
    ``None`` deletes the var for the duration of the block."""
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import api.settings as s

        s.get_settings.cache_clear()
        import api.main as m

        importlib.reload(m)
        yield m
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        import api.settings as s

        s.get_settings.cache_clear()
        import api.main as m

        importlib.reload(m)


def test_health_endpoint_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_response_has_models_loaded_field() -> None:
    response = client.get("/health")
    data = response.json()
    assert "models_loaded" in data


@pytest.mark.skipif(
    not _models_present(), reason="flagship models are DVC/local-only (absent in CI)"
)
def test_predict_returns_200_with_valid_input() -> None:
    """With models present, /predict returns a real 200 and the documented shape —
    a hard success assertion, not the old 'either 200 or 503'."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"zone", "price"}
    assert body["zone"]["price_zone"] in {"Low", "Medium", "High", "Very High"}
    assert 0.0 <= body["zone"]["confidence"] <= 1.0
    assert body["price"]["predicted_price"] > 0
    assert body["price"]["price_range"]["low"] <= body["price"]["price_range"]["high"]


def test_predict_returns_503_when_models_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model artefacts can't be loaded, /predict must return 503 (not 500
    or a corrupt 200). Forced deterministically by making the first model access
    raise FileNotFoundError, independent of whether models exist locally."""
    import api.main as m

    def _missing() -> dict:
        raise FileNotFoundError("models absent")

    monkeypatch.setattr(m, "_get_capped_categories", _missing)
    resp = TestClient(m.app).post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 503
    assert "not yet trained" in resp.json()["detail"].lower()


def test_predict_requires_api_key_returns_401() -> None:
    """With API_KEY configured, a request missing X-API-Key is rejected 401 —
    auth runs before model load, so this holds with or without models present."""
    with reloaded_app(API_KEY="s3cret") as m:
        resp = TestClient(m.app).post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 401


def test_predict_wrong_api_key_returns_403() -> None:
    with reloaded_app(API_KEY="s3cret") as m:
        resp = TestClient(m.app).post(
            "/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "wrong"}
        )
        assert resp.status_code == 403


def test_predict_rate_limit_returns_429() -> None:
    """Under a tight per-IP limit, repeated calls eventually return 429. The limit
    is enforced before the body, so this holds even when models are absent (the
    pre-429 calls may be 200 or 503; only the 429 transition matters)."""
    with reloaded_app(DAILY_RATE_LIMIT="3/minute") as m:
        c = TestClient(m.app)
        statuses = [
            c.post("/predict", json=VALID_PAYLOAD).status_code for _ in range(8)
        ]
        assert 429 in statuses, f"rate limit never enforced: {statuses}"
        first_reject = statuses.index(429)
        # Every status before the first 429 is a normal handled response, never 429.
        assert all(s != 429 for s in statuses[:first_reject])


def test_predict_rejects_invalid_zipcode() -> None:
    response = client.post(
        "/predict",
        json={
            "beds": 2,
            "bath": 2.0,
            "propertysqft": 1200.0,
            "borough": "manhattan",
            "type": "condo",
            "zipcode": "abc",  # Invalid
            "latitude": 40.758,
            "longitude": -73.985,
        },
    )
    assert response.status_code == 422


def test_predict_rejects_negative_sqft() -> None:
    response = client.post(
        "/predict",
        json={
            "beds": 2,
            "bath": 2.0,
            "propertysqft": -100.0,  # Invalid
            "borough": "manhattan",
            "type": "condo",
            "zipcode": "10022",
            "latitude": 40.758,
            "longitude": -73.985,
        },
    )
    assert response.status_code == 422


def test_docs_endpoint_accessible() -> None:
    response = client.get("/docs")
    assert response.status_code == 200
