"""Tests for FastAPI endpoints."""

from __future__ import annotations

import importlib
import json
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

    return (Path(MODELS_DIR) / "price_regressor_best.joblib").exists()


@contextmanager
def reloaded_app(**env: str | None) -> Iterator[object]:
    """Reload ``api.main`` with the given env applied so import-time globals
    (``api_key``, ``predict_rate_limit`` → the limiter decorator) are rebuilt.
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


def test_health_reports_missing_price_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the calibrated interval is a REQUIRED fourth artefact.

    /predict calls get_price_interval, which raises rather than falling back to
    a guess. A deployment shipped without price_interval.json therefore had a
    loadable model and answered /health with 200 + models_loaded true, while
    every prediction returned 500.
    """
    import api.main as m
    import src.models.predict as predict_module

    def _interval_missing() -> object:
        raise FileNotFoundError("price_interval.json missing")

    monkeypatch.setattr(m, "_get_regressor", lambda: object())
    monkeypatch.setattr(predict_module, "get_price_interval", _interval_missing)

    resp = TestClient(m.app).get("/health")
    assert resp.status_code == 503, resp.text
    assert resp.json()["models_loaded"] is False


def test_health_returns_503_when_no_models_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container with nothing loaded must fail its healthcheck.

    This is the case the CI smoke test (`curl -fsS /health`, body discarded)
    is meant to catch. While /health returned a hardcoded "ok", an image
    built with zero model artifacts passed that gate green.
    """
    import api.main as m

    def _missing() -> object:
        raise FileNotFoundError("no artifacts in image")

    monkeypatch.setattr(m, "_get_regressor", _missing)

    resp = TestClient(m.app).get("/health")
    assert resp.status_code == 503, resp.text
    assert resp.json()["models_loaded"] is False


def test_health_reports_healthy_when_full_stack_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the regressor and the calibrated interval load, /health is ready."""
    import api.main as m

    monkeypatch.setattr(m, "_get_regressor", lambda: object())

    resp = TestClient(m.app).get("/health")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


@pytest.mark.skipif(
    not _models_present(),
    reason="serving artifacts missing (partial checkout) — they are committed and pinned by models/MANIFEST.sha256",
)
def test_predict_returns_200_with_valid_input() -> None:
    """With models present, /predict returns a real 200 and the documented shape —
    a hard success assertion, not the old 'either 200 or 503'."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"zone", "price"}
    assert body["zone"]["price_zone"] in {"Low", "Medium", "High", "Very High"}
    # A floor with domain meaning, not `> 0`: the payload is a 1,200 sqft
    # Manhattan condo, and the cheapest row anywhere in the training data is
    # ~$2.5k. `> 0` would have passed the historical bug that served a
    # Manhattan condo at single-digit dollars.
    assert body["price"]["predicted_price"] > 10_000, body["price"]

    # The SERVED band must be the calibrated artefact, not merely ordered.
    # `low <= high` passes for any fabricated pair: replacing the endpoint's
    # price_range() call with {"low": price*0.5, "high": price*2.0} left the
    # whole suite green, because the artefact was pinned only in the library
    # function and nothing tied the endpoint to it. MODEL_CARD publishes this
    # band's measured coverage, so an unpinned endpoint can publish a number
    # the served interval does not honour.
    interval = json.loads(
        (
            Path(__file__).resolve().parents[1] / "models" / "price_interval.json"
        ).read_text(encoding="utf-8")
    )
    # Compared as ratios, not equalities: the endpoint derives the band from
    # the unrounded prediction while predicted_price is rounded to the nearest
    # $100, so the two disagree by up to one rounding unit (~5e-5 relative on a
    # $1.8M prediction). 1e-3 sits well above that and far below any real
    # drift — the fabricated 0.5/2.0 band above misses by 200x.
    predicted = body["price"]["predicted_price"]
    band = body["price"]["price_range"]
    assert band["low"] / predicted == pytest.approx(
        interval["low_multiplier"], abs=1e-3
    )
    assert band["high"] / predicted == pytest.approx(
        interval["high_multiplier"], abs=1e-3
    )


def test_predict_returns_503_when_models_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model artefacts can't be loaded, /predict must return 503 (not 500
    or a corrupt 200). Forced deterministically by making the shared inference
    path's first model access raise FileNotFoundError, independent of whether
    models exist locally."""
    import api.main as m
    import src.models.predict as predict_module

    def _missing(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("models absent")

    monkeypatch.setattr(predict_module, "get_regressor", _missing)
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
    with reloaded_app(PREDICT_RATE_LIMIT="3/minute") as m:
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


def test_predict_rejects_out_of_range_beds() -> None:
    """The le=20 bound must hold — an unbounded count feeds the model a value
    far outside anything it trained on."""
    response = client.post(
        "/predict",
        json={
            "beds": 21,  # over the bound
            "bath": 2.0,
            "propertysqft": 1200.0,
            "borough": "manhattan",
            "type": "condo",
            "zipcode": "10022",
            "latitude": 40.758,
            "longitude": -73.985,
        },
    )
    assert response.status_code == 422


def test_predict_rejects_an_unknown_borough() -> None:
    """The contract is the five boroughs. An unknown one would one-hot encode
    to all zeros and still return a confident price."""
    response = client.post(
        "/predict",
        json={
            "beds": 2,
            "bath": 2.0,
            "propertysqft": 1200.0,
            "borough": "chicago",
            "type": "condo",
            "zipcode": "10022",
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


def test_rate_limit_applies_to_rejected_api_keys() -> None:
    """SECURITY.md scopes DoS out BECAUSE /predict is rate-limited, so the
    limit has to hold for the requests an attacker actually sends.

    It did not. `dependencies=[Depends(verify_api_key)]` on the route resolved
    before slowapi's decorator, so a wrong key returned 403 without reaching
    the counter -- measured, 15 wrong-key requests gave 15x 403 and zero 429,
    leaving key brute-force unbounded behind a policy that relied on the
    control. The check now runs inside the handler, after the limiter.
    """
    with reloaded_app(API_KEY="secret", PREDICT_RATE_LIMIT="3/minute") as m:
        client = TestClient(m.app)
        codes = [
            client.post(
                "/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "wrong"}
            ).status_code
            for _ in range(8)
        ]

    assert 403 in codes, codes
    assert 429 in codes, f"brute force unbounded: {codes}"
