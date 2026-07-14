"""Serving-parity regression tests: one decode path for API + dashboard.

The headline macro-F1 is measured under per-class THRESHOLD decoding
(``src/models/threshold.py``), and the train-time frequency cap must be
mirrored at serve time on EVERY surface. These tests pin the two halves
that historically diverged:

- the API decoded with bare argmax, so the published (threshold-tuned)
  metric was never the rule API users were actually served;
- the dashboard skipped ``apply_serving_cap``, so rare/unseen categories
  got the encoder's unseen default instead of the trained "other" encoding.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

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

ENCODER_CLASSES = ["High", "Low", "Medium", "Very High"]

# Chosen so threshold decoding and argmax DISAGREE: argmax picks "High"
# (0.42), but with High's threshold at 0.9 the adjusted scores are
# [0.42/0.9, 0.38/0.5, 0.12/0.5, 0.08/0.5] = [0.467, 0.76, 0.24, 0.16],
# so the threshold rule serves "Low" with confidence 0.38 (the probability
# of the served class, not proba.max()).
PROBA = [0.42, 0.38, 0.12, 0.08]
THRESHOLDS = {"High": 0.9, "Low": 0.5, "Medium": 0.5, "Very High": 0.5}


class _StubClf:
    def predict_proba(self, features: object) -> object:
        return np.asarray([PROBA])


class _StubReg:
    def predict(self, features: object) -> object:
        return np.asarray([14.0])


def _stubbed_client(
    monkeypatch: pytest.MonkeyPatch, thresholds: dict[str, float] | None
) -> TestClient:
    import api.main as m

    monkeypatch.setattr(m, "_get_classifier", lambda: _StubClf())
    monkeypatch.setattr(m, "_get_regressor", lambda: _StubReg())
    monkeypatch.setattr(m, "_get_capped_categories", dict)
    monkeypatch.setattr(m, "_get_zone_classes", lambda: ENCODER_CLASSES)
    monkeypatch.setattr(m, "_get_thresholds", lambda: thresholds)
    return TestClient(m.app)


def test_api_serves_threshold_decision_not_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/predict must apply the tuned per-class thresholds when they ship.

    With bare argmax the response would be High@0.42 — i.e. the serving
    rule the headline macro-F1 was measured under would never reach API
    users. Confidence must be the probability of the class actually
    served (0.38), not proba.max() (0.42), which describes a class the
    user was not shown.
    """
    resp = _stubbed_client(monkeypatch, THRESHOLDS).post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200, resp.text
    zone = resp.json()["zone"]
    assert zone["price_zone"] == "Low"
    assert zone["confidence"] == 0.38
    # The full probability vector is still reported unmodified.
    assert zone["probabilities"] == dict(zip(ENCODER_CLASSES, PROBA, strict=True))


def test_api_falls_back_to_argmax_without_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the optional thresholds artifact the decode is plain argmax
    with confidence = probability of the served class."""
    resp = _stubbed_client(monkeypatch, None).post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200, resp.text
    zone = resp.json()["zone"]
    assert zone["price_zone"] == "High"
    assert zone["confidence"] == 0.42


def _calls_in(source_path: Path) -> set[str]:
    """Names of all functions called anywhere in a module's source."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_applies_serving_cap() -> None:
    """The Streamlit script must mirror the train-time frequency cap.

    app.py is a top-level Streamlit script (module import executes st.*
    calls), so this is pinned structurally: the source must call
    apply_serving_cap on the features it builds, deriving the category
    sets from the shipped classifier. Skipping the cap silently sends
    rare/unseen SUBLOCALITY/ZIPCODE values down the encoder's unseen
    path — a train/serve skew invisible to every runtime test here.
    """
    calls = _calls_in(REPO_ROOT / "streamlit_app" / "app.py")
    assert "apply_serving_cap" in calls
    assert "learned_capped_categories" in calls


def test_both_surfaces_decode_via_served_zone() -> None:
    """API and dashboard must share ONE decode: src.models.threshold.served_zone.

    This is the structural pin against re-divergence — the exact rot that
    shipped once already (API on argmax while the dashboard used
    thresholds). If either surface stops calling served_zone, this fails
    before any behavioural test has to notice.
    """
    assert "served_zone" in _calls_in(REPO_ROOT / "api" / "main.py")
    assert "served_zone" in _calls_in(REPO_ROOT / "streamlit_app" / "app.py")
