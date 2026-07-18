"""Serving-parity regression tests: one decode path for API + dashboard.

The published macro-F1 is measured under argmax (``src/models/decode.py``),
which is exactly what both surfaces serve, and the train-time frequency cap
must be mirrored at serve time on EVERY surface. These tests pin the two
halves that historically diverged:

- the two surfaces decoded differently, so the published metric was not the
  rule API users were served;
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

# Deliberately not one-hot: the top two classes are close (0.42 vs 0.38) so
# an off-by-one in the class-order decode picks a different, wrong zone name
# instead of coincidentally landing on the right one.
PROBA = [0.42, 0.38, 0.12, 0.08]


class _StubClf:
    def predict_proba(self, features: object) -> object:
        return np.asarray([PROBA])


class _StubReg:
    def predict(self, features: object) -> object:
        return np.asarray([14.0])


def _stubbed_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import api.main as m

    monkeypatch.setattr(m, "_get_classifier", lambda: _StubClf())
    monkeypatch.setattr(m, "_get_regressor", lambda: _StubReg())
    monkeypatch.setattr(m, "_get_capped_categories", dict)
    monkeypatch.setattr(m, "_get_zone_classes", lambda: ENCODER_CLASSES)
    return TestClient(m.app)


def test_api_decodes_through_the_encoder_class_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/predict names the zone via the encoder's class order, and reports the
    served class's own probability as confidence.

    ENCODER_CLASSES is alphabetical, so index 0 is "High" — not the semantic
    config order's "Low". Decoding through the config list would return the
    wrong zone name for a correct prediction, which is precisely the rot that
    shipped once already.
    """
    resp = _stubbed_client(monkeypatch).post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200, resp.text
    zone = resp.json()["zone"]
    assert zone["price_zone"] == "High"
    assert zone["confidence"] == 0.42
    assert zone["probabilities"] == dict(zip(ENCODER_CLASSES, PROBA, strict=True))


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
    """API and dashboard must share ONE decode: src.models.decode.served_zone.

    This is the structural pin against re-divergence — the exact rot that
    shipped once already, when the two surfaces applied different decision
    rules to the same probabilities. If either surface stops calling
    served_zone, this fails before any behavioural test has to notice.
    """
    assert "served_zone" in _calls_in(REPO_ROOT / "api" / "main.py")
    assert "served_zone" in _calls_in(REPO_ROOT / "streamlit_app" / "app.py")
