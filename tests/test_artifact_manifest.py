"""The committed serving artifacts must match models/MANIFEST.sha256.

This is the recorded-AND-compared half of the model registry: the live
demo served April's misattributed thresholds for three months precisely
because nothing machine-checked which artifact vintage was where. The
manifest pins the exact bytes of every serving artifact; this test
enforces it in CI, and the deploy workflow ships the same files to the
HF Space, where the weekly drift guard re-compares them. Regenerate the
manifest ONLY as part of a deliberate retrain commit
(sha256 of each file in models/, one line per artifact).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MANIFEST = MODELS_DIR / "MANIFEST.sha256"

# The exact set the API + dashboard load at serve time (loader defaults in
# src/models/predict.py + drift baseline). Manifest must cover exactly this
# set — a served file missing from the manifest is ungoverned vintage.
SERVING_ARTIFACTS = {
    "price_zone_best.joblib",
    "price_regressor_best.joblib",
    "optimal_thresholds.joblib",
    "label_encoder.joblib",
    "drift_baseline.json",
}


def _manifest_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        entries[name.strip()] = digest
    return entries


def test_manifest_covers_exactly_the_serving_set() -> None:
    assert set(_manifest_entries()) == SERVING_ARTIFACTS


def test_every_serving_artifact_matches_its_manifest_hash() -> None:
    for name, expected in _manifest_entries().items():
        path = MODELS_DIR / name
        assert path.exists(), f"{name} is in the manifest but not in models/"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{name}: sha256 {actual} != manifest {expected} — the committed "
            f"artifact changed without a manifest update (or vice versa). "
            f"Regenerate the manifest only as part of a deliberate retrain."
        )


def test_committed_thresholds_are_the_canonical_july_values() -> None:
    """Pins the exact failure the 2026-07-14 re-audit caught live: the Space
    served the CHANGELOG-convicted misattributed thresholds over April
    models (21.4% of a probe grid decoded differently). The canonical
    values come from the 2026-07-04 run recorded in
    reports/training_metrics.json."""
    thresholds = joblib.load(MODELS_DIR / "optimal_thresholds.joblib")
    assert thresholds == {
        "High": 0.361,
        "Low": 0.9,
        "Medium": 0.492,
        "Very High": 0.5,
    }


def test_committed_encoder_is_alphabetical_and_complete() -> None:
    encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    assert [str(c) for c in encoder.classes_] == [
        "High",
        "Low",
        "Medium",
        "Very High",
    ]
