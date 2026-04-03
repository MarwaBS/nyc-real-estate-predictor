"""Tests for prediction module."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.config import MODELS_DIR, PRICE_ZONE_LABELS
from src.models.pipelines import (
    build_classification_pipeline,
    build_regression_pipeline,
)


@pytest.fixture
def mock_models(tmp_path: Path) -> Path:
    """Create and save minimal mock models for testing prediction."""
    n = 50
    rng = np.random.RandomState(42)
    features = pd.DataFrame({
        "BEDS": rng.randint(1, 6, n),
        "BATH": rng.uniform(1, 4, n).round(1),
        "PROPERTYSQFT": rng.uniform(400, 4000, n),
        "TOTAL_ROOMS": rng.uniform(2, 10, n),
        "BED_BATH_RATIO": rng.uniform(0.5, 3.0, n),
        "LOG_SQFT": rng.uniform(6, 9, n),
        "ROOMS_PER_SQFT": rng.uniform(0.001, 0.01, n),
        "DIST_MANHATTAN_CENTER": rng.uniform(0, 30, n),
        "DIST_CENTRAL_PARK": rng.uniform(0, 30, n),
        "DIST_NEAREST_SUBWAY": rng.uniform(0, 5, n),
        "BOROUGH": rng.choice(["manhattan", "brooklyn", "queens"], n),
        "TYPE": rng.choice(["condo", "house", "co-op"], n),
        "PROPERTY_CATEGORY": rng.choice(["residential", "commercial"], n),
        "ZIPCODE": rng.choice(["10022", "11217", "10001"], n),
        "SUBLOCALITY": rng.choice(["midtown", "fort greene", "chelsea"], n),
    })
    y_cls = rng.randint(0, 4, n)
    y_reg = rng.uniform(11, 15, n)

    clf = build_classification_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    clf.fit(features, y_cls)
    joblib.dump(clf, tmp_path / "price_zone_best.joblib")

    reg = build_regression_pipeline(RandomForestRegressor(n_estimators=10, random_state=42))
    reg.fit(features, y_reg)
    joblib.dump(reg, tmp_path / "price_regressor_best.joblib")

    return tmp_path


@pytest.fixture
def _test_row() -> pd.DataFrame:
    return pd.DataFrame([{
        "BEDS": 2, "BATH": 2.0, "PROPERTYSQFT": 1200.0,
        "TOTAL_ROOMS": 4.0, "BED_BATH_RATIO": 1.0, "LOG_SQFT": 7.09,
        "ROOMS_PER_SQFT": 0.003, "DIST_MANHATTAN_CENTER": 0.5,
        "DIST_CENTRAL_PARK": 3.0, "DIST_NEAREST_SUBWAY": 0.5,
        "BOROUGH": "manhattan", "TYPE": "condo", "PROPERTY_CATEGORY": "residential",
        "ZIPCODE": "10022", "SUBLOCALITY": "midtown",
    }])


def test_predict_price_zone(mock_models: Path, _test_row: pd.DataFrame) -> None:
    import src.models.predict as pred_mod
    pred_mod._classifier_cache = None
    # Load from mock path
    pred_mod.get_classifier(mock_models / "price_zone_best.joblib")
    result = pred_mod.predict_price_zone(_test_row)
    assert "price_zone" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["price_zone"] in PRICE_ZONE_LABELS
    assert 0 <= result["confidence"] <= 1


def test_predict_price(mock_models: Path, _test_row: pd.DataFrame) -> None:
    import src.models.predict as pred_mod
    pred_mod._regressor_cache = None
    # Load from mock path
    pred_mod.get_regressor(mock_models / "price_regressor_best.joblib")
    result = pred_mod.predict_price(_test_row)
    assert "predicted_price" in result
    assert "price_range" in result
    assert result["predicted_price"] > 0
    assert result["price_range"]["low"] < result["price_range"]["high"]
