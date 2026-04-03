"""Tests for sklearn pipeline construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.models.pipelines import (
    build_classification_pipeline,
    build_preprocessor,
    build_regression_pipeline,
)


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Minimal training data matching the pipeline's expected features."""
    n = 50
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
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
    y = rng.randint(0, 4, n)
    return df, y


def test_build_preprocessor_returns_column_transformer() -> None:
    preprocessor = build_preprocessor()
    assert hasattr(preprocessor, "transformers")


def test_classification_pipeline_fits_and_predicts(training_data: tuple) -> None:
    features, y = training_data
    pipeline = build_classification_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipeline.fit(features, y)
    preds = pipeline.predict(features)
    assert len(preds) == len(features)
    assert set(preds).issubset({0, 1, 2, 3})


def test_regression_pipeline_fits_and_predicts(training_data: tuple) -> None:
    features, y = training_data
    y_cont = np.random.RandomState(42).uniform(11, 15, len(features))  # LOG_PRICE range
    pipeline = build_regression_pipeline(RandomForestRegressor(n_estimators=10, random_state=42))
    pipeline.fit(features, y_cont)
    preds = pipeline.predict(features)
    assert len(preds) == len(features)
    assert all(np.isfinite(preds))


def test_pipeline_has_preprocessor_and_model_steps(training_data: tuple) -> None:
    features, y = training_data
    pipeline = build_classification_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipeline.fit(features, y)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
