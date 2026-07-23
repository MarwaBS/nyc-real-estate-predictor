"""Tests for sklearn pipeline construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.config import NUMERIC_FEATURES, ONEHOT_FEATURES, TARGET_ENCODED_FEATURES
from src.models.pipelines import (
    build_preprocessor,
    build_regression_pipeline,
)


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Minimal frame carrying exactly the configured feature columns, so a
    feature added to or dropped from src.config changes this fixture too."""
    n = 50
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            **{c: rng.uniform(0.5, 30, n) for c in NUMERIC_FEATURES},
            **{c: rng.choice(["a", "b", "c"], n) for c in ONEHOT_FEATURES},
            **{c: rng.choice(["p", "q", "r"], n) for c in TARGET_ENCODED_FEATURES},
        }
    )
    y = rng.randint(0, 4, n)
    return df, y


def test_build_preprocessor_returns_column_transformer() -> None:
    preprocessor = build_preprocessor()
    assert hasattr(preprocessor, "transformers")


def test_regression_pipeline_fits_and_predicts(training_data: tuple) -> None:
    features, y = training_data
    y_cont = np.random.RandomState(42).uniform(11, 15, len(features))  # LOG_PRICE range
    pipeline = build_regression_pipeline(
        RandomForestRegressor(n_estimators=10, random_state=42)
    )
    pipeline.fit(features, y_cont)
    preds = pipeline.predict(features)
    assert len(preds) == len(features)
    assert all(np.isfinite(preds))


def test_pipeline_has_preprocessor_and_model_steps(training_data: tuple) -> None:
    features, _ = training_data
    y_cont = np.random.RandomState(42).uniform(11, 15, len(features))
    pipeline = build_regression_pipeline(
        RandomForestRegressor(n_estimators=10, random_state=42)
    )
    pipeline.fit(features, y_cont)
    assert "preprocessor" in pipeline.named_steps
    assert "regressor" in pipeline.named_steps
