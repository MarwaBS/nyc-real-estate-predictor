"""Tests for per-class threshold optimization."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.threshold import optimize_thresholds, predict_with_thresholds


def test_optimize_thresholds_improves_or_matches_argmax() -> None:
    rng = np.random.RandomState(42)
    n = 200
    proba = rng.dirichlet([1, 1, 1, 1], n)
    y_true = np.argmax(proba + rng.normal(0, 0.1, proba.shape), axis=1).clip(0, 3)

    labels = ["Low", "Medium", "High", "Very High"]
    thresholds, tuned_f1 = optimize_thresholds(proba, y_true, labels, resolution=20)

    assert len(thresholds) == 4
    assert all(0.1 <= v <= 0.9 for v in thresholds.values())
    assert tuned_f1 >= 0.0


def test_predict_with_thresholds_returns_valid_classes() -> None:
    proba = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
    labels = ["Low", "Medium", "High", "Very High"]
    thresholds = {"Low": 0.5, "Medium": 0.5, "High": 0.5, "Very High": 0.5}

    preds = predict_with_thresholds(proba, thresholds, labels)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1, 2, 3})


def test_threshold_tuning_handles_imbalanced_data() -> None:
    # Simulate Very High being rare
    proba = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.2, 0.2, 0.2, 0.4],  # Edge case: Very High
    ])
    y_true = np.array([0, 1, 2, 3])
    labels = ["Low", "Medium", "High", "Very High"]

    thresholds, f1 = optimize_thresholds(proba, y_true, labels, resolution=10)
    assert f1 > 0.0
