"""Tests for model evaluation module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    evaluate_classifier,
    evaluate_fairness_by_group,
    evaluate_regressor,
)


def test_evaluate_classifier_returns_expected_keys() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1])
    metrics = evaluate_classifier(y_true, y_pred, labels=["Low", "Medium", "High"])
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "cohen_kappa" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics


def test_evaluate_classifier_perfect_score() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    metrics = evaluate_classifier(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_evaluate_regressor_returns_expected_keys() -> None:
    y_true = np.array([12.0, 13.0, 14.0, 11.0])
    y_pred = np.array([12.1, 12.9, 14.2, 10.8])
    metrics = evaluate_regressor(y_true, y_pred, log_target=True)
    assert "r2" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "rmse_usd" in metrics
    assert "mape" in metrics


def test_evaluate_regressor_perfect_score() -> None:
    y = np.array([12.0, 13.0, 14.0])
    metrics = evaluate_regressor(y, y, log_target=False)
    assert metrics["r2"] == 1.0
    assert metrics["rmse"] == 0.0


def test_evaluate_fairness_by_group() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
    groups = pd.Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
    result = evaluate_fairness_by_group(y_true, y_pred, groups)
    assert "A" in result
    assert "B" in result
    assert all(0 <= v <= 1 for v in result.values())
