"""Tests for per-class threshold optimization."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.threshold import (
    optimize_thresholds,
    predict_with_thresholds,
    served_zone,
)


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


def test_predict_with_thresholds_survives_zero_threshold() -> None:
    # A zero threshold must not blow up to inf/nan on division — the public
    # path clips like its internal twin. argmax should still return a valid
    # class for every row.
    proba = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
    labels = ["Low", "Medium", "High", "Very High"]
    thresholds = {"Low": 0.0, "Medium": 0.5, "High": 0.5, "Very High": 0.5}

    preds = predict_with_thresholds(proba, thresholds, labels)
    assert not np.isnan(preds).any()
    assert set(preds).issubset({0, 1, 2, 3})


def test_served_zone_confidence_is_probability_of_served_class() -> None:
    """Regression: in threshold mode the served zone is chosen by the per-class
    threshold logic, which may NOT be argmax(proba). Confidence must be the
    probability of the class ACTUALLY served (proba[zone_idx]), never
    proba.max() — otherwise the UI reports a confidence that belongs to a
    different class than the one it shows the user.
    """
    labels = ["Low", "Medium", "High", "Very High"]
    # Plain argmax is class 0 (0.40). A tight threshold on "Low" and a loose
    # one on "Very High" push the threshold-adjusted argmax onto class 3 (0.30).
    proba = np.array([0.40, 0.20, 0.10, 0.30])
    thresholds = {"Low": 0.9, "Medium": 0.5, "High": 0.5, "Very High": 0.1}

    served_idx = int(
        predict_with_thresholds(proba.reshape(1, -1), thresholds, labels)[0]
    )
    # Guard: the input must actually exercise the bug (served != argmax).
    assert served_idx != int(np.argmax(proba)), "test input no longer exercises the bug"

    zone_name, confidence = served_zone(proba, labels, thresholds)

    assert zone_name == labels[served_idx]
    assert confidence == pytest.approx(
        float(proba[served_idx])
    )  # 0.30, the served class
    assert confidence != pytest.approx(float(proba.max()))  # NOT 0.40, the argmax class


def test_served_zone_argmax_mode_uses_max_probability() -> None:
    """Without thresholds the served class IS argmax, so confidence is proba.max()."""
    labels = ["Low", "Medium", "High", "Very High"]
    proba = np.array([0.10, 0.55, 0.25, 0.10])

    zone_name, confidence = served_zone(proba, labels, thresholds=None)

    assert zone_name == "Medium"
    assert confidence == pytest.approx(0.55)


def test_threshold_tuning_handles_imbalanced_data() -> None:
    # Simulate Very High being rare
    proba = np.array(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.2, 0.2, 0.2, 0.4],  # Edge case: Very High
        ]
    )
    y_true = np.array([0, 1, 2, 3])
    labels = ["Low", "Medium", "High", "Very High"]

    thresholds, f1 = optimize_thresholds(proba, y_true, labels, resolution=10)
    assert f1 > 0.0
