"""SHAP explainability helpers.

``run_training.py`` records `regression.shap_top10` from these, and the
explainability notebook renders them, but nothing executed them under test -
`src/models/explain.py` sat at 0% coverage, so a shape-handling regression
would have surfaced only during a training run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.models.explain import (
    compute_shap_values,
    get_top_features_for_prediction,
    global_feature_importance,
)


@pytest.fixture(scope="module")
def fitted() -> tuple:
    """A small tree model over three features with one dominant signal."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=200),
            "b": rng.normal(size=200),
            "c": rng.normal(size=200),
        }
    )
    y = 5.0 * X["a"] + 0.1 * X["b"]  # 'a' dominates, 'c' is noise
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    shap_values, explainer = compute_shap_values(model, X, max_samples=100)
    return model, X, shap_values, explainer


def test_compute_shap_values_returns_one_row_per_sampled_input(fitted: tuple) -> None:
    _, _, shap_values, _ = fitted
    assert np.asarray(shap_values).shape[0] == 100


def test_global_importance_ranks_the_dominant_feature_first(fitted: tuple) -> None:
    """The ranking must reflect the signal, not just return without error -
    a broken shape reduction would still produce a sorted frame."""
    _, X, shap_values, _ = fitted
    importance = global_feature_importance(shap_values, list(X.columns))
    assert list(importance.columns) == ["feature", "mean_abs_shap"]
    assert importance.iloc[0]["feature"] == "a"
    assert importance.iloc[-1]["feature"] == "c"


def test_top_features_for_a_prediction_are_ranked_by_magnitude(fitted: tuple) -> None:
    _, X, shap_values, explainer = fitted
    top = get_top_features_for_prediction(
        explainer, shap_values, list(X.columns), idx=0, top_n=2
    )
    assert [f["feature"] for f in top][0] == "a"
    magnitudes = [abs(f["shap_value"]) for f in top]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(f["direction"] in {"+", "-"} for f in top)


def test_mismatched_feature_names_raise_rather_than_mis_pair(fitted: tuple) -> None:
    """The strict zip guards against silently pairing values to the wrong
    names, which would produce a plausible but wrong importance table."""
    _, X, shap_values, explainer = fitted
    with pytest.raises(ValueError):
        get_top_features_for_prediction(explainer, shap_values, ["a", "b"], idx=0)
