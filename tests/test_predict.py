"""Tests for prediction module."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.models.pipelines import build_regression_pipeline


@pytest.fixture
def mock_models(tmp_path: Path) -> Path:
    """The one model that ships: a regressor over the shipped feature frame."""
    n = 50
    rng = np.random.RandomState(42)
    features = pd.DataFrame(
        {
            "BEDS": rng.randint(1, 6, n),
            "BATH": rng.uniform(1, 4, n).round(1),
            "PROPERTYSQFT": rng.uniform(400, 4000, n),
            "TOTAL_ROOMS": rng.uniform(2, 10, n),
            "BED_BATH_RATIO": rng.uniform(0.5, 3.0, n),
            "ROOMS_PER_SQFT": rng.uniform(0.001, 0.01, n),
            "DIST_MANHATTAN_CENTER": rng.uniform(0, 30, n),
            "DIST_CENTRAL_PARK": rng.uniform(0, 30, n),
            "BOROUGH": rng.choice(["manhattan", "brooklyn", "queens"], n),
            "TYPE": rng.choice(["condo", "house", "co-op"], n),
            "ZIPCODE": rng.choice(["10022", "11217", "10001"], n),
            "SUBLOCALITY": rng.choice(["midtown", "fort greene", "chelsea"], n),
        }
    )
    y_reg = rng.uniform(11, 15, n)

    reg = build_regression_pipeline(
        RandomForestRegressor(n_estimators=10, random_state=42)
    )
    reg.fit(features, y_reg)
    joblib.dump(reg, tmp_path / "price_regressor_best.joblib")

    return tmp_path


@pytest.fixture
def _test_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "BEDS": 2,
                "BATH": 2.0,
                "PROPERTYSQFT": 1200.0,
                "TOTAL_ROOMS": 4.0,
                "BED_BATH_RATIO": 1.0,
                "ROOMS_PER_SQFT": 0.003,
                "DIST_MANHATTAN_CENTER": 0.5,
                "DIST_CENTRAL_PARK": 3.0,
                "BOROUGH": "manhattan",
                "TYPE": "condo",
                "ZIPCODE": "10022",
                "SUBLOCALITY": "midtown",
            }
        ]
    )


def test_predict_price_zone(mock_models: Path, _test_row: pd.DataFrame) -> None:
    """The zone is the predicted price, bucketed through the shared decode."""
    import src.models.predict as pred_mod
    from src.models.decode import zone_for_price

    pred_mod._regressor_cache = None
    pred_mod.get_regressor(mock_models / "price_regressor_best.joblib")

    results = pred_mod.predict_price_zone(_test_row)
    assert isinstance(results, list) and len(results) == 1
    # Correctness, not membership: the zone must be the bucket the served
    # price falls in, not merely a valid label.
    price = pred_mod.predict_price(_test_row)[0]["predicted_price"]
    assert results[0]["price_zone"] == zone_for_price(price)


def test_predict_price(mock_models: Path, _test_row: pd.DataFrame) -> None:
    import src.models.predict as pred_mod

    pred_mod._regressor_cache = None
    # Load from mock path
    pred_mod.get_regressor(mock_models / "price_regressor_best.joblib")
    results = pred_mod.predict_price(_test_row)
    assert isinstance(results, list) and len(results) == 1
    result = results[0]
    assert "predicted_price" in result
    assert "price_range" in result
    # Bounded by the training target's own range rather than `> 0`: the model
    # predicts log-price and the caller un-logs it, so a broken transform
    # surfaces as a wildly out-of-scale number that is still positive. The
    # cleaned dataset's PRICE spans $2,494 to $4,483,000 (the IQR cap), so a
    # served price outside one order of magnitude either side of that range is
    # a scaling bug, not a listing.
    assert 250 < result["predicted_price"] < 45_000_000
    assert result["price_range"]["low"] < result["price_range"]["high"]


def test_served_band_reproduces_from_the_rounded_price(
    _test_row: pd.DataFrame, monkeypatch
) -> None:
    """low/high must be the displayed price times the multipliers, not the
    unrounded prediction.

    The prediction is pinned so the difference crosses a $100 boundary: an
    unrounded price near $1,000,049 (log1p) rounds to $1,000,000, and at the
    high multiplier the two bases round to endpoints $100 apart. A test whose
    price happens not to straddle a boundary passes on the bug — the exact
    knife-edge this file has been burned by before."""
    import src.models.predict as pred_mod

    unrounded = 1_000_049.0
    interval = {
        "low_multiplier": 0.6108,
        "high_multiplier": 1.5368,
        "target_coverage": 0.8,
    }

    class _Stub:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), np.log1p(unrounded))

    monkeypatch.setattr(pred_mod, "_regressor_cache", _Stub())
    monkeypatch.setattr(pred_mod, "_price_interval", interval)

    result = pred_mod.predict_price(_test_row)[0]
    shown = result["predicted_price"]
    assert shown == 1_000_000  # rounded to $100

    assert result["price_range"]["high"] == round(
        shown * interval["high_multiplier"], -2
    )
    # And this is provably NOT the unrounded band — the defect being pinned.
    assert result["price_range"]["high"] != round(
        unrounded * interval["high_multiplier"], -2
    )


def test_predict_returns_one_entry_per_row(
    mock_models: Path, _test_row: pd.DataFrame
) -> None:
    """The list contract holds for batches, not just single rows."""
    import src.models.predict as pred_mod

    pred_mod._regressor_cache = None
    pred_mod.get_regressor(mock_models / "price_regressor_best.joblib")
    batch = pd.concat([_test_row, _test_row], ignore_index=True)
    results = pred_mod.predict_price(batch)
    assert isinstance(results, list) and len(results) == 2


def test_version_mismatch_is_refused(tmp_path: Path, monkeypatch) -> None:
    """A model artefact from another sklearn version must be REFUSED.

    Regression guard for the MODEL_CARD postmortem: sklearn used to emit
    InconsistentVersionWarning and keep serving silently-corrupt
    predictions ($2 Manhattan condos). The loader now promotes that
    warning to ModelVersionError. Simulated by monkeypatching joblib.load
    to emit the warning — producing a genuinely cross-version pickle would
    require a second sklearn install.
    """
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    import src.models.predict as pred_mod

    artefact = tmp_path / "model.joblib"
    artefact.write_bytes(b"placeholder")

    def _fake_load(path):
        # InconsistentVersionWarning has a keyword-only constructor — emit a
        # properly-constructed instance, exactly as sklearn's unpickler does.
        warnings.warn(
            InconsistentVersionWarning(
                estimator_name="FakeEstimator",
                current_sklearn_version="9.9.9",
                original_sklearn_version="1.0.0",
            ),
            stacklevel=2,
        )
        return object()

    monkeypatch.setattr(pred_mod.joblib, "load", _fake_load)
    with pytest.raises(pred_mod.ModelVersionError, match="refusing to load"):
        pred_mod._load_model(artefact)


def test_xgboost_version_mismatch_is_refused(tmp_path: Path, monkeypatch) -> None:
    """XGBoost's cross-version load warning must be refused like sklearn's.

    The first hardening pass promoted only InconsistentVersionWarning;
    a booster serialized by another xgboost version warned and served on.
    Simulated with the UserWarning xgboost emits at unpickle time (verbatim
    first line, captured from a real 2.1.4-artefact load under 3.2.0).
    """
    import warnings

    import src.models.predict as pred_mod

    artefact = tmp_path / "model.joblib"
    artefact.write_bytes(b"placeholder")

    def _fake_load(path):
        warnings.warn(
            "[00:00:00] WARNING: error_msg.h:83: If you are loading a "
            "serialized model (like pickle in Python, RDS in R) or\n"
            "configuration generated by an older version of XGBoost, please "
            "export the model by calling `Booster.save_model`.",
            UserWarning,
            stacklevel=2,
        )
        return object()

    monkeypatch.setattr(pred_mod.joblib, "load", _fake_load)
    with pytest.raises(pred_mod.ModelVersionError, match="refusing to load"):
        pred_mod._load_model(artefact)
