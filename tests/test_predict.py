"""Tests for prediction module."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from src.config import MODELS_DIR, PRICE_ZONE_LABELS
from src.models.pipelines import (
    build_classification_pipeline,
    build_regression_pipeline,
)


@pytest.fixture
def mock_models(tmp_path: Path) -> Path:
    """Create and save minimal mock models for testing prediction.

    The mock label encoder is fitted on the zone STRINGS, so its class
    order is ALPHABETICAL ('High', 'Low', 'Medium', 'Very High') — exactly
    like the shipped artefact and deliberately DIFFERENT from the semantic
    ``PRICE_ZONE_LABELS`` config order. Any decode path that falls back to
    the config order mislabels 3 of the 4 classes and fails these tests;
    a mock whose orders coincide would leave the suite structurally blind
    to that bug (which is how it shipped the first time).
    """
    n = 50
    rng = np.random.RandomState(42)
    features = pd.DataFrame(
        {
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
        }
    )
    le = LabelEncoder()
    le.fit(PRICE_ZONE_LABELS)  # classes_ sorts alphabetically != config order
    assert list(le.classes_) != PRICE_ZONE_LABELS
    joblib.dump(le, tmp_path / "label_encoder.joblib")

    y_cls = rng.randint(0, 4, n)
    y_reg = rng.uniform(11, 15, n)

    clf = build_classification_pipeline(
        RandomForestClassifier(n_estimators=10, random_state=42)
    )
    clf.fit(features, y_cls)
    joblib.dump(clf, tmp_path / "price_zone_best.joblib")
    # The training frame doubles as a multi-class prediction batch in the
    # decode-order regression test below.
    joblib.dump(features, tmp_path / "train_features.joblib")

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
                "LOG_SQFT": 7.09,
                "ROOMS_PER_SQFT": 0.003,
                "DIST_MANHATTAN_CENTER": 0.5,
                "DIST_CENTRAL_PARK": 3.0,
                "DIST_NEAREST_SUBWAY": 0.5,
                "BOROUGH": "manhattan",
                "TYPE": "condo",
                "PROPERTY_CATEGORY": "residential",
                "ZIPCODE": "10022",
                "SUBLOCALITY": "midtown",
            }
        ]
    )


def test_predict_price_zone(mock_models: Path, _test_row: pd.DataFrame) -> None:
    import src.models.predict as pred_mod

    pred_mod._classifier_cache = None
    pred_mod._label_encoder_cache = None
    # Load from mock path
    clf = pred_mod.get_classifier(mock_models / "price_zone_best.joblib")
    le = pred_mod.get_label_encoder(mock_models / "label_encoder.joblib")
    results = pred_mod.predict_price_zone(_test_row)
    assert isinstance(results, list) and len(results) == 1
    result = results[0]
    assert "price_zone" in result
    assert "confidence" in result
    assert "probabilities" in result
    # Correctness, not membership: the decoded label must be the encoder's
    # name for the predicted class index (a bare membership check in
    # PRICE_ZONE_LABELS stayed green while 3 of 4 labels decoded wrong).
    expected = le.inverse_transform(clf.predict(_test_row))[0]
    assert result["price_zone"] == expected
    assert 0 <= result["confidence"] <= 1


def test_predict_decodes_via_encoder_classes_not_config_order(
    mock_models: Path,
) -> None:
    """Regression: serving decode order must equal ``label_encoder.classes_``.

    The mock encoder's alphabetical order differs from the config order, so
    a decode through ``PRICE_ZONE_LABELS`` mislabels every prediction except
    'Very High'. Asserts exact label equality across a batch spanning
    multiple predicted classes, and that the probabilities dict is keyed in
    encoder-class order.
    """
    import src.models.predict as pred_mod

    pred_mod._classifier_cache = None
    pred_mod._label_encoder_cache = None
    clf = pred_mod.get_classifier(mock_models / "price_zone_best.joblib")
    le = pred_mod.get_label_encoder(mock_models / "label_encoder.joblib")

    # Re-use the training frame as the batch — the overfit mock RF predicts
    # a spread of classes on it, so the check cannot pass vacuously.
    batch = joblib.load(mock_models / "train_features.joblib")
    predicted_idx = clf.predict(batch)
    assert len(set(predicted_idx.tolist())) >= 2, "batch must span classes"

    results = pred_mod.predict_price_zone(batch)
    expected_labels = le.inverse_transform(predicted_idx)
    for result, expected in zip(results, expected_labels, strict=True):
        assert result["price_zone"] == expected
        assert list(result["probabilities"]) == list(le.classes_)


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
    # A real NYC listing price, not just positive: expm1 of a plausible
    # log-price is positive for almost any broken model, so `> 0` passes on
    # output the pipeline should never produce.
    assert 10_000 < result["predicted_price"] < 100_000_000
    assert result["price_range"]["low"] < result["price_range"]["high"]


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
