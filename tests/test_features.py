"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_numeric_features,
    add_target_variables,
    apply_serving_cap,
    apply_top_categories,
    fit_top_categories,
    learned_capped_categories,
)


def test_add_numeric_features_creates_expected_columns(
    sample_raw_data: pd.DataFrame,
) -> None:
    result = add_numeric_features(sample_raw_data)
    assert "TOTAL_ROOMS" in result.columns
    assert "BED_BATH_RATIO" in result.columns
    assert "ROOMS_PER_SQFT" in result.columns


def test_total_rooms_is_beds_plus_bath(sample_raw_data: pd.DataFrame) -> None:
    result = add_numeric_features(sample_raw_data)
    expected = sample_raw_data["BEDS"] + sample_raw_data["BATH"]
    pd.testing.assert_series_equal(result["TOTAL_ROOMS"], expected, check_names=False)


def test_add_target_variables_creates_price_zone(sample_raw_data: pd.DataFrame) -> None:
    result = add_target_variables(sample_raw_data)
    assert "PRICE_ZONE" in result.columns
    assert "LOG_PRICE" in result.columns
    assert set(result["PRICE_ZONE"].dropna().unique()).issubset(
        {"Low", "Medium", "High", "Very High"}
    )


def test_log_price_is_log1p(sample_raw_data: pd.DataFrame) -> None:
    result = add_target_variables(sample_raw_data)
    expected = np.log1p(sample_raw_data["PRICE"])
    np.testing.assert_array_almost_equal(result["LOG_PRICE"].values, expected.values)


def test_fit_top_categories_defaults_to_keeping_fifty() -> None:
    """The shipped path (run_protocol) calls fit_top_categories without
    max_categories, so the DEFAULT shapes every ZIPCODE and SUBLOCALITY
    encoding. Exact count, not <= 51: a bound is satisfied by any lower cap."""
    df = pd.DataFrame({"COL": [f"cat_{i}" for i in range(60)]})

    vocab = fit_top_categories(df, columns=["COL"])

    assert len(vocab["COL"]) == 50
    result = apply_top_categories(df, vocab)
    assert (result["COL"] == "other").sum() == 10


# ── Train/serve cap parity ───────────────────────────────────────────────────


def test_apply_serving_cap_maps_unknown_to_other() -> None:
    known = {"ZIPCODE": {"10021", "10025", "other"}}
    df = pd.DataFrame(
        {"ZIPCODE": ["10021", "99999", "other"], "BOROUGH": ["manhattan"] * 3}
    )
    out = apply_serving_cap(df, known)
    # In-set values and the literal "other" survive; the unseen one is remapped.
    assert out["ZIPCODE"].tolist() == ["10021", "other", "other"]
    # A column not in `known` (never capped at train time) is left untouched.
    assert out["BOROUGH"].tolist() == ["manhattan"] * 3


def _real_model():
    import joblib

    from src.config import MODELS_DIR

    path = MODELS_DIR / "price_regressor_best.joblib"
    if not path.exists():
        pytest.skip("shipped model artefact not present")
    return joblib.load(path)


def test_learned_capped_categories_finds_only_capped_columns() -> None:
    """The helper must surface exactly the columns that learned an 'other' bucket.

    Asserted as an invariant rather than a hardcoded column list, because which
    columns get capped is a property of the data, not of the helper: the cap
    keeps the top 50 categories, and only ZIPCODE (178 distinct) exceeds that.
    SUBLOCALITY has 21 distinct values in the raw export, so it is never capped
    and must NOT appear -- listing it here previously described a pre-cleaned
    CSV that no code in this repo produced.
    """
    clf = _real_model()
    known = learned_capped_categories(clf)

    assert "ZIPCODE" in known
    for column, cats in known.items():
        assert "other" in {str(c) for c in cats}, column
    # Low-cardinality columns have no 'other' to learn.
    assert {"BOROUGH", "TYPE", "SUBLOCALITY"}.isdisjoint(known)


def test_serving_cap_maps_unseen_categories_to_the_trained_bucket() -> None:
    """Serving must encode an unseen ZIP the way training encoded rare ones.

    Asserts PARITY only. An earlier version also asserted the un-capped input
    predicts differently -- "the skew was real" -- which held for the deleted
    classifier's probability vector but does NOT hold for the regressor:
    measured on the shipped model, raw and "other" differ by 2e-15, because
    TargetEncoder's unseen fallback is the global mean and this model's
    "other" encoding sits on it. The cap is kept for train/serve parity (a
    retrain can move those apart), not because it currently changes an answer,
    and this test no longer claims otherwise.
    """
    model = _real_model()
    known = learned_capped_categories(model)

    base = {
        "BEDS": 2,
        "BATH": 2.0,
        "PROPERTYSQFT": 1200.0,
        "TOTAL_ROOMS": 4.0,
        "BED_BATH_RATIO": 1.0,
        "ROOMS_PER_SQFT": 4.0 / 1200.0,
        "DIST_MANHATTAN_CENTER": 1.0,
        "DIST_CENTRAL_PARK": 1.0,
        "BOROUGH": "manhattan",
        "TYPE": "condo",
        "ZIPCODE": "99999",  # unseen
        "SUBLOCALITY": "nowhere_ville",  # unseen
    }
    raw = pd.DataFrame([base])
    capped = apply_serving_cap(raw, known)
    explicit_other = raw.copy()
    explicit_other["ZIPCODE"] = "other"

    np.testing.assert_allclose(
        model.predict(capped)[0], model.predict(explicit_other)[0], atol=1e-9
    )
