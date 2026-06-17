"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_numeric_features,
    add_target_variables,
    apply_serving_cap,
    cap_categorical_cardinality,
    learned_capped_categories,
)


def test_add_numeric_features_creates_expected_columns(
    sample_raw_data: pd.DataFrame,
) -> None:
    result = add_numeric_features(sample_raw_data)
    assert "TOTAL_ROOMS" in result.columns
    assert "BED_BATH_RATIO" in result.columns
    assert "LOG_SQFT" in result.columns
    assert "ROOMS_PER_SQFT" in result.columns


def test_total_rooms_is_beds_plus_bath(sample_raw_data: pd.DataFrame) -> None:
    result = add_numeric_features(sample_raw_data)
    expected = sample_raw_data["BEDS"] + sample_raw_data["BATH"]
    pd.testing.assert_series_equal(result["TOTAL_ROOMS"], expected, check_names=False)


def test_log_sqft_is_positive(sample_raw_data: pd.DataFrame) -> None:
    result = add_numeric_features(sample_raw_data)
    assert (result["LOG_SQFT"] > 0).all()


def test_add_target_variables_creates_price_zone(sample_raw_data: pd.DataFrame) -> None:
    result = add_target_variables(sample_raw_data)
    assert "PRICE_ZONE" in result.columns
    assert "LOG_PRICE" in result.columns
    assert "SQFT_CATEGORY" in result.columns
    assert set(result["PRICE_ZONE"].dropna().unique()).issubset(
        {"Low", "Medium", "High", "Very High"}
    )


def test_log_price_is_log1p(sample_raw_data: pd.DataFrame) -> None:
    result = add_target_variables(sample_raw_data)
    expected = np.log1p(sample_raw_data["PRICE"])
    np.testing.assert_array_almost_equal(result["LOG_PRICE"].values, expected.values)


def test_cap_cardinality_limits_categories() -> None:
    df = pd.DataFrame({"COL": [f"cat_{i}" for i in range(100)]})
    result = cap_categorical_cardinality(df, columns=["COL"], max_categories=10)
    unique = result["COL"].unique()
    # 10 real categories + "other"
    assert len(unique) <= 11


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


def _real_classifier():
    import joblib

    from src.config import MODELS_DIR

    path = MODELS_DIR / "price_zone_best.joblib"
    if not path.exists():
        pytest.skip("shipped classifier artefact not present")
    return joblib.load(path)


def test_learned_capped_categories_finds_only_capped_columns() -> None:
    """The helper must surface exactly the columns the model frequency-capped (an
    'other' bucket was learned) — ZIPCODE/SUBLOCALITY — and not low-cardinality
    columns (BOROUGH/TYPE) where no 'other' exists."""
    clf = _real_classifier()
    known = learned_capped_categories(clf)
    assert set(known) == {"ZIPCODE", "SUBLOCALITY"}, set(known)
    for cats in known.values():
        assert "other" in {str(c) for c in cats}


def test_serving_cap_closes_train_serve_skew() -> None:
    """End-to-end parity on the SHIPPED model: an unseen ZIPCODE/SUBLOCALITY, after
    the serving cap, predicts identically to the explicit 'other' bucket — and
    differently from the un-capped raw input (proving the skew was real and is now
    closed)."""
    clf = _real_classifier()
    known = learned_capped_categories(clf)

    base = {
        "BEDS": 2,
        "BATH": 2.0,
        "PROPERTYSQFT": 1200.0,
        "TOTAL_ROOMS": 4.0,
        "BED_BATH_RATIO": 1.0,
        "LOG_SQFT": float(np.log1p(1200.0)),
        "ROOMS_PER_SQFT": 4.0 / 1200.0,
        "DIST_MANHATTAN_CENTER": 1.0,
        "DIST_CENTRAL_PARK": 1.0,
        "DIST_NEAREST_SUBWAY": 1.0,
        "BOROUGH": "manhattan",
        "TYPE": "condo",
        "PROPERTY_CATEGORY": "residential",
        "ZIPCODE": "99999",  # unseen
        "SUBLOCALITY": "nowhere_ville",  # unseen
    }
    raw = pd.DataFrame([base])
    capped = apply_serving_cap(raw, known)
    explicit_other = raw.copy()
    explicit_other["ZIPCODE"] = "other"
    explicit_other["SUBLOCALITY"] = "other"

    p_capped = clf.predict_proba(capped)[0]
    p_other = clf.predict_proba(explicit_other)[0]
    p_raw = clf.predict_proba(raw)[0]

    # Parity: capped unseen == explicit "other".
    np.testing.assert_allclose(p_capped, p_other, atol=1e-9)
    # The skew was real: raw (un-capped) differs from the trained "other" bucket.
    assert not np.allclose(p_raw, p_other, atol=1e-6)
