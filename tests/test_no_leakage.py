"""DATA LEAKAGE PREVENTION TESTS.

These tests are the most critical in the suite. They ensure that
PRICE_PER_SQFT (derived from the target variable PRICE) is never
used as a training feature. The R2=0.997 in previous experiments
was fake because of this leakage.
"""
from __future__ import annotations

import pytest

from src.config import NUMERIC_FEATURES, ONEHOT_FEATURES, TARGET_ENCODED_FEATURES
from src.utils.validation import assert_no_leakage


def test_no_price_per_sqft_in_numeric_features() -> None:
    """PRICE_PER_SQFT must not be in the numeric feature list."""
    forbidden = {"PRICE_PER_SQFT", "price_per_sqft"}
    assert not forbidden & set(NUMERIC_FEATURES), (
        "DATA LEAKAGE: PRICE_PER_SQFT found in NUMERIC_FEATURES"
    )


def test_no_leaky_features_in_any_feature_list() -> None:
    """No target-derived feature in any feature list."""
    all_features = NUMERIC_FEATURES + ONEHOT_FEATURES + TARGET_ENCODED_FEATURES
    assert_no_leakage(all_features)


def test_assert_no_leakage_raises_on_bad_features() -> None:
    """Verify the guard function actually catches leakage."""
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        assert_no_leakage(["BEDS", "BATH", "PRICE_PER_SQFT"])


def test_assert_no_leakage_raises_on_log_price() -> None:
    """LOG_PRICE is also derived from target."""
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        assert_no_leakage(["BEDS", "LOG_PRICE"])


def test_assert_no_leakage_passes_clean_features() -> None:
    """Clean features should pass without error."""
    assert_no_leakage(["BEDS", "BATH", "PROPERTYSQFT", "BOROUGH"])
