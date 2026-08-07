"""PRICE_PER_SQFT is derived from the target PRICE; used as a feature it
produced a fake R2=0.997 (ADR-001). These tests keep it out of training."""

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


@pytest.mark.parametrize(
    "leaky",
    ["PRICE", "price", "Price", "SALE PRICE", "PRICE_ZONE_ENCODED", "price_bucket"],
)
def test_assert_no_leakage_catches_the_target_itself(leaky: str) -> None:
    """The guard must reject the raw target, not just its derived spellings.

    The enumerated-spellings version passed ``["BEDS", "PRICE"]``, the guard
    against leakage certified the most direct leak there is. Every case here
    fails against that implementation.
    """
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        assert_no_leakage(["BEDS", leaky])
