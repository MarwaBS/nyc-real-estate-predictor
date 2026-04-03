"""Tests for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_numeric_features,
    add_target_variables,
    cap_categorical_cardinality,
)


def test_add_numeric_features_creates_expected_columns(sample_raw_data: pd.DataFrame) -> None:
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
    assert set(result["PRICE_ZONE"].dropna().unique()).issubset({"Low", "Medium", "High", "Very High"})


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
