"""Tests for data validation module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.validation import assert_no_leakage, validate_cleaned_data


def test_validate_passes_on_clean_data(sample_raw_data: pd.DataFrame) -> None:
    issues = validate_cleaned_data(sample_raw_data)
    assert len(issues) == 0


def test_validate_catches_missing_columns() -> None:
    df = pd.DataFrame({"FOO": [1, 2]})
    issues = validate_cleaned_data(df)
    assert any("Missing required" in i for i in issues)


def test_validate_catches_negative_price(sample_raw_data: pd.DataFrame) -> None:
    df = sample_raw_data.copy()
    df.loc[0, "PRICE"] = -100
    issues = validate_cleaned_data(df)
    assert any("non-positive" in i for i in issues)


def test_validate_catches_negative_sqft(sample_raw_data: pd.DataFrame) -> None:
    df = sample_raw_data.copy()
    df.loc[0, "PROPERTYSQFT"] = -50
    issues = validate_cleaned_data(df)
    assert any("non-positive" in i for i in issues)


def test_validate_catches_out_of_range_latitude(sample_raw_data: pd.DataFrame) -> None:
    df = sample_raw_data.copy()
    df.loc[0, "LATITUDE"] = 50.0  # Not NYC
    issues = validate_cleaned_data(df)
    assert any("LATITUDE" in i for i in issues)


def test_validate_catches_out_of_range_longitude(sample_raw_data: pd.DataFrame) -> None:
    df = sample_raw_data.copy()
    df.loc[0, "LONGITUDE"] = -80.0  # Not NYC
    issues = validate_cleaned_data(df)
    assert any("LONGITUDE" in i for i in issues)


def test_validate_catches_null_price(sample_raw_data: pd.DataFrame) -> None:
    df = sample_raw_data.copy()
    df.loc[0, "PRICE"] = np.nan
    issues = validate_cleaned_data(df)
    assert any("null" in i for i in issues)
