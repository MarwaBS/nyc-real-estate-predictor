"""Tests for the data cleaning pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import (
    cap_outliers,
    clean_pipeline,
    deduplicate,
    impute_missing,
    normalize_borough,
    normalize_text_columns,
    normalize_zipcode,
)


def test_deduplicate_removes_exact_dupes(sample_raw_data: pd.DataFrame) -> None:
    df = pd.concat([sample_raw_data, sample_raw_data.iloc[:2]], ignore_index=True)
    result = deduplicate(df)
    assert len(result) <= len(df)


def test_impute_missing_fills_nulls(sample_raw_data: pd.DataFrame) -> None:
    df = sample_raw_data.copy()
    df.loc[0, "BATH"] = np.nan
    df.loc[1, "BEDS"] = np.nan
    result = impute_missing(df)
    assert result["BATH"].isna().sum() == 0
    assert result["BEDS"].isna().sum() == 0


def test_cap_outliers_clips_extreme_values() -> None:
    df = pd.DataFrame({"PRICE": [100, 200, 300, 400, 100_000_000]})
    result = cap_outliers(df, columns=["PRICE"], factor=3.0)
    assert result["PRICE"].max() < 100_000_000


def test_normalize_text_lowercases() -> None:
    df = pd.DataFrame({"TYPE": ["CONDO", "  House  ", "Co-Op"]})
    result = normalize_text_columns(df)
    assert list(result["TYPE"]) == ["condo", "house", "co-op"]


def test_normalize_borough_maps_counties() -> None:
    df = pd.DataFrame({"BOROUGH": ["kings county", "richmond county", "manhattan"]})
    result = normalize_borough(df)
    assert list(result["BOROUGH"]) == ["brooklyn", "staten island", "manhattan"]


def test_normalize_zipcode_extracts_5_digits() -> None:
    df = pd.DataFrame({"ZIPCODE": ["10022.0", "11217", "0073", "99999"]})
    result = normalize_zipcode(df)
    assert list(result["ZIPCODE"]) == ["10022", "11217", "00000", "99999"]


def test_clean_pipeline_produces_valid_output(sample_raw_data: pd.DataFrame) -> None:
    result = clean_pipeline(sample_raw_data)
    assert len(result) > 0
    assert (result["PRICE"] > 0).all()
    assert result["PROPERTYSQFT"].isna().sum() == 0
