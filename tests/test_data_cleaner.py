"""Tests for the data cleaning pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import (
    cap_outliers,
    clean_pipeline,
    deduplicate,
    derive_borough,
    derive_zipcode,
    fit_cap_bounds,
    impute_missing,
    normalize_text_columns,
    normalize_type,
)


@pytest.fixture
def raw_shaped_data() -> pd.DataFrame:
    """A frame with the RAW Kaggle columns -- no BOROUGH, no ZIPCODE.

    ``sample_raw_data`` in conftest is named "raw" but already carries BOROUGH
    and ZIPCODE, so it cannot exercise the derivation at all. Rows 1-3 resolve
    from a different source column each, and row 4 reproduces the shifted-column
    records in the real snapshot (LOCALITY "United States",
    ADMINISTRATIVE_AREA_LEVEL_2 holding a ZIP) that no source can resolve.
    """
    return pd.DataFrame(
        {
            "PRICE": [500_000, 750_000, 1_200_000, 300_000],
            "BEDS": [2, 3, 4, 1],
            "BATH": [1.0, 2.0, 3.0, 1.0],
            "PROPERTYSQFT": [800.0, 1_400.0, 2_200.0, 500.0],
            "LATITUDE": [40.758, 40.689, 40.650, 40.820],
            "LONGITUDE": [-73.985, -73.944, -73.949, -73.874],
            "TYPE": ["condo for sale", "house", "condo", "co-op"],
            "SUBLOCALITY": ["kings county", "not a borough", "New York", "New York"],
            "LOCALITY": [
                "United States",
                "richmond county",
                "United States",
                "United States",
            ],
            "ADMINISTRATIVE_AREA_LEVEL_2": ["x", "y", "bronx county", "11214"],
            "STATE": [
                "Brooklyn, NY 11217",
                "Staten Island, NY 10312",
                "Bronx, NY 10473",
                "Brooklyn, NY 11214",
            ],
        }
    )


def test_derive_borough_falls_back_across_source_columns(
    raw_shaped_data: pd.DataFrame,
) -> None:
    """Each of the three sources must be able to resolve a row on its own."""
    result = derive_borough(raw_shaped_data)
    assert list(result["BOROUGH"][:3]) == ["brooklyn", "staten island", "the bronx"]


def test_derive_borough_leaves_shifted_column_rows_null(
    raw_shaped_data: pd.DataFrame,
) -> None:
    """A record whose geocode fields are misaligned must not be guessed at."""
    result = derive_borough(raw_shaped_data)
    assert pd.isna(result["BOROUGH"].iloc[3])


def test_derive_borough_preserves_an_existing_borough() -> None:
    """Re-deriving must not null out a borough a neighbourhood name can't map."""
    df = pd.DataFrame({"BOROUGH": ["manhattan"], "SUBLOCALITY": ["midtown east"]})
    assert derive_borough(df)["BOROUGH"].iloc[0] == "manhattan"


def test_derive_zipcode_extracts_from_the_state_field(
    raw_shaped_data: pd.DataFrame,
) -> None:
    result = derive_zipcode(raw_shaped_data)
    assert list(result["ZIPCODE"]) == ["11217", "10312", "10473", "11214"]


def test_derive_zipcode_leaves_unparseable_rows_null_not_sentinel() -> None:
    """A "00000" sentinel would become a real target-encoded ZIP category."""
    df = pd.DataFrame({"ZIPCODE": ["10022.0", "no digits here"]})
    result = derive_zipcode(df)
    assert result["ZIPCODE"].iloc[0] == "10022"
    assert pd.isna(result["ZIPCODE"].iloc[1])


def test_clean_pipeline_drops_the_overflow_sentinel_rather_than_capping_it(
    raw_shaped_data: pd.DataFrame,
) -> None:
    """2**31-1 must leave the dataset, not survive as a capped listing.

    cap_outliers clips to the IQR bound, so a sentinel that reaches it becomes
    an ordinary-looking listing at the cap instead of being removed.
    """
    df = raw_shaped_data.copy()
    df.loc[0, "PRICE"] = 2_147_483_647
    result = clean_pipeline(df)
    assert 2_147_483_647 not in set(result["PRICE"])
    assert len(result) == 2  # sentinel row and shifted-column row both gone


def test_clean_pipeline_derives_both_model_inputs_from_raw_columns(
    raw_shaped_data: pd.DataFrame,
) -> None:
    """The regression guard: raw input in, both model inputs out.

    BOROUGH is one-hot encoded and ZIPCODE target-encoded, so training cannot
    run without them. Before the derivation existed this returned a frame with
    neither column and nothing failed until predict time.
    """
    result = clean_pipeline(raw_shaped_data)
    assert "BOROUGH" in result.columns
    assert "ZIPCODE" in result.columns
    assert result["BOROUGH"].notna().all()
    assert result["ZIPCODE"].notna().all()
    assert len(result) == 3  # the shifted-column row is dropped


def test_deduplicate_removes_exact_dupes(sample_raw_data: pd.DataFrame) -> None:
    df = pd.concat([sample_raw_data, sample_raw_data.iloc[:2]], ignore_index=True)
    result = deduplicate(df)
    # Exact count, not `<= len(df)`: that holds for any dedup implementation
    # including one that removes nothing.
    assert len(result) == len(sample_raw_data)


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


def test_fit_cap_bounds_defaults_to_the_factor_the_measurement_chose() -> None:
    """3.0 is a measured trade-off (scripts/measure_cap_factor.py); the shipped
    path (run_protocol) calls fit_cap_bounds without factor=, so the DEFAULT is
    what shapes the training distribution — and must be pinned exactly."""
    prices = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 10_000_000.0])
    q1, q3 = prices.quantile(0.25), prices.quantile(0.75)

    bounds = fit_cap_bounds(pd.DataFrame({"PRICE": prices}), columns=["PRICE"])

    assert bounds["PRICE"][1] == pytest.approx(q3 + 3.0 * (q3 - q1))
    assert bounds["PRICE"][0] == pytest.approx(q1 - 3.0 * (q3 - q1))


def test_normalize_type_strips_the_listing_suffix() -> None:
    """Raw TYPE values are "condo for sale"-shaped while the API sends bare
    "condo" — without the strip, training and serving one-hot different
    categories for the same property type."""
    df = pd.DataFrame({"TYPE": ["condo for sale", "house for rent", "co-op"]})
    result = normalize_type(df)
    assert list(result["TYPE"]) == ["condo", "house", "co-op"]


def test_normalize_text_lowercases() -> None:
    df = pd.DataFrame({"TYPE": ["CONDO", "  House  ", "Co-Op"]})
    result = normalize_text_columns(df)
    assert list(result["TYPE"]) == ["condo", "house", "co-op"]


def test_clean_pipeline_produces_valid_output(sample_raw_data: pd.DataFrame) -> None:
    result = clean_pipeline(sample_raw_data)
    assert len(result) > 0
    assert (result["PRICE"] > 0).all()
    assert result["PROPERTYSQFT"].isna().sum() == 0
