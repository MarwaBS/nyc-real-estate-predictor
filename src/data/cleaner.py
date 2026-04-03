"""Data cleaning pipeline — deduplicate, impute, normalize, validate."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import BOROUGH_MAP

logger = logging.getLogger(__name__)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates and near-duplicates by lat/lon + price."""
    before = len(df)
    df = df.drop_duplicates().copy()
    # Near-duplicates: round lat/lon to 4 decimals (~11m precision) + same price
    df["_lat_round"] = df["LATITUDE"].round(4)
    df["_lon_round"] = df["LONGITUDE"].round(4)
    df = df.drop_duplicates(subset=["_lat_round", "_lon_round", "PRICE"], keep="first")
    df = df.drop(columns=["_lat_round", "_lon_round"])
    logger.info("Deduplication: %d -> %d rows (-%d)", before, len(df), before - len(df))
    return df.reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values — borough-aware median for numerics."""
    result = df.copy()

    # BEDS/BATH: median per borough (smarter than global median)
    for col in ["BEDS", "BATH"]:
        if col in result.columns and result[col].isna().any():
            if "BOROUGH" in result.columns:
                medians = result.groupby("BOROUGH")[col].transform("median")
                result[col] = result[col].fillna(medians)
            # Fallback: global median for any remaining NaN
            result[col] = result[col].fillna(result[col].median())
            logger.info("Imputed %s: %d values filled", col, df[col].isna().sum())

    # PROPERTYSQFT: median (no borough split — less correlated)
    if "PROPERTYSQFT" in result.columns and result["PROPERTYSQFT"].isna().any():
        median_sqft = result["PROPERTYSQFT"].median()
        result["PROPERTYSQFT"] = result["PROPERTYSQFT"].fillna(median_sqft)

    return result


def cap_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 3.0,
) -> pd.DataFrame:
    """Cap outliers using IQR * factor method (cap, don't drop)."""
    result = df.copy()
    columns = columns or ["PRICE", "PROPERTYSQFT", "BEDS", "BATH"]

    for col in columns:
        if col not in result.columns:
            continue
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        capped = result[col].clip(lower=lower, upper=upper)
        n_capped = (result[col] != capped).sum()
        result[col] = capped
        if n_capped > 0:
            logger.info("Capped %d outliers in %s (range: %.0f - %.0f)", n_capped, col, lower, upper)

    return result


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip whitespace on all text columns."""
    result = df.copy()
    text_cols = result.select_dtypes(include=["object"]).columns
    for col in text_cols:
        result[col] = result[col].str.strip().str.lower()
    return result


def normalize_borough(df: pd.DataFrame, col: str = "BOROUGH") -> pd.DataFrame:
    """Map county names and variants to standard borough names."""
    result = df.copy()
    if col in result.columns:
        result[col] = result[col].str.lower().str.strip().map(BOROUGH_MAP).fillna(result[col])
    return result


def normalize_zipcode(df: pd.DataFrame, col: str = "ZIPCODE") -> pd.DataFrame:
    """Extract 5-digit ZIP and zero-pad."""
    result = df.copy()
    if col in result.columns:
        result[col] = (
            result[col]
            .astype(str)
            .str.extract(r"(\d{5})")[0]
            .fillna("00000")
            .str.zfill(5)
        )
    return result


def normalize_type(df: pd.DataFrame, col: str = "TYPE") -> pd.DataFrame:
    """Simplify property type labels."""
    result = df.copy()
    if col in result.columns:
        # Remove trailing " for sale" etc.
        result[col] = (
            result[col]
            .str.replace(r"\s+for\s+sale$", "", regex=True)
            .str.replace(r"\s+for\s+rent$", "", regex=True)
            .str.strip()
        )
    return result


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline end-to-end."""
    logger.info("Starting cleaning pipeline on %d rows", len(df))

    df = deduplicate(df)
    df = normalize_text_columns(df)
    df = normalize_borough(df)
    df = normalize_zipcode(df)
    df = normalize_type(df)
    df = impute_missing(df)
    df = cap_outliers(df)

    # Drop rows with non-positive price or sqft (invalid data)
    before = len(df)
    df = df[df["PRICE"] > 0]
    df = df[df["PROPERTYSQFT"] > 0]
    logger.info("Dropped %d rows with non-positive PRICE/SQFT", before - len(df))

    # Validate lat/lon in NYC range
    df = df[df["LATITUDE"].between(40.4, 40.95)]
    df = df[df["LONGITUDE"].between(-74.3, -73.6)]

    df = df.reset_index(drop=True)
    logger.info("Cleaning pipeline complete: %d rows x %d cols", *df.shape)
    return df
