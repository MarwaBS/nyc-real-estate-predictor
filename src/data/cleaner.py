"""Data cleaning pipeline — deduplicate, impute, normalize, validate."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import BOROUGH_MAP

logger = logging.getLogger(__name__)

# 2**31 - 1. A PRICE at exactly this value is an integer-overflow sentinel from
# the upstream export, not a listing price.
INT32_MAX = 2_147_483_647


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


# An existing BOROUGH is consulted first so that re-running on already-derived
# output preserves it (and normalises a county-name spelling through the same
# map). The raw Kaggle export has no such column, so it contributes nothing on
# a first pass.
#
# The rest are ordered by measured hit rate on the 4,801-row raw snapshot:
# SUBLOCALITY resolves 78.5% on its own, ADMINISTRATIVE_AREA_LEVEL_2 0.8%,
# LOCALITY 47.0%; chained in this order they reach 99.2% together. SUBLOCALITY
# leads because it is both the highest-coverage and the most specific field --
# LOCALITY is frequently the literal string "United States".
_BOROUGH_SOURCE_COLUMNS = (
    "BOROUGH",
    "SUBLOCALITY",
    "ADMINISTRATIVE_AREA_LEVEL_2",
    "LOCALITY",
)


def derive_borough(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a canonical BOROUGH column from the raw geocode fields.

    The Kaggle export ships no BOROUGH column at all. Google's geocoder put the
    borough in a different field depending on how each listing resolved, so a
    single source column cannot recover it -- hence the fallback chain above.

    Rows that no source resolves (37, 0.77%) are left null on purpose: their
    geocode columns are shifted, with LOCALITY reading "United States" and
    ADMINISTRATIVE_AREA_LEVEL_2 holding a ZIP code. Their fields are provably in
    the wrong columns, so ``clean_pipeline`` drops them rather than guessing a
    borough from lat/lon, which would need a polygon lookup to rescue under 1%
    of rows.

    Re-running this on already-derived output preserves the existing BOROUGH:
    it heads the source chain and BOROUGH_MAP maps each canonical borough name
    to itself. Without that entry the chain would overwrite a valid borough with
    a null whenever SUBLOCALITY held a neighbourhood name ("midtown east")
    rather than a county, and the dropna below would then discard the row.
    """
    result = df.copy()
    borough = pd.Series(pd.NA, index=result.index, dtype="object")

    for col in _BOROUGH_SOURCE_COLUMNS:
        if col not in result.columns:
            continue
        borough = borough.fillna(
            result[col].astype(str).str.lower().str.strip().map(BOROUGH_MAP)
        )

    result["BOROUGH"] = borough
    logger.info("Derived BOROUGH for %d/%d rows", borough.notna().sum(), len(result))
    return result


def derive_zipcode(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ZIPCODE, preferring an existing column over the raw STATE field.

    The raw STATE field is not a state: it holds "Brooklyn, NY 11238"-style
    strings, from which a 5-digit ZIP extracts for 100% of the raw snapshot.

    A non-matching row is left null rather than filled with a "00000" sentinel.
    The sentinel was worse than useless here: ZIPCODE is a target-encoded
    feature, so every unparseable row would silently share one encoded category
    and the model would learn a price for a ZIP that does not exist.
    """
    result = df.copy()
    source = "ZIPCODE" if "ZIPCODE" in result.columns else "STATE"
    if source not in result.columns:
        return result

    result["ZIPCODE"] = result[source].astype(str).str.extract(r"(\d{5})")[0]
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
    # Both derivations run before impute_missing, which fills BEDS/BATH with a
    # per-borough median and silently falls back to the global median when
    # BOROUGH is absent.
    df = derive_borough(df)
    df = derive_zipcode(df)
    df = normalize_type(df)

    # Drop rows whose borough or ZIP could not be derived. Both are model
    # inputs -- BOROUGH is one-hot encoded and ZIPCODE target-encoded -- so a
    # null here becomes a phantom category rather than a missing value.
    before = len(df)
    df = df.dropna(subset=["BOROUGH", "ZIPCODE"])
    logger.info("Dropped %d rows with underivable BOROUGH/ZIPCODE", before - len(df))

    df = impute_missing(df)

    # Drop 32-bit integer overflow sentinels before capping. The raw snapshot
    # holds exactly one PRICE of 2,147,483,647 (2**31 - 1) where the next
    # highest real listing is 195,000,000 -- it is a serialisation artefact,
    # not a price. Capping would silently convert it into a plausible-looking
    # listing at the IQR bound rather than removing it, so it has to go first.
    before = len(df)
    df = df[df["PRICE"] < INT32_MAX]
    logger.info("Dropped %d rows with overflow-sentinel PRICE", before - len(df))

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
