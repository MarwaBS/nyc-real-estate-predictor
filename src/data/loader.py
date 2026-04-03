"""Data loading — single entry point for all dataset I/O."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import CLEANED_DATASET, RAW_DATASET, GEOCODE_FILE

logger = logging.getLogger(__name__)

# Explicit dtypes to prevent silent type coercion
_RAW_DTYPES: dict[str, str] = {
    "PRICE": "float64",
    "BEDS": "int64",
    "BATH": "float64",
    "PROPERTYSQFT": "float64",
    "LATITUDE": "float64",
    "LONGITUDE": "float64",
}


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw NY-House-Dataset.csv with enforced dtypes."""
    path = path or RAW_DATASET
    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path)
    # Normalize column names to uppercase
    df.columns = df.columns.str.upper().str.strip()
    logger.info("Raw dataset: %d rows x %d cols", *df.shape)
    return df


def load_cleaned(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned dataset (output of the cleaning pipeline)."""
    path = path or CLEANED_DATASET
    logger.info("Loading cleaned dataset from %s", path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.upper().str.strip()
    # Ensure ZIPCODE is string
    if "ZIPCODE" in df.columns:
        df["ZIPCODE"] = df["ZIPCODE"].astype(str).str.extract(r"(\d{5})")[0].fillna("00000")
    logger.info("Cleaned dataset: %d rows x %d cols", *df.shape)
    return df


def load_geocode(path: Path | None = None) -> pd.DataFrame:
    """Load geocoding enrichment data."""
    path = path or GEOCODE_FILE
    logger.info("Loading geocode data from %s", path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.upper().str.strip()
    return df
