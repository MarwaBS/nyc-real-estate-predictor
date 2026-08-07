"""Data validation, schema enforcement and quality checks."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_cleaned_data(df: pd.DataFrame) -> list[str]:
    """Run quality checks on cleaned dataset. Returns list of issues (empty = OK)."""
    issues: list[str] = []

    # Required columns
    required = {
        "PRICE",
        "BEDS",
        "BATH",
        "PROPERTYSQFT",
        "LATITUDE",
        "LONGITUDE",
        "BOROUGH",
    }
    missing = required - set(df.columns)
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # No nulls in critical columns
    for col in ["PRICE", "BEDS", "BATH", "PROPERTYSQFT"]:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                issues.append(f"{col} has {null_count} null values")

    # Value range checks
    if "PRICE" in df.columns and (df["PRICE"] <= 0).any():
        issues.append(f"PRICE has {(df['PRICE'] <= 0).sum()} non-positive values")
    if "PROPERTYSQFT" in df.columns and (df["PROPERTYSQFT"] <= 0).any():
        issues.append(
            f"PROPERTYSQFT has {(df['PROPERTYSQFT'] <= 0).sum()} non-positive values"
        )
    if "LATITUDE" in df.columns:
        out_of_range = ~df["LATITUDE"].between(40.4, 40.95)
        if out_of_range.any():
            issues.append(f"LATITUDE has {out_of_range.sum()} values outside NYC range")
    if "LONGITUDE" in df.columns:
        out_of_range = ~df["LONGITUDE"].between(-74.3, -73.6)
        if out_of_range.any():
            issues.append(
                f"LONGITUDE has {out_of_range.sum()} values outside NYC range"
            )

    for issue in issues:
        logger.warning("Data validation: %s", issue)
    if not issues:
        logger.info("Data validation passed, all checks OK")

    return issues


def assert_no_leakage(feature_names: list[str] | np.ndarray) -> None:
    """Reject any feature derived from the price target.

    The rule is "the name mentions price", not a list of known spellings.
    The list version passed a bare ``PRICE`` column, the target itself.
    Only five derived spellings were enumerated, so the check
    certified as clean exactly the input it existed to reject. No legitimate
    feature in ``src.config`` mentions price, so anything that does is either
    the target or built from it.
    """
    present = sorted(f for f in (str(n) for n in feature_names) if "price" in f.lower())
    if present:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: target-derived features in training set: {present}"
        )
