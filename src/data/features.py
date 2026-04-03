"""Feature engineering — all derived features, encoding prep, target creation.

CRITICAL DESIGN RULE:
    PRICE_PER_SQFT must NEVER appear in any feature set.
    It is derived from the target variable (PRICE) and causes data leakage.
    The R2=0.997 in previous experiments was fake because of this.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    CENTRAL_PARK,
    MANHATTAN_CENTER,
    PRICE_ZONE_BINS,
    PRICE_ZONE_LABELS,
    SQFT_BINS,
    SQFT_LABELS,
)
from src.utils.geo import (
    add_distance_features,
    add_h3_index,
    add_neighborhood_clusters,
    nearest_station_distance,
)
from src.utils.validation import assert_no_leakage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference points for distance features
# ---------------------------------------------------------------------------
REFERENCE_POINTS: dict[str, tuple[float, float]] = {
    "MANHATTAN_CENTER": MANHATTAN_CENTER,
    "CENTRAL_PARK": CENTRAL_PARK,
}


def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numerical features (no target-derived features)."""
    result = df.copy()

    result["TOTAL_ROOMS"] = result["BEDS"] + result["BATH"]
    result["BED_BATH_RATIO"] = result["BEDS"] / result["BATH"].clip(lower=1)
    result["LOG_SQFT"] = np.log1p(result["PROPERTYSQFT"])
    result["ROOMS_PER_SQFT"] = result["TOTAL_ROOMS"] / result["PROPERTYSQFT"].clip(lower=1)

    logger.info("Added 4 numeric features: TOTAL_ROOMS, BED_BATH_RATIO, LOG_SQFT, ROOMS_PER_SQFT")
    return result


def add_geospatial_features(
    df: pd.DataFrame,
    subway_stations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add all geospatial features: distances, H3, clusters."""
    result = df.copy()

    # Haversine distances to key landmarks
    result = add_distance_features(result, REFERENCE_POINTS)
    logger.info("Added distance features: DIST_MANHATTAN_CENTER, DIST_CENTRAL_PARK")

    # Nearest subway station
    if subway_stations is not None and len(subway_stations) > 0:
        result["DIST_NEAREST_SUBWAY"] = nearest_station_distance(result, subway_stations)
        logger.info("Added DIST_NEAREST_SUBWAY from %d stations", len(subway_stations))
    else:
        # Fallback: use distance to Manhattan center as proxy
        result["DIST_NEAREST_SUBWAY"] = result["DIST_MANHATTAN_CENTER"]
        logger.warning("No subway data — using DIST_MANHATTAN_CENTER as proxy")

    # H3 hexagonal grid
    result = add_h3_index(result, resolution=7)
    logger.info("Added H3_RES7 hexagonal index")

    # KMeans neighborhood clusters
    result = add_neighborhood_clusters(result, n_clusters=15)
    logger.info("Added NEIGHBORHOOD_CLUSTER (15 clusters)")

    return result


def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create target columns for classification and regression."""
    result = df.copy()

    # Price zones (classification target)
    result["PRICE_ZONE"] = pd.cut(
        result["PRICE"],
        bins=PRICE_ZONE_BINS,
        labels=PRICE_ZONE_LABELS,
        include_lowest=True,
    )

    # Log price (regression target — stabilizes variance)
    result["LOG_PRICE"] = np.log1p(result["PRICE"])

    # SQFT category (secondary classification)
    result["SQFT_CATEGORY"] = pd.cut(
        result["PROPERTYSQFT"],
        bins=SQFT_BINS,
        labels=SQFT_LABELS,
        include_lowest=True,
    )

    logger.info("Added targets: PRICE_ZONE (4 classes), LOG_PRICE, SQFT_CATEGORY (3 classes)")
    return result


def cap_categorical_cardinality(
    df: pd.DataFrame,
    columns: list[str],
    max_categories: int = 50,
) -> pd.DataFrame:
    """Frequency-cap high-cardinality categoricals — keep top N, rest = 'other'."""
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        top = result[col].value_counts().nlargest(max_categories).index
        n_capped = (~result[col].isin(top)).sum()
        result[col] = result[col].where(result[col].isin(top), "other")
        if n_capped > 0:
            logger.info("Capped %s: %d values -> 'other' (top %d kept)", col, n_capped, max_categories)
    return result


def feature_pipeline(
    df: pd.DataFrame,
    subway_stations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    logger.info("Starting feature pipeline on %d rows", len(df))

    df = add_numeric_features(df)
    df = add_geospatial_features(df, subway_stations=subway_stations)
    df = add_target_variables(df)
    df = cap_categorical_cardinality(df, columns=["SUBLOCALITY", "TYPE", "ZIPCODE"])

    # SAFETY CHECK: assert no leaky features
    feature_cols = [
        c for c in df.columns
        if c not in {"PRICE", "LOG_PRICE", "PRICE_ZONE", "SQFT_CATEGORY"}
    ]
    assert_no_leakage(feature_cols)

    logger.info("Feature pipeline complete: %d rows x %d cols", *df.shape)
    return df
