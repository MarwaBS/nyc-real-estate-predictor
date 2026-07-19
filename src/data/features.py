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
from src.utils.geo import add_distance_features
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
    listings = df.copy()

    # Sums and ratios, not transforms. A gradient-boosted tree splits on
    # thresholds, so it is invariant to any monotone transform of a single
    # feature -- LOG_SQFT gave it exactly the partitions PROPERTYSQFT already
    # gives and was removed. A ratio is different: the tree can only approximate
    # BEDS/BATH through many awkward splits, so computing it is real signal.
    listings["TOTAL_ROOMS"] = listings["BEDS"] + listings["BATH"]
    listings["BED_BATH_RATIO"] = listings["BEDS"] / listings["BATH"].clip(lower=1)
    listings["ROOMS_PER_SQFT"] = listings["TOTAL_ROOMS"] / listings[
        "PROPERTYSQFT"
    ].clip(lower=1)

    logger.info("Added 3 numeric features: TOTAL_ROOMS, BED_BATH_RATIO, ROOMS_PER_SQFT")
    return listings


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Haversine distances to the two landmarks the models consume.

    DIST_NEAREST_SUBWAY used to be assigned here as a copy of
    DIST_MANHATTAN_CENTER and called a proxy. A duplicate column is not a
    proxy -- it carries zero information by construction and cost a feature
    slot while implying the model knew something about transit access. It is
    gone; station-level data is still not bundled.

    H3 indexing and KMeans clustering were explored in EDA and never fed any
    model, so they are not computed here either.
    """
    listings = add_distance_features(df.copy(), REFERENCE_POINTS)
    logger.info("Added distance features: DIST_MANHATTAN_CENTER, DIST_CENTRAL_PARK")
    return listings


def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create target columns for classification and regression."""
    listings = df.copy()

    # Price zones (classification target)
    listings["PRICE_ZONE"] = pd.cut(
        listings["PRICE"],
        bins=PRICE_ZONE_BINS,
        labels=PRICE_ZONE_LABELS,
        include_lowest=True,
    )

    # Log price (regression target — stabilizes variance)
    listings["LOG_PRICE"] = np.log1p(listings["PRICE"])

    # SQFT category (secondary classification)
    listings["SQFT_CATEGORY"] = pd.cut(
        listings["PROPERTYSQFT"],
        bins=SQFT_BINS,
        labels=SQFT_LABELS,
        include_lowest=True,
    )

    logger.info(
        "Added targets: PRICE_ZONE (4 classes), LOG_PRICE, SQFT_CATEGORY (3 classes)"
    )
    return listings


def cap_categorical_cardinality(
    df: pd.DataFrame,
    columns: list[str],
    max_categories: int = 50,
) -> pd.DataFrame:
    """Frequency-cap high-cardinality categoricals — keep top N, rest = 'other'."""
    listings = df.copy()
    for col in columns:
        if col not in listings.columns:
            continue
        top = listings[col].value_counts().nlargest(max_categories).index
        n_capped = (~listings[col].isin(top)).sum()
        listings[col] = listings[col].where(listings[col].isin(top), "other")
        if n_capped > 0:
            logger.info(
                "Capped %s: %d values -> 'other' (top %d kept)",
                col,
                n_capped,
                max_categories,
            )
    return listings


def learned_capped_categories(pipeline: object) -> dict[str, set]:
    """Categories a *fitted* pipeline learned, per column — but only for columns
    that learned an explicit ``"other"`` bucket (i.e. were frequency-capped at
    train time by :func:`cap_categorical_cardinality`).

    Serving uses this to map any value outside the learned set to ``"other"`` so a
    rare/unseen category gets the trained ``"other"`` encoding instead of the
    encoder's unseen default (TargetEncoder → global mean, OneHot → all-zeros).
    Without it, training caps SUBLOCALITY/ZIPCODE rare values to ``"other"`` but
    serving passes them raw, so the model sees a different encoding than it was
    trained on — a silent train/serve skew. Derived from the *shipped* artifact so
    it can never drift from the model. Columns the model did not cap (no
    ``"other"`` learned, e.g. low-cardinality BOROUGH/TYPE) are omitted.
    """
    pre = getattr(pipeline, "named_steps", {}).get("preprocessor")
    known: dict[str, set] = {}
    if pre is None:
        return known
    for _name, trans, cols in getattr(pre, "transformers_", []):
        # OneHotEncoder exposes per-column categories directly.
        cats_attr = getattr(trans, "categories_", None)
        if cats_attr is not None:
            # strict=True: cols and categories_ are the same encoder's columns
            # and their fitted categories. A length mismatch means the encoder
            # is not the one these columns came from, and silently zipping to
            # the shorter of the two would drop a column's capped categories
            # and let an unseen value through the serving cap unmapped.
            for col, cats in zip(cols, cats_attr, strict=True):
                if "other" in {str(c) for c in cats}:
                    known[col] = set(cats)
            continue
        # category_encoders' TargetEncoder keeps its categories on an inner
        # OrdinalEncoder mapping (one frame per column, indexed by category).
        ordinal = getattr(trans, "ordinal_encoder", None)
        if ordinal is not None:
            for col_map in getattr(ordinal, "mapping", []):
                col = col_map["col"]
                cats = [c for c in col_map["mapping"].index.tolist() if pd.notna(c)]
                if "other" in {str(c) for c in cats}:
                    known[col] = set(cats)
    return known


def apply_serving_cap(df: pd.DataFrame, known: dict[str, set]) -> pd.DataFrame:
    """Map any value outside its learned category set to ``"other"`` for each
    capped column — the inference-time mirror of :func:`cap_categorical_cardinality`,
    keyed off the categories the fitted model actually learned (see
    :func:`learned_capped_categories`)."""
    listings = df.copy()
    for col, allowed in known.items():
        if col in listings.columns:
            listings[col] = listings[col].where(listings[col].isin(allowed), "other")
    return listings


def feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    logger.info("Starting feature pipeline on %d rows", len(df))

    df = add_numeric_features(df)
    df = add_geospatial_features(df)
    df = add_target_variables(df)
    df = cap_categorical_cardinality(df, columns=["SUBLOCALITY", "TYPE", "ZIPCODE"])

    # SAFETY CHECK: assert no leaky features
    feature_cols = [
        c
        for c in df.columns
        if c not in {"PRICE", "LOG_PRICE", "PRICE_ZONE", "SQFT_CATEGORY"}
    ]
    assert_no_leakage(feature_cols)

    logger.info("Feature pipeline complete: %d rows x %d cols", *df.shape)
    return df
