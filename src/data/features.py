"""Feature engineering, all derived features, encoding prep, target creation.

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
)
from src.utils.geo import add_distance_features

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

    # Ratios, not transforms: a tree splits on thresholds, so it is invariant
    # to any monotone transform of a single feature, but can only approximate
    # a ratio like BEDS/BATH through many awkward splits.
    listings["TOTAL_ROOMS"] = listings["BEDS"] + listings["BATH"]
    listings["BED_BATH_RATIO"] = listings["BEDS"] / listings["BATH"].clip(lower=1)
    listings["ROOMS_PER_SQFT"] = listings["TOTAL_ROOMS"] / listings[
        "PROPERTYSQFT"
    ].clip(lower=1)

    logger.info("Added 3 numeric features: TOTAL_ROOMS, BED_BATH_RATIO, ROOMS_PER_SQFT")
    return listings


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Haversine distances to the two landmarks the model consumes."""
    listings = add_distance_features(df.copy(), REFERENCE_POINTS)
    logger.info("Added distance features: DIST_MANHATTAN_CENTER, DIST_CENTRAL_PARK")
    return listings


def add_target_variables(
    df: pd.DataFrame, bins: list[float] | None = None
) -> pd.DataFrame:
    """Create target columns for classification and regression.

    ``bins`` defaults to the shipped PRICE_ZONE_BINS; training passes the
    cut-points it derived from its own train split, so labels can never lean
    on quantiles of the held-out rows.
    """
    listings = df.copy()

    listings["PRICE_ZONE"] = pd.cut(
        listings["PRICE"],
        bins=bins if bins is not None else PRICE_ZONE_BINS,
        labels=PRICE_ZONE_LABELS,
        include_lowest=True,
    )

    # Log price (regression target, stabilizes variance)
    listings["LOG_PRICE"] = np.log1p(listings["PRICE"])

    logger.info("Added targets: PRICE_ZONE (4 classes), LOG_PRICE")
    return listings


def fit_top_categories(
    df: pd.DataFrame,
    columns: list[str],
    max_categories: int = 50,
) -> dict[str, set]:
    """The top-N category vocabulary per column, fitted on the given rows.

    Fit/apply are split so the vocabulary can be counted on the TRAIN split
    only, pooled counts let val/test frequencies decide which categories the
    model learns.
    """
    return {
        col: set(df[col].value_counts().nlargest(max_categories).index)
        for col in columns
        if col in df.columns
    }


def apply_top_categories(df: pd.DataFrame, top: dict[str, set]) -> pd.DataFrame:
    """Map any value outside its fitted vocabulary to 'other'."""
    listings = df.copy()
    for col, keep in top.items():
        if col not in listings.columns:
            continue
        n_capped = (~listings[col].isin(keep)).sum()
        listings[col] = listings[col].where(listings[col].isin(keep), "other")
        if n_capped > 0:
            logger.info("Capped %s: %d values -> 'other'", col, n_capped)
    return listings


def learned_capped_categories(pipeline: object) -> dict[str, set]:
    """Categories a *fitted* pipeline learned, per column, but only for columns
    that learned an explicit ``"other"`` bucket (i.e. were frequency-capped at
    train time by :func:`fit_top_categories`).

    Serving uses this to map any value outside the learned set to ``"other"`` so a
    rare/unseen category gets the trained ``"other"`` encoding instead of the
    encoder's unseen default (TargetEncoder → global mean, OneHot → all-zeros).
    Without it, training caps SUBLOCALITY/ZIPCODE rare values to ``"other"`` but
    serving passes them raw, so the model sees a different encoding than it was
    trained on, a silent train/serve skew. Derived from the *shipped* artifact so
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
    capped column, the inference-time mirror of :func:`apply_top_categories`,
    keyed off the categories the fitted model actually learned (see
    :func:`learned_capped_categories`)."""
    listings = df.copy()
    for col, allowed in known.items():
        if col in listings.columns:
            listings[col] = listings[col].where(listings[col].isin(allowed), "other")
    return listings
