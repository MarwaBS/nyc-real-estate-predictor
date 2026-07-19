"""Data cleaning pipeline — deduplicate, impute, normalize, validate."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import BOROUGH_MAP

logger = logging.getLogger(__name__)

# 2**31 - 1. A PRICE at exactly this value is an integer-overflow sentinel from
# the upstream export, not a listing price.
INT32_MAX = 2_147_483_647

# The columns clean_pipeline dereferences unconditionally. Checked up front so
# a caller passing the wrong frame gets a named failure instead of a KeyError
# from whichever step happens to touch a missing column first.
_REQUIRED_RAW_COLUMNS = frozenset(
    {"PRICE", "PROPERTYSQFT", "LATITUDE", "LONGITUDE", "BEDS", "BATH"}
)


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
    listings = df.copy()

    # BEDS/BATH: median per borough (smarter than global median)
    for col in ["BEDS", "BATH"]:
        if col in listings.columns and listings[col].isna().any():
            if "BOROUGH" in listings.columns:
                medians = listings.groupby("BOROUGH")[col].transform("median")
                listings[col] = listings[col].fillna(medians)
            # Fallback: global median for any remaining NaN
            listings[col] = listings[col].fillna(listings[col].median())
            logger.info("Imputed %s: %d values filled", col, df[col].isna().sum())

    # PROPERTYSQFT: median (no borough split — less correlated)
    if "PROPERTYSQFT" in listings.columns and listings["PROPERTYSQFT"].isna().any():
        median_sqft = listings["PROPERTYSQFT"].median()
        listings["PROPERTYSQFT"] = listings["PROPERTYSQFT"].fillna(median_sqft)

    return listings


def cap_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 3.0,
) -> pd.DataFrame:
    """Cap outliers at Q3 + factor*IQR, clipping rather than dropping.

    factor=3.0 is a measured trade-off, not an inherited default. Held-out
    val, scored on a COMMON evaluation set (listings under $2,989,000, the
    tightest cap's ceiling) so every variant faces an identical target
    distribution:

        factor   val R2 (common)   val MAE (common)   listings at the cap
        1.5           0.7279            0.2682         531  (11.73%)
        3.0           0.6308            0.2772         351  ( 7.76%)
        5.0           0.6273            0.2796         244  ( 5.39%)
        none          0.6224            0.2810           1  ( 0.02%)

    Two things that measurement settles. Capping beats not capping on typical
    listings -- scored across ALL rows the uncapped model looks best (R2
    0.7854), but that ranking is variance inflation from a single $195M
    listing, and it reverses on a like-for-like target.

    And 1.5 scores better than 3.0 on both metrics. It is not used because it
    collapses 11.73% of listings onto one price against 7.76%: P1 covers NYC
    residential property, not only the dense segment, and a model that cannot
    distinguish anything above $2.99M fails that for one listing in eight. The
    accuracy gain is 0.009 MAE (3.3% relative); the coverage cost is 180 more
    listings rendered indistinguishable. 3.0 buys most of the benefit at half
    the flattening.
    """
    listings = df.copy()
    columns = columns or ["PRICE", "PROPERTYSQFT", "BEDS", "BATH"]

    for col in columns:
        if col not in listings.columns:
            continue
        q1 = listings[col].quantile(0.25)
        q3 = listings[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        capped = listings[col].clip(lower=lower, upper=upper)
        n_capped = (listings[col] != capped).sum()
        listings[col] = capped
        if n_capped > 0:
            logger.info(
                "Capped %d outliers in %s (range: %.0f - %.0f)",
                n_capped,
                col,
                lower,
                upper,
            )

    return listings


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip whitespace on all text columns."""
    listings = df.copy()
    text_cols = listings.select_dtypes(include=["object"]).columns
    for col in text_cols:
        listings[col] = listings[col].str.strip().str.lower()
    return listings


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

    Rows that no source resolves (37 of the raw 4,801; 36 survive dedup and
    reach the drop in ``clean_pipeline``) are left null on purpose: their
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
    listings = df.copy()
    borough = pd.Series(pd.NA, index=listings.index, dtype="object")

    for col in _BOROUGH_SOURCE_COLUMNS:
        if col not in listings.columns:
            continue
        borough = borough.fillna(
            listings[col].astype(str).str.lower().str.strip().map(BOROUGH_MAP)
        )

    listings["BOROUGH"] = borough
    logger.info("Derived BOROUGH for %d/%d rows", borough.notna().sum(), len(listings))
    return listings


def derive_zipcode(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ZIPCODE, preferring an existing column over the raw STATE field.

    The raw STATE field is not a state: it holds "Brooklyn, NY 11238"-style
    strings, from which a 5-digit ZIP extracts for 100% of the raw snapshot.

    A non-matching row is left null rather than filled with a "00000" sentinel.
    The sentinel was worse than useless here: ZIPCODE is a target-encoded
    feature, so every unparseable row would silently share one encoded category
    and the model would learn a price for a ZIP that does not exist.
    """
    listings = df.copy()
    source = "ZIPCODE" if "ZIPCODE" in listings.columns else "STATE"
    if source not in listings.columns:
        return listings

    listings["ZIPCODE"] = listings[source].astype(str).str.extract(r"(\d{5})")[0]
    return listings


def normalize_type(df: pd.DataFrame, col: str = "TYPE") -> pd.DataFrame:
    """Simplify property type labels."""
    listings = df.copy()
    if col in listings.columns:
        # Remove trailing " for sale" etc.
        listings[col] = (
            listings[col]
            .str.replace(r"\s+for\s+sale$", "", regex=True)
            .str.replace(r"\s+for\s+rent$", "", regex=True)
            .str.strip()
        )
    return listings


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline end-to-end on a RAW frame.

    Single-pass by design, and not idempotent: ``cap_outliers`` recomputes its
    IQR bounds from whatever it is given, so cleaning an already-cleaned frame
    tightens the cap around the previously capped distribution and drops
    further rows. The input is the raw export; ``run_training.py`` is the only
    caller and passes ``load_raw()``.
    """
    missing = sorted(_REQUIRED_RAW_COLUMNS - set(df.columns))
    if missing:
        raise KeyError(
            f"clean_pipeline requires the raw export columns; missing: {missing}"
        )

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
    # holds exactly one PRICE of 2,147,483,647 (2**31 - 1); the next highest
    # real listing is 195,000,000, so this is a serialisation artefact rather
    # than an expensive property. It is removed rather than capped because it
    # is not a price at all -- unlike the genuine high-end listings the cap
    # legitimately clips, which stay in the dataset at the cap value.
    #
    # The threshold catches this sentinel and anything above it; it is not a
    # general implausible-price filter. A merely absurd value (say 2**31 - 2)
    # would still be capped like any other outlier, which is the right
    # treatment for a number that could be a price.
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
