"""Schema mapping from NYC.gov Rolling Sales 2024 to the model feature space.

Contract: ``benchmarks/SCHEMA_MAP.md`` (version pinned in
``SCHEMA_MAP_VERSIONS.json``). This module is the reference
implementation of that contract; the adversarial test suite in
``tests/benchmarks/test_schema_firewall.py`` is the executable
specification.

Invariants enforced by construction here (not only by tests):

* Every transform in ``_build_features`` is row-wise and stateless;
  no function consumes more than one row's values at a time.
* The feature frame is assembled from a column dict with fixed key
  order, so the output shape does not depend on the input column
  ordering.
* The target column is extracted into a separate Series and never
  read while ``X`` is being built.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BOROUGH_NAMES: dict[int, str] = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island",
}


@dataclass(frozen=True)
class MappingReport:
    """Per-run summary of a mapping application.

    Attributes:
        n_raw: number of rows in the input frame before filtering.
        n_dropped: number of rows removed by structural filters.
        n_scored: number of rows retained for downstream inference.
        drop_reasons: count per named drop reason (see SCHEMA_MAP.md §4).
            Must satisfy ``sum(drop_reasons.values()) == n_dropped``.
        kept_mask: boolean mask over the raw frame's index, True where
            the row survived all filters. ``kept_mask.sum() == n_scored``.
    """

    n_raw: int
    n_dropped: int
    n_scored: int
    drop_reasons: dict[str, int]
    kept_mask: pd.Series


# ─── 4.1 Input normalisation ────────────────────────────────────────

def _normalise_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and return a copy.

    No content transformation; column renaming is deliberately out of
    scope in v1 because SCHEMA_MAP.md specifies raw column names
    verbatim.
    """
    out = raw.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Row-wise numeric coercion tolerant to comma-thousands and whitespace.

    For already-numeric dtypes this is identity modulo NaN coercion;
    for object/string dtypes it strips commas and whitespace before
    converting. Values that fail conversion become NaN so the drop
    engine can reject them.
    """
    if series.dtype.kind in "iufb":
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


# ─── 4.2 Drop engine ────────────────────────────────────────────────

def _run_drop_engine(
    raw: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int]]:
    """Apply SCHEMA_MAP.md §4 row filters in a fixed priority order.

    Each dropped row is assigned exactly one reason — the first rule
    in priority order that rejects it. The priority order is itself
    a versioned design choice (it determines the per-reason counts,
    not the final kept set) and is fixed for v1.

    Returns:
        ``(kept_mask, drop_reasons)`` where ``kept_mask`` is a bool
        Series aligned to ``raw.index`` and ``drop_reasons`` holds
        only non-zero counts.
    """
    sale_price = _coerce_numeric(raw["SALE PRICE"])
    gross_sqft = _coerce_numeric(raw["GROSS SQUARE FEET"])
    year_built = _coerce_numeric(raw["YEAR BUILT"])
    bcc = raw["BUILDING CLASS CATEGORY"].astype(str).str.strip()

    kept = pd.Series(True, index=raw.index)
    reasons: dict[str, int] = {}

    def _apply(mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False) & kept
        count = int(mask.sum())
        if count > 0:
            reasons[reason] = count
        kept.loc[mask] = False

    _apply(sale_price <= 0, "sale_price_non_positive")
    _apply(
        (sale_price < 10_000) | (sale_price > 100_000_000),
        "sale_price_out_of_range",
    )
    _apply(gross_sqft == 0, "missing_gross_sqft")
    _apply(year_built == 0, "missing_year_built")
    _apply(~bcc.str.startswith("R"), "non_residential_class")

    return kept, reasons


# ─── 4.3 Feature mapping ────────────────────────────────────────────

def _build_features(kept_raw: pd.DataFrame) -> pd.DataFrame:
    """Assemble the model feature frame from kept rows.

    Every column is derived from a single raw column via a row-wise
    transform. No aggregation, no dataset-wide statistic, no target
    reference.

    Column order is fixed by the dict literal below; the output frame
    is therefore independent of the input column ordering.
    """
    property_sqft = _coerce_numeric(kept_raw["GROSS SQUARE FEET"]).astype("int64")
    land_sqft = _coerce_numeric(kept_raw["LAND SQUARE FEET"]).astype("int64")
    borough_id = _coerce_numeric(kept_raw["BOROUGH"]).astype("int64")
    borough_name = borough_id.map(BOROUGH_NAMES).astype("object")
    property_type = (
        kept_raw["BUILDING CLASS CATEGORY"].astype(str).str.strip().str[:2]
    ).astype("object")
    year_built = _coerce_numeric(kept_raw["YEAR BUILT"]).astype("int64")
    zip_code = _coerce_numeric(kept_raw["ZIP CODE"]).astype("int64")

    return pd.DataFrame(
        {
            "borough_name": borough_name,
            "land_sqft": land_sqft,
            "property_sqft": property_sqft,
            "property_type": property_type,
            "year_built": year_built,
            "zip_code": zip_code,
        },
        index=kept_raw.index,
    )


# ─── 4.4 Final output contract ──────────────────────────────────────

def apply_schema_map(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, MappingReport]:
    """Map NYC.gov Rolling Sales columns to the trained model's feature space.

    See ``benchmarks/SCHEMA_MAP.md`` for the transformation contract.

    Returns:
        ``(X, target, report)``:

        - ``X``: feature frame, containing only columns defined by
          the mapping contract and none from ``FORBIDDEN_COLUMNS``.
        - ``target``: ``log1p(SALE PRICE)`` as a pandas Series aligned
          to ``X``'s index.
        - ``report``: :class:`MappingReport` with raw/dropped/scored
          counts and per-reason breakdown.
    """
    normalised = _normalise_columns(raw)

    kept_mask, drop_reasons = _run_drop_engine(normalised)
    kept_raw = normalised.loc[kept_mask]

    X = _build_features(kept_raw)
    target = pd.Series(
        np.log1p(_coerce_numeric(kept_raw["SALE PRICE"]).to_numpy()),
        index=kept_raw.index,
        name="TARGET",
    )

    n_raw = len(normalised)
    n_scored = int(kept_mask.sum())
    report = MappingReport(
        n_raw=n_raw,
        n_dropped=n_raw - n_scored,
        n_scored=n_scored,
        drop_reasons=drop_reasons,
        kept_mask=kept_mask.copy(),
    )

    return X, target, report


__all__ = ["MappingReport", "apply_schema_map"]
