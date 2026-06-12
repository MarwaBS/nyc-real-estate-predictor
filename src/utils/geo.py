"""Geospatial utilities — haversine distances to fixed reference points.

Scope is deliberately exactly what the models consume. Earlier revisions
also carried H3 hex indexing, KMeans neighborhood clustering, and a
KDTree nearest-subway-station lookup; none of those ever reached a model
feature list (the ColumnTransformer dropped their outputs, station data
was never bundled, and the KMeans transform was dataset-stateful — unusable
for single-row inference). They were removed rather than left as
README-decorating dead compute; the EDA notebooks remain the record of
that exploration.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Return distance in km between two lat/lon points (Haversine formula)."""
    earth_radius = 6_371.0  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return earth_radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_vectorized(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: float,
    lon2: float,
) -> pd.Series:
    """Vectorized haversine — returns distances in km for a full column."""
    earth_radius = 6_371.0
    lat1_r = np.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_r) * math.cos(lat2_r) * np.sin(dlon / 2) ** 2
    )
    # numpy's ufunc return type widens to Any under default stubs; cast via
    # pd.Series to match the declared return type and silence no-any-return.
    return pd.Series(earth_radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def add_distance_features(
    df: pd.DataFrame,
    reference_points: dict[str, tuple[float, float]],
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
) -> pd.DataFrame:
    """Add haversine distance columns for each named reference point."""
    result = df.copy()
    for name, (ref_lat, ref_lon) in reference_points.items():
        col_name = f"DIST_{name.upper().replace(' ', '_')}"
        result[col_name] = haversine_vectorized(
            result[lat_col],
            result[lon_col],
            ref_lat,
            ref_lon,
        )
    return result
