"""Geospatial utilities — haversine, H3, nearest-neighbor lookups."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def haversine(
    lat1: float, lon1: float, lat2: float, lon2: float,
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
            result[lat_col], result[lon_col], ref_lat, ref_lon,
        )
    return result


def nearest_station_distance(
    df: pd.DataFrame,
    stations: pd.DataFrame,
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
    station_lat: str = "latitude",
    station_lon: str = "longitude",
) -> pd.Series:
    """Compute distance to nearest subway station using scipy KDTree."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        # Fallback: return NaN if scipy not installed
        return pd.Series(np.nan, index=df.index)

    station_coords = np.radians(stations[[station_lat, station_lon]].values)
    tree = cKDTree(station_coords)

    property_coords = np.radians(df[[lat_col, lon_col]].values)
    distances, _ = tree.query(property_coords, k=1)

    # Convert radians to km (approximate using Earth's radius)
    return pd.Series(distances * 6_371.0, index=df.index, name="DIST_NEAREST_SUBWAY")


def add_h3_index(
    df: pd.DataFrame,
    resolution: int = 7,
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
) -> pd.DataFrame:
    """Add H3 hexagonal grid index column."""
    result = df.copy()
    col_name = f"H3_RES{resolution}"
    try:
        import h3

        result[col_name] = df.apply(
            lambda row: h3.latlng_to_cell(row[lat_col], row[lon_col], resolution),
            axis=1,
        )
    except ImportError:
        result[col_name] = "h3_unavailable"
    return result


def add_neighborhood_clusters(
    df: pd.DataFrame,
    n_clusters: int = 15,
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
    random_state: int = 42,
) -> pd.DataFrame:
    """Add KMeans cluster labels based on geographic coordinates."""
    from sklearn.cluster import KMeans

    result = df.copy()
    coords = result[[lat_col, lon_col]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    result["NEIGHBORHOOD_CLUSTER"] = kmeans.fit_predict(coords)
    return result
