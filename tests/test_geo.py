"""Tests for geospatial utilities."""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils.geo import (
    add_distance_features,
    add_h3_index,
    add_neighborhood_clusters,
    haversine,
    haversine_vectorized,
)


def test_haversine_same_point_is_zero() -> None:
    assert haversine(40.758, -73.985, 40.758, -73.985) == 0.0


def test_haversine_manhattan_to_brooklyn() -> None:
    # Midtown Manhattan to Downtown Brooklyn ~= 5-10 km
    dist = haversine(40.758, -73.985, 40.689, -73.984)
    assert 5 < dist < 15


def test_haversine_vectorized_matches_scalar() -> None:
    lats = pd.Series([40.758, 40.689])
    lons = pd.Series([-73.985, -73.984])
    distances = haversine_vectorized(lats, lons, 40.758, -73.985)
    assert abs(distances.iloc[0]) < 0.01  # Same point
    assert distances.iloc[1] > 5  # Different point


def test_add_distance_features_creates_columns(sample_raw_data: pd.DataFrame) -> None:
    ref = {"MANHATTAN_CENTER": (40.758, -73.985)}
    result = add_distance_features(sample_raw_data, ref)
    assert "DIST_MANHATTAN_CENTER" in result.columns
    assert (result["DIST_MANHATTAN_CENTER"] >= 0).all()


def test_add_neighborhood_clusters(sample_raw_data: pd.DataFrame) -> None:
    result = add_neighborhood_clusters(sample_raw_data, n_clusters=3)
    assert "NEIGHBORHOOD_CLUSTER" in result.columns
    assert set(result["NEIGHBORHOOD_CLUSTER"].unique()).issubset({0, 1, 2})


def test_add_h3_index_produces_valid_cells() -> None:
    """add_h3_index must map lat/lon to H3 v4 cells (regression: requirements
    pinned h3 v3 while geo.py uses the v4 latlng_to_cell API)."""
    df = pd.DataFrame(
        {"LATITUDE": [40.7580, 40.7128], "LONGITUDE": [-73.9855, -74.0060]}
    )
    result = add_h3_index(df, resolution=7)
    assert "H3_RES7" in result.columns
    cells = result["H3_RES7"].tolist()
    assert len(cells) == 2
    # Distinct points → distinct cells; each a non-empty hex string.
    assert cells[0] != cells[1]
    assert all(isinstance(c, str) and len(c) == 15 for c in cells)
