"""Tests for geospatial utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.geo import (
    add_distance_features,
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


def test_geo_module_carries_no_dead_features() -> None:
    """Regression guard: H3 indexing, KMeans clustering, and the KDTree
    subway lookup were removed because no model ever consumed their
    output (the ColumnTransformer dropped the columns and station data
    was never bundled). If someone reintroduces them, they must wire
    them into a model feature list — not just into this module."""
    import src.utils.geo as geo

    for dead in (
        "add_h3_index",
        "add_neighborhood_clusters",
        "nearest_station_distance",
    ):
        assert not hasattr(geo, dead), (
            f"{dead} reappeared in src.utils.geo — either wire it into the "
            f"model feature contract or keep it out of the production path"
        )
