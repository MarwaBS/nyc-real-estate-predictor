"""Tests for geospatial utilities."""

from __future__ import annotations

import math

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
    """Element-for-element agreement — training uses the vectorized form, the
    API the scalar one, so any divergence is train/serve skew."""
    lats = pd.Series([40.758, 40.689, 40.9, 40.5])
    lons = pd.Series([-73.985, -73.984, -73.7, -74.2])
    distances = haversine_vectorized(lats, lons, 40.758, -73.985)
    for i in range(len(lats)):
        scalar = haversine(lats.iloc[i], lons.iloc[i], 40.758, -73.985)
        assert distances.iloc[i] == pytest.approx(scalar, abs=1e-9)


def test_haversine_uses_the_earth_radius() -> None:
    """One degree of latitude is 111.195 km at R=6371; a shifted radius
    (e.g. 6400) yields 111.70 and fails both implementations."""
    expected = 6371.0 * math.pi / 180.0
    assert haversine(40.0, -74.0, 41.0, -74.0) == pytest.approx(expected, abs=0.01)
    vec = haversine_vectorized(pd.Series([41.0]), pd.Series([-74.0]), 40.0, -74.0)
    assert vec.iloc[0] == pytest.approx(expected, abs=0.01)


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
