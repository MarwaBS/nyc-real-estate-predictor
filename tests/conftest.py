"""Shared fixtures for the test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Minimal raw dataset for testing."""
    return pd.DataFrame(
        {
            "PRICE": [500_000, 750_000, 1_200_000, 300_000, 2_500_000],
            "BEDS": [2, 3, 4, 1, 5],
            "BATH": [1.0, 2.0, 3.0, 1.0, 4.0],
            "PROPERTYSQFT": [800.0, 1_400.0, 2_200.0, 500.0, 3_500.0],
            "LATITUDE": [40.758, 40.689, 40.650, 40.820, 40.541],
            "LONGITUDE": [-73.985, -73.944, -73.949, -73.874, -74.196],
            "BOROUGH": [
                "manhattan",
                "brooklyn",
                "brooklyn",
                "the bronx",
                "staten island",
            ],
            "TYPE": ["condo", "house", "condo", "co-op", "house"],
            "SUBLOCALITY": [
                "midtown east",
                "fort greene",
                "park slope",
                "pelham bay",
                "tottenville",
            ],
            "ZIPCODE": ["10022", "11217", "11215", "10473", "10312"],
            "BROKERTITLE": ["broker a", "broker b", "broker c", "broker d", "broker e"],
            "ADDRESS": [
                "123 Main St",
                "456 Oak Ave",
                "789 Elm Rd",
                "101 Pine Ln",
                "202 Maple Dr",
            ],
        }
    )
