"""Fixtures for the external-benchmark firewall suite.

Provides a synthetic NYC.gov Rolling Sales sample frame that matches
the column names the real 2024 dataset uses, without depending on an
external download. Every valid row respects ``SCHEMA_MAP.md v1`` §2/§4
(building class prefix ``R*`` only); every drop reason listed in §4
is exercised by at least one row.

The fixture is deliberately narrow — v1 of the contract scopes the
benchmark to NYC condominiums (building class starting with R). If
the real NYC.gov download in Step 5 reveals the scope should be
broader, that triggers a ``SCHEMA_MAP_VERSION`` bump to v2 per Rule
D2 in ``INTERNAL/EXECUTION_CONTRACT.md``. For Step 3 we honour v1
exactly so the adversarial suite exercises the real contract.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def nyc_rolling_sales_fixture() -> pd.DataFrame:
    """Synthetic sample mirroring the NYC.gov Rolling Sales 2024 schema.

    Not the real dataset — the real download happens in Step 5. This
    fixture exists only to drive the adversarial test suite with
    deterministic, schema-accurate rows covering every drop reason
    under SCHEMA_MAP.md v1 and every borough.
    """
    rows = [
        # ── Valid residential rows (R* building class) — all five boroughs ──
        {"BOROUGH": 1, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1200, "LAND SQUARE FEET": 2000,
         "YEAR BUILT": 1925, "ZIP CODE": 10001, "SALE PRICE": 1_500_000},
        {"BOROUGH": 1, "BUILDING CLASS CATEGORY": "R6 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1800, "LAND SQUARE FEET": 2400,
         "YEAR BUILT": 1940, "ZIP CODE": 10002, "SALE PRICE": 2_200_000},
        {"BOROUGH": 1, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 750, "LAND SQUARE FEET": 1000,
         "YEAR BUILT": 2005, "ZIP CODE": 10011, "SALE PRICE": 1_850_000},
        {"BOROUGH": 2, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1400, "LAND SQUARE FEET": 3000,
         "YEAR BUILT": 1965, "ZIP CODE": 10451, "SALE PRICE": 650_000},
        {"BOROUGH": 2, "BUILDING CLASS CATEGORY": "R5 CONDOMINIUMS",
         "GROSS SQUARE FEET": 900, "LAND SQUARE FEET": 1200,
         "YEAR BUILT": 1980, "ZIP CODE": 10452, "SALE PRICE": 420_000},
        {"BOROUGH": 2, "BUILDING CLASS CATEGORY": "R7 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1900, "LAND SQUARE FEET": 2300,
         "YEAR BUILT": 1948, "ZIP CODE": 10462, "SALE PRICE": 720_000},
        {"BOROUGH": 3, "BUILDING CLASS CATEGORY": "R6 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1600, "LAND SQUARE FEET": 2200,
         "YEAR BUILT": 1955, "ZIP CODE": 11201, "SALE PRICE": 1_100_000},
        {"BOROUGH": 3, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1100, "LAND SQUARE FEET": 1500,
         "YEAR BUILT": 1995, "ZIP CODE": 11215, "SALE PRICE": 875_000},
        {"BOROUGH": 3, "BUILDING CLASS CATEGORY": "R6 CONDOMINIUMS",
         "GROSS SQUARE FEET": 2200, "LAND SQUARE FEET": 2500,
         "YEAR BUILT": 1925, "ZIP CODE": 11225, "SALE PRICE": 1_450_000},
        {"BOROUGH": 4, "BUILDING CLASS CATEGORY": "R5 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1500, "LAND SQUARE FEET": 2800,
         "YEAR BUILT": 1950, "ZIP CODE": 11354, "SALE PRICE": 780_000},
        {"BOROUGH": 4, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 950, "LAND SQUARE FEET": 1300,
         "YEAR BUILT": 1990, "ZIP CODE": 11375, "SALE PRICE": 540_000},
        {"BOROUGH": 4, "BUILDING CLASS CATEGORY": "R7 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1700, "LAND SQUARE FEET": 3200,
         "YEAR BUILT": 1960, "ZIP CODE": 11385, "SALE PRICE": 920_000},
        {"BOROUGH": 5, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 2000, "LAND SQUARE FEET": 4000,
         "YEAR BUILT": 1970, "ZIP CODE": 10301, "SALE PRICE": 680_000},
        {"BOROUGH": 5, "BUILDING CLASS CATEGORY": "R6 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1300, "LAND SQUARE FEET": 1800,
         "YEAR BUILT": 1985, "ZIP CODE": 10304, "SALE PRICE": 450_000},
        {"BOROUGH": 5, "BUILDING CLASS CATEGORY": "R5 CONDOMINIUMS",
         "GROSS SQUARE FEET": 2100, "LAND SQUARE FEET": 3500,
         "YEAR BUILT": 1972, "ZIP CODE": 10306, "SALE PRICE": 595_000},
        # ── Drop: SALE PRICE <= 0 (deed transfer / $0 sale) ──
        {"BOROUGH": 1, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1400, "LAND SQUARE FEET": 2100,
         "YEAR BUILT": 1930, "ZIP CODE": 10003, "SALE PRICE": 0},
        # ── Drop: SALE PRICE < 10,000 (family transfer / $1 sale) ──
        {"BOROUGH": 2, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1300, "LAND SQUARE FEET": 1900,
         "YEAR BUILT": 1945, "ZIP CODE": 10453, "SALE PRICE": 1},
        # ── Drop: SALE PRICE > 100,000,000 (commercial outlier) ──
        {"BOROUGH": 1, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 5000, "LAND SQUARE FEET": 6000,
         "YEAR BUILT": 2010, "ZIP CODE": 10022, "SALE PRICE": 250_000_000},
        # ── Drop: GROSS SQUARE FEET == 0 (missing structural feature) ──
        {"BOROUGH": 3, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 0, "LAND SQUARE FEET": 1800,
         "YEAR BUILT": 1955, "ZIP CODE": 11218, "SALE PRICE": 900_000},
        # ── Drop: YEAR BUILT == 0 (missing structural feature) ──
        {"BOROUGH": 4, "BUILDING CLASS CATEGORY": "R4 CONDOMINIUMS",
         "GROSS SQUARE FEET": 1600, "LAND SQUARE FEET": 2400,
         "YEAR BUILT": 0, "ZIP CODE": 11377, "SALE PRICE": 680_000},
        # ── Drop: non-R building class (v1 scope exclusion) ──
        {"BOROUGH": 1, "BUILDING CLASS CATEGORY": "01 ONE FAMILY DWELLINGS",
         "GROSS SQUARE FEET": 1400, "LAND SQUARE FEET": 2000,
         "YEAR BUILT": 1925, "ZIP CODE": 10001, "SALE PRICE": 1_400_000},
        {"BOROUGH": 2, "BUILDING CLASS CATEGORY": "02 TWO FAMILY DWELLINGS",
         "GROSS SQUARE FEET": 1800, "LAND SQUARE FEET": 2400,
         "YEAR BUILT": 1940, "ZIP CODE": 10462, "SALE PRICE": 650_000},
        {"BOROUGH": 1, "BUILDING CLASS CATEGORY": "21 OFFICE BUILDINGS",
         "GROSS SQUARE FEET": 10000, "LAND SQUARE FEET": 8000,
         "YEAR BUILT": 1970, "ZIP CODE": 10007, "SALE PRICE": 15_000_000},
        {"BOROUGH": 2, "BUILDING CLASS CATEGORY": "29 COMMERCIAL GARAGES",
         "GROSS SQUARE FEET": 4500, "LAND SQUARE FEET": 5000,
         "YEAR BUILT": 1965, "ZIP CODE": 10456, "SALE PRICE": 3_400_000},
    ]
    return pd.DataFrame(rows)
