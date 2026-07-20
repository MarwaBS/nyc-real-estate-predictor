"""Centralized configuration — all paths, constants, and env vars in one place."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "Resources"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATASET = DATA_RAW_DIR / "NY-House-Dataset.csv"
CLEANED_DATASET = PROJECT_ROOT / "output" / "cleaned_house_dataset.csv"

# Ensure output dirs exist (best-effort — read-only runtimes like HF Spaces skip silently)
for _dir in (DATA_PROCESSED_DIR, MODELS_DIR):
    with contextlib.suppress(PermissionError, OSError):
        _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API Keys (from environment — never hardcoded)
# ---------------------------------------------------------------------------
GEOAPIFY_API_KEY: str = os.getenv("GEOAPIFY_API_KEY", "")
GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY", "")

# ---------------------------------------------------------------------------
# Model / training constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
# Fraction of the post-test remainder held out for model selection. Every
# choice made during training reads VAL; TEST is scored once, at the end, by
# the already-chosen model.
VAL_SIZE: float = 0.2
CV_FOLDS: int = 5
OPTUNA_TRIALS: int = 50

# Price zone thresholds (USD): equal-frequency quartiles of the cleaned
# training prices, so the four zones carry ~1,130 listings each.
# test_config_artefact_agreement.py fails if these drift from the fitted model.
PRICE_ZONE_BINS: list[float] = [0, 499_000, 825_000, 1_495_000, float("inf")]
PRICE_ZONE_LABELS: list[str] = ["Low", "Medium", "High", "Very High"]

# Feature lists — CRITICAL: no PRICE_PER_SQFT (data leakage)
NUMERIC_FEATURES: list[str] = [
    "BEDS",
    "BATH",
    "PROPERTYSQFT",
    "TOTAL_ROOMS",
    "BED_BATH_RATIO",
    "ROOMS_PER_SQFT",
    "DIST_MANHATTAN_CENTER",
    "DIST_CENTRAL_PARK",
]

ONEHOT_FEATURES: list[str] = ["BOROUGH", "TYPE"]

TARGET_ENCODED_FEATURES: list[str] = ["ZIPCODE", "SUBLOCALITY"]

# NYC geographic constants
MANHATTAN_CENTER = (40.7580, -73.9855)
CENTRAL_PARK = (40.7829, -73.9654)

# Boroughs
BOROUGH_MAP: dict[str, str] = {
    "new york county": "manhattan",
    "kings county": "brooklyn",
    "queens county": "queens",
    "bronx county": "the bronx",
    "richmond county": "staten island",
    "manhattan": "manhattan",
    "brooklyn": "brooklyn",
    "queens": "queens",
    "the bronx": "the bronx",
    "staten island": "staten island",
}
