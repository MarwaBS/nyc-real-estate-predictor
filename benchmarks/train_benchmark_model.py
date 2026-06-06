"""Train the lean external-benchmark regressor.

The flagship model uses listing-level features (BEDS, BATH, lat/lon-derived
distances, SUBLOCALITY) that do **not** exist in NYC.gov Rolling Sales
transaction records, so it cannot be validated against that external source.

This lean model trains on only the three features that the Kaggle training
data and NYC.gov Rolling Sales genuinely share — **borough**, **property
square footage**, and **ZIP** — so :mod:`benchmarks.run_benchmark` can score
real, unseen NYC.gov 2024 sales and report an honest out-of-distribution R².

It is deliberately lower-accuracy than the flagship: the point is *honest
external validation on real data*, not peak in-distribution accuracy. The
artefact is written to ``models/benchmark_regressor.joblib`` (gitignored;
the benchmark job loads it at run time).

Feature contract (must match :func:`benchmarks.mapping.apply_schema_map`):
    borough        : str   US Census borough name (Manhattan, Bronx, ...)
    property_sqft  : float gross/property square footage
    zip_code       : str   5-digit ZIP, as a category

Target: ``log1p(PRICE)`` (the benchmark compares in log space).
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import CLEANED_DATASET, MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

#: The three features shared between Kaggle training data and NYC.gov sales.
BENCHMARK_FEATURES: list[str] = ["borough", "property_sqft", "zip_code"]
BENCHMARK_MODEL_PATH: Path = MODELS_DIR / "benchmark_regressor.joblib"


def build_benchmark_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Project the cleaned Kaggle dataset onto the shared benchmark schema.

    Returns ``(X, y)`` where ``X`` has exactly :data:`BENCHMARK_FEATURES`
    and ``y`` is ``log1p(PRICE)``. Rows missing any shared feature or with a
    non-positive price are dropped — the model only learns from complete,
    valid examples.
    """
    work = df[["BOROUGH", "PROPERTYSQFT", "ZIPCODE", "PRICE"]].copy()
    work = work[(work["PRICE"] > 0) & (work["PROPERTYSQFT"] > 0)].dropna()

    x = pd.DataFrame(
        {
            "borough": work["BOROUGH"].astype(str),
            "property_sqft": work["PROPERTYSQFT"].astype(float),
            "zip_code": work["ZIPCODE"].astype(int).astype(str),
        }
    )
    y = pd.Series(np.log1p(work["PRICE"].to_numpy(dtype=float)), index=x.index, name="log_price")
    return x, y


def build_benchmark_pipeline() -> Pipeline:
    """Pipeline: one-hot borough + ZIP (rare-grouped), passthrough sqft → HGB.

    A tree model needs no scaling. ``min_frequency`` folds rare ZIPs into an
    ``infrequent`` bucket so an unseen NYC.gov ZIP maps cleanly instead of
    exploding the feature space or erroring.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["borough"]),
            (
                "zip",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=25, sparse_output=False
                ),
                ["zip_code"],
            ),
            ("sqft", "passthrough", ["property_sqft"]),
        ]
    )
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6, random_state=RANDOM_SEED
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_benchmark_model(cleaned_path: Path = CLEANED_DATASET) -> Pipeline:
    """Train and persist the benchmark regressor; return the fitted pipeline."""
    logger.info("Loading cleaned dataset from %s", cleaned_path)
    df = pd.read_csv(cleaned_path)
    x, y = build_benchmark_frame(df)
    logger.info("Benchmark training frame: %d rows, features=%s", len(x), BENCHMARK_FEATURES)

    pipeline = build_benchmark_pipeline()
    pipeline.fit(x, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, BENCHMARK_MODEL_PATH)
    logger.info("Saved benchmark regressor to %s", BENCHMARK_MODEL_PATH)
    return pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    train_benchmark_model()
