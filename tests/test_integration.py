"""Integration test — full pipeline: load data -> features -> train -> predict."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.data.cleaner import clean_pipeline
from src.data.features import add_numeric_features, add_target_variables
from src.models.evaluate import evaluate_classifier, evaluate_regressor
from src.models.pipelines import (
    build_classification_pipeline,
    build_regression_pipeline,
)
from src.utils.geo import add_distance_features
from src.utils.validation import assert_no_leakage, validate_cleaned_data


@pytest.fixture
def integration_data() -> pd.DataFrame:
    """Larger synthetic dataset for integration testing."""
    rng = np.random.RandomState(42)
    n = 200
    boroughs = ["manhattan", "brooklyn", "queens", "the bronx", "staten island"]
    types = ["condo", "house", "co-op", "townhouse"]

    return pd.DataFrame({
        "PRICE": rng.uniform(100_000, 3_000_000, n),
        "BEDS": rng.randint(1, 6, n),
        "BATH": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n),
        "PROPERTYSQFT": rng.uniform(400, 4000, n),
        "LATITUDE": rng.uniform(40.5, 40.9, n),
        "LONGITUDE": rng.uniform(-74.2, -73.7, n),
        "BOROUGH": rng.choice(boroughs, n),
        "TYPE": rng.choice(types, n),
        "SUBLOCALITY": rng.choice(["midtown", "fort greene", "astoria", "pelham"], n),
        "ZIPCODE": rng.choice(["10022", "11217", "11101", "10473", "10312"], n),
        "ADDRESS": [f"{i} Test St" for i in range(n)],
        "BROKERTITLE": ["test broker"] * n,
        "PROPERTY_CATEGORY": ["residential"] * n,
    })


def test_full_pipeline_data_to_prediction(integration_data: pd.DataFrame) -> None:
    """End-to-end: raw data -> clean -> features -> train -> predict -> evaluate."""
    # 1. Clean
    df = clean_pipeline(integration_data)
    issues = validate_cleaned_data(df)
    assert len(issues) == 0, f"Validation failed: {issues}"
    assert len(df) > 50, "Too many rows dropped during cleaning"

    # 2. Feature engineering
    df = add_numeric_features(df)
    ref_points = {"MANHATTAN_CENTER": (40.7580, -73.9855)}
    df = add_distance_features(df, ref_points)
    df["DIST_CENTRAL_PARK"] = df["DIST_MANHATTAN_CENTER"]
    df["DIST_NEAREST_SUBWAY"] = df["DIST_MANHATTAN_CENTER"]
    df = add_target_variables(df)
    df = df.dropna(subset=["PRICE_ZONE", "LOG_PRICE"])

    # 3. Prepare features (NO leakage)
    feature_cols = [
        "BEDS", "BATH", "PROPERTYSQFT", "TOTAL_ROOMS", "BED_BATH_RATIO",
        "LOG_SQFT", "ROOMS_PER_SQFT", "DIST_MANHATTAN_CENTER",
        "DIST_CENTRAL_PARK", "DIST_NEAREST_SUBWAY",
        "BOROUGH", "TYPE", "PROPERTY_CATEGORY", "ZIPCODE", "SUBLOCALITY",
    ]
    available = [c for c in feature_cols if c in df.columns]
    assert_no_leakage(available)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_zone = le.fit_transform(df["PRICE_ZONE"])
    y_price = df["LOG_PRICE"].values

    features = df[available]
    x_train, x_test, yz_train, yz_test, yp_train, yp_test = train_test_split(
        features, y_zone, y_price, test_size=0.2, random_state=42,
    )

    # 4. Train classification
    clf_pipeline = build_classification_pipeline(
        RandomForestClassifier(n_estimators=20, random_state=42),
    )
    clf_pipeline.fit(x_train, yz_train)
    clf_pred = clf_pipeline.predict(x_test)
    clf_metrics = evaluate_classifier(yz_test, clf_pred)
    assert clf_metrics["accuracy"] > 0.0, "Classifier accuracy must be positive"
    assert 0 <= clf_metrics["macro_f1"] <= 1.0

    # 5. Train regression
    reg_pipeline = build_regression_pipeline(
        RandomForestRegressor(n_estimators=20, random_state=42),
    )
    reg_pipeline.fit(x_train, yp_train)
    reg_pred = reg_pipeline.predict(x_test)
    reg_metrics = evaluate_regressor(yp_test, reg_pred, log_target=True)
    assert reg_metrics["r2"] > -10, "Regressor R2 must be reasonable"

    # 6. Predict single sample
    single = x_test.iloc[:1]
    zone_pred = clf_pipeline.predict(single)
    price_pred = reg_pipeline.predict(single)
    assert len(zone_pred) == 1
    assert np.isfinite(price_pred[0])


def test_no_leakage_survives_full_pipeline(integration_data: pd.DataFrame) -> None:
    """Verify leakage guard catches PRICE_PER_SQFT even after feature engineering."""
    df = clean_pipeline(integration_data)
    df = add_numeric_features(df)
    # Intentionally add leaky feature
    df["PRICE_PER_SQFT"] = df["PRICE"] / df["PROPERTYSQFT"]

    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        assert_no_leakage(list(df.columns))
