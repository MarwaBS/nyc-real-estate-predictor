"""Integration test — full pipeline: load data -> features -> train -> predict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.config import NUMERIC_FEATURES, ONEHOT_FEATURES, TARGET_ENCODED_FEATURES
from src.data.cleaner import clean_pipeline
from src.data.features import (
    add_geospatial_features,
    add_numeric_features,
    add_target_variables,
)
from src.models.decode import zone_for_price
from src.models.evaluate import evaluate_classifier, evaluate_regressor
from src.models.pipelines import (
    build_regression_pipeline,
)
from src.utils.validation import assert_no_leakage, validate_cleaned_data


@pytest.fixture
def integration_data() -> pd.DataFrame:
    """Larger synthetic dataset for integration testing."""
    rng = np.random.RandomState(42)
    # 600, not 200: the majority-baseline assertion below is measured on a 20%
    # split, and 40 test rows put it within one or two predictions of the
    # baseline. At that size the comparison tracked RandomForest's column
    # sampling rather than whether the pipeline learned anything -- dropping a
    # constant feature, which carries no information at all, flipped it.
    n = 600
    boroughs = ["manhattan", "brooklyn", "queens", "the bronx", "staten island"]
    # PRICE carries REAL signal from sqft + borough (log-linear + noise).
    # It was originally uniform noise independent of every feature, which
    # made any learning assertion impossible - the accuracy check could
    # only be the unfalsifiable "> 0.0". With signal, a correctly wired
    # pipeline beats the majority-class baseline comfortably, and a
    # wiring bug (target misalignment, scrambled features) fails the
    # floor below.
    borough_level = {
        "manhattan": 14.0,
        "brooklyn": 13.3,
        "queens": 13.0,
        "the bronx": 12.6,
        "staten island": 12.8,
    }
    types = ["condo", "house", "co-op", "townhouse"]
    borough_col = rng.choice(boroughs, n)
    sqft_col = rng.uniform(400, 4000, n)
    log_price = (
        np.array([borough_level[b] for b in borough_col])
        + 0.6 * np.log(sqft_col / 1500)
        + rng.normal(0, 0.25, n)
    )

    return pd.DataFrame(
        {
            "PRICE": np.exp(log_price),
            "BEDS": rng.randint(1, 6, n),
            "BATH": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n),
            "PROPERTYSQFT": sqft_col,
            "LATITUDE": rng.uniform(40.5, 40.9, n),
            "LONGITUDE": rng.uniform(-74.2, -73.7, n),
            "BOROUGH": borough_col,
            "TYPE": rng.choice(types, n),
            "SUBLOCALITY": rng.choice(
                ["midtown", "fort greene", "astoria", "pelham"], n
            ),
            "ZIPCODE": rng.choice(["10022", "11217", "11101", "10473", "10312"], n),
            "ADDRESS": [f"{i} Test St" for i in range(n)],
            "BROKERTITLE": ["test broker"] * n,
            "PROPERTY_CATEGORY": ["residential"] * n,
        }
    )


def test_full_pipeline_data_to_prediction(integration_data: pd.DataFrame) -> None:
    """End-to-end: raw data -> clean -> features -> train -> predict -> evaluate."""
    # 1. Clean
    df = clean_pipeline(integration_data)
    issues = validate_cleaned_data(df)
    assert len(issues) == 0, f"Validation failed: {issues}"
    assert len(df) > 50, "Too many rows dropped during cleaning"

    # 2. Feature engineering — the same pipeline training runs, no
    # test-manufactured columns.
    df = add_numeric_features(df)
    df = add_geospatial_features(df)
    df = add_target_variables(df)
    df = df.dropna(subset=["PRICE_ZONE", "LOG_PRICE"])

    # 3. The config feature contract IS the feature list (NO leakage)
    feature_cols = NUMERIC_FEATURES + ONEHOT_FEATURES + TARGET_ENCODED_FEATURES
    missing = [c for c in feature_cols if c not in df.columns]
    assert not missing, f"pipeline did not produce configured features: {missing}"
    available = feature_cols
    assert_no_leakage(available)

    from sklearn.model_selection import train_test_split

    y_price = df["LOG_PRICE"].values
    features = df[available]
    x_train, x_test, yp_train, yp_test, zone_train, zone_test = train_test_split(
        features,
        y_price,
        df["PRICE_ZONE"].astype(str).to_numpy(),
        test_size=0.2,
        random_state=42,
    )

    # 4. Train the one model that ships
    reg_pipeline = build_regression_pipeline(
        RandomForestRegressor(n_estimators=20, random_state=42),
    )
    reg_pipeline.fit(x_train, yp_train)
    reg_pred = reg_pipeline.predict(x_test)
    reg_metrics = evaluate_regressor(yp_test, reg_pred, log_target=True)
    # R2 is defined so that predicting the target's mean scores exactly 0, so
    # this asserts the regressor beats the trivial baseline. The previous bound
    # (-10) was satisfied by a constant predictor.
    assert reg_metrics["r2"] > 0, (
        f"R2 {reg_metrics['r2']:.3f} does not beat predicting the mean"
    )

    # 5. Zones are derived from those predictions, as in serving
    zone_pred = [zone_for_price(p) for p in np.expm1(reg_pred)]
    zone_metrics = evaluate_classifier(zone_test, zone_pred)
    majority_rate = float(pd.Series(zone_test).value_counts(normalize=True).iloc[0])
    assert zone_metrics["accuracy"] > majority_rate, (
        f"accuracy {zone_metrics['accuracy']:.3f} does not beat the "
        f"majority-class baseline {majority_rate:.3f}"
    )

    # 6. Predict single sample
    single = x_test.iloc[:1]
    price_pred = reg_pipeline.predict(single)
    assert np.isfinite(price_pred[0])
    assert zone_for_price(float(np.expm1(price_pred[0]))) in set(zone_test)


def test_no_leakage_survives_full_pipeline(integration_data: pd.DataFrame) -> None:
    """Verify leakage guard catches PRICE_PER_SQFT even after feature engineering."""
    df = clean_pipeline(integration_data)
    df = add_numeric_features(df)
    # Intentionally add leaky feature
    df["PRICE_PER_SQFT"] = df["PRICE"] / df["PROPERTYSQFT"]

    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        assert_no_leakage(list(df.columns))
