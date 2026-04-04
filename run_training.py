"""End-to-end training orchestrator — load data, engineer features, train models, save artifacts."""
from __future__ import annotations

import logging
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

from src.config import (
    MODELS_DIR,
    NUMERIC_FEATURES,
    ONEHOT_FEATURES,
    PRICE_ZONE_LABELS,
    RANDOM_SEED,
    TARGET_ENCODED_FEATURES,
    TEST_SIZE,
)
from src.data.features import add_numeric_features, add_target_variables, cap_categorical_cardinality
from src.data.loader import load_cleaned
from src.models.evaluate import evaluate_classifier, evaluate_fairness_by_group, evaluate_regressor
from src.models.pipelines import build_classification_pipeline, build_regression_pipeline
from src.utils.geo import add_distance_features, add_neighborhood_clusters
from src.utils.logging_config import setup_logging
from src.utils.validation import assert_no_leakage

setup_logging()
logger = logging.getLogger(__name__)

REFERENCE_POINTS = {
    "MANHATTAN_CENTER": (40.7580, -73.9855),
    "CENTRAL_PARK": (40.7829, -73.9654),
}


def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load, clean, and feature-engineer the full dataset."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)
    df = load_cleaned()

    # Normalize column names
    df.columns = df.columns.str.upper().str.strip()

    # Ensure BOROUGH exists
    if "BOROUGH" not in df.columns:
        logger.warning("BOROUGH column missing — creating from SUBLOCALITY")
        df["BOROUGH"] = df.get("SUBLOCALITY", "unknown")

    # Ensure PROPERTY_CATEGORY exists
    if "PROPERTY_CATEGORY" not in df.columns:
        df["PROPERTY_CATEGORY"] = "residential"

    # Ensure SUBLOCALITY exists
    if "SUBLOCALITY" not in df.columns:
        df["SUBLOCALITY"] = "unknown"

    logger.info("=" * 60)
    logger.info("STEP 2: Feature engineering")
    logger.info("=" * 60)

    # Numeric features
    df = add_numeric_features(df)

    # Geospatial features
    df = add_distance_features(df, REFERENCE_POINTS)

    # Subway proxy (use manhattan distance until we have subway data)
    df["DIST_NEAREST_SUBWAY"] = df["DIST_MANHATTAN_CENTER"]

    # Neighborhood clusters
    df = add_neighborhood_clusters(df, n_clusters=15)

    # Target variables
    df = add_target_variables(df)

    # Cap high-cardinality categoricals
    df = cap_categorical_cardinality(df, columns=["SUBLOCALITY", "TYPE", "ZIPCODE"])

    # Ensure all needed columns are lowercase string for categoricals
    for col in ONEHOT_FEATURES + TARGET_ENCODED_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # Drop rows with NaN in target
    df = df.dropna(subset=["PRICE_ZONE", "LOG_PRICE"])

    logger.info("Engineered dataset: %d rows x %d cols", *df.shape)

    # Extract targets
    y_zone = df["PRICE_ZONE"]
    y_log_price = df["LOG_PRICE"]
    borough_col = df["BOROUGH"].copy()

    return df, y_zone, y_log_price, borough_col


def get_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only feature columns — no targets, no leaky features."""
    all_features = NUMERIC_FEATURES + ONEHOT_FEATURES + TARGET_ENCODED_FEATURES
    available = [c for c in all_features if c in df.columns]
    missing = set(all_features) - set(available)
    if missing:
        logger.warning("Missing features (will be skipped): %s", missing)

    X = df[available].copy()
    assert_no_leakage(list(X.columns))
    return X


def train_classification(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    borough_test: pd.Series,
) -> None:
    """Train and evaluate all classification models."""
    logger.info("=" * 60)
    logger.info("STEP 3: Training classification models")
    logger.info("=" * 60)

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    models = {
        "xgboost": XGBClassifier(
            max_depth=6, n_estimators=500, learning_rate=0.1,
            eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            num_leaves=63, n_estimators=500, learning_rate=0.1,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        ),
    }

    best_f1 = -1.0
    best_name = ""
    best_pipeline = None

    for name, model in models.items():
        logger.info("--- Training %s ---", name)
        pipeline = build_classification_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_classifier(y_test, y_pred, PRICE_ZONE_LABELS)

        logger.info(
            "%s: accuracy=%.4f, macro_f1=%.4f, kappa=%.4f",
            name, metrics["accuracy"], metrics["macro_f1"], metrics["cohen_kappa"],
        )

        # MLflow experiment tracking
        if _HAS_MLFLOW:
            mlflow.set_experiment("price_zone_classification")
            with mlflow.start_run(run_name=f"clf_{name}"):
                mlflow.log_params({"model": name, "n_features": len(X_train.columns),
                                   "train_size": len(X_train), "test_size": len(X_test)})
                mlflow.log_metrics({"accuracy": metrics["accuracy"],
                                    "macro_f1": metrics["macro_f1"],
                                    "cohen_kappa": metrics["cohen_kappa"]})
                mlflow.sklearn.log_model(pipeline, f"model_{name}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_name = name
            best_pipeline = pipeline

    # Fairness analysis
    if best_pipeline is not None:
        y_best_pred = best_pipeline.predict(X_test)
        fairness = evaluate_fairness_by_group(y_test, y_best_pred, borough_test)
        logger.info("Fairness by borough: %s", fairness)

        # Save best model
        path = MODELS_DIR / "price_zone_best.joblib"
        joblib.dump(best_pipeline, path)
        logger.info("Saved best classifier (%s, macro_f1=%.4f) to %s", best_name, best_f1, path)


def train_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> None:
    """Train and evaluate all regression models."""
    logger.info("=" * 60)
    logger.info("STEP 4: Training regression models")
    logger.info("=" * 60)

    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            max_depth=6, n_estimators=500, learning_rate=0.1,
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(
            num_leaves=63, n_estimators=500, learning_rate=0.1,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        ),
    }

    best_r2 = -999.0
    best_name = ""
    best_pipeline = None

    for name, model in models.items():
        logger.info("--- Training %s regressor ---", name)
        pipeline = build_regression_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_regressor(y_test, y_pred, log_target=True)

        logger.info(
            "%s: R2=%.4f, RMSE=%.4f, MAE_USD=$%.0f",
            name, metrics["r2"], metrics["rmse"], metrics.get("mae_usd", 0),
        )

        # MLflow experiment tracking
        if _HAS_MLFLOW:
            mlflow.set_experiment("price_regression")
            with mlflow.start_run(run_name=f"reg_{name}"):
                mlflow.log_params({"model": name, "target": "LOG_PRICE",
                                   "train_size": len(X_train), "test_size": len(X_test)})
                mlflow.log_metrics({"r2": metrics["r2"], "rmse": metrics["rmse"],
                                    "mae": metrics["mae"],
                                    "mae_usd": metrics.get("mae_usd", 0)})
                mlflow.sklearn.log_model(pipeline, f"model_{name}")

        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is not None:
        path = MODELS_DIR / "price_regressor_best.joblib"
        joblib.dump(best_pipeline, path)
        logger.info("Saved best regressor (%s, R2=%.4f) to %s", best_name, best_r2, path)


def main() -> None:
    """Run the full training pipeline."""
    logger.info("=" * 60)
    logger.info("NYC PRICE PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Prepare data
    df, y_zone, y_log_price, borough = prepare_data()
    X = get_feature_df(df)

    # 2. Encode zone labels
    le = LabelEncoder()
    y_zone_encoded = le.fit_transform(y_zone)

    # Save label encoder
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    logger.info("Zone classes: %s", list(le.classes_))

    # 3. Train/test split (stratified for classification)
    X_train, X_test, y_zone_train, y_zone_test, y_price_train, y_price_test, borough_train, borough_test = (
        train_test_split(
            X, y_zone_encoded, y_log_price, borough,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y_zone_encoded,
        )
    )

    logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))
    logger.info("Features: %s", list(X_train.columns))

    # 4. Train classification
    train_classification(X_train, y_zone_train, X_test, y_zone_test, borough_test)

    # 5. Train regression
    train_regression(X_train, y_price_train, X_test, y_price_test)

    # 6. Generate SHAP explanations
    logger.info("=" * 60)
    logger.info("STEP 5: SHAP explainability")
    logger.info("=" * 60)
    try:
        best_clf = joblib.load(MODELS_DIR / "price_zone_best.joblib")
        # Get preprocessed features for SHAP
        preprocessor = best_clf.named_steps["preprocessor"]
        X_test_transformed = preprocessor.transform(X_test)
        feature_names = list(preprocessor.get_feature_names_out())

        from src.models.explain import compute_shap_values, global_feature_importance
        classifier_step = best_clf.named_steps["classifier"]
        shap_values, explainer = compute_shap_values(classifier_step, pd.DataFrame(X_test_transformed, columns=feature_names), max_samples=200)
        importance_df = global_feature_importance(shap_values, feature_names)
        logger.info("Top 10 features by SHAP:\n%s", importance_df.head(10).to_string())
    except Exception as exc:
        logger.warning("SHAP analysis failed (non-critical): %s", exc)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Models saved to: %s", MODELS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
