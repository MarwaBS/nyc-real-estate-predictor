"""Train price regression models — XGBoost, LightGBM, CatBoost, RF."""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import MODELS_DIR, RANDOM_SEED
from src.models.evaluate import evaluate_regressor
from src.models.pipelines import build_regression_pipeline
from src.utils.validation import assert_no_leakage

logger = logging.getLogger(__name__)


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Train all regression models, evaluate, save best."""
    assert_no_leakage(list(X_train.columns))
    results: dict[str, Any] = {}

    models_to_train = {
        "random_forest": RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }

    # Add gradient boosting models
    try:
        from xgboost import XGBRegressor
        models_to_train["xgboost"] = XGBRegressor(
            max_depth=6, n_estimators=500, learning_rate=0.1,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    except ImportError:
        logger.warning("XGBoost not installed — skipping")

    try:
        from lightgbm import LGBMRegressor
        models_to_train["lightgbm"] = LGBMRegressor(
            num_leaves=63, n_estimators=500, learning_rate=0.1,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        logger.warning("LightGBM not installed — skipping")

    try:
        from catboost import CatBoostRegressor
        models_to_train["catboost"] = CatBoostRegressor(
            depth=6, iterations=500, learning_rate=0.1,
            random_seed=RANDOM_SEED, verbose=0,
        )
    except ImportError:
        logger.warning("CatBoost not installed — skipping")

    for name, model in models_to_train.items():
        logger.info("=== Training %s regressor ===", name)
        pipeline = build_regression_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_regressor(y_test, y_pred, log_target=True)
        results[name] = {**metrics, "pipeline": pipeline}

    # Find and save best
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_pipeline = results[best_name]["pipeline"]
    logger.info("Best regressor: %s (R2=%.4f)", best_name, results[best_name]["r2"])

    model_path = MODELS_DIR / "price_regressor_best.joblib"
    joblib.dump(best_pipeline, model_path)
    logger.info("Saved best regressor to %s", model_path)

    return results
