"""Train price regression models — XGBoost, LightGBM, CatBoost, RF."""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

from src.config import CV_FOLDS, MODELS_DIR, OPTUNA_TRIALS, RANDOM_SEED
from src.models.evaluate import evaluate_regressor
from src.models.pipelines import build_preprocessor, build_regression_pipeline
from src.utils.validation import assert_no_leakage

logger = logging.getLogger(__name__)


def tune_regression_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str = "xgboost",
    n_trials: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Optuna tuning for regression models."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = n_trials or OPTUNA_TRIALS
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    preprocessor = build_preprocessor()

    def objective(trial: optuna.Trial) -> float:
        if model_name == "xgboost":
            from xgboost import XGBRegressor
            model = XGBRegressor(
                max_depth=trial.suggest_int("max_depth", 3, 10),
                n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=RANDOM_SEED, n_jobs=-1,
            )
        elif model_name == "lightgbm":
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                num_leaves=trial.suggest_int("num_leaves", 20, 150),
                n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
                random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
            )
        else:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(
                depth=trial.suggest_int("depth", 4, 10),
                iterations=trial.suggest_int("iterations", 200, 1000, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                random_seed=RANDOM_SEED, verbose=0,
            )

        pipeline = build_regression_pipeline(model, preprocessor)
        scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", study_name=f"reg_{model_name}")
    study.optimize(objective, n_trials=n_trials)

    logger.info("Optuna %s regression: best R2=%.4f", model_name, study.best_value)
    return study.best_params, study.best_params


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
