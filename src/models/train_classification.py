"""Train price zone classification models — XGBoost, LightGBM, CatBoost, RF, Stacking."""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import (
    CV_FOLDS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    ONEHOT_FEATURES,
    OPTUNA_TRIALS,
    PRICE_ZONE_LABELS,
    RANDOM_SEED,
    TARGET_ENCODED_FEATURES,
)
from src.models.evaluate import evaluate_classifier, evaluate_fairness_by_group
from src.models.pipelines import build_classification_pipeline, build_preprocessor
from src.utils.validation import assert_no_leakage

logger = logging.getLogger(__name__)


def _get_feature_columns() -> list[str]:
    """Return the feature columns used for training (no targets, no leaky features)."""
    return NUMERIC_FEATURES + ONEHOT_FEATURES + TARGET_ENCODED_FEATURES


def _resample_smote_enn(
    X: pd.DataFrame, y: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply SMOTE-ENN for balanced resampling (better than SMOTE alone)."""
    try:
        from imblearn.combine import SMOTEENN
        sampler = SMOTEENN(random_state=RANDOM_SEED)
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info("SMOTE-ENN: %d -> %d samples", len(X), len(X_res))
        return pd.DataFrame(X_res, columns=X.columns), y_res
    except ImportError:
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=RANDOM_SEED)
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info("SMOTE fallback: %d -> %d samples", len(X), len(X_res))
        return pd.DataFrame(X_res, columns=X.columns), y_res


def _build_xgboost(trial: Any = None) -> Any:
    """Build XGBoost classifier with optional Optuna trial for hyperparams."""
    from xgboost import XGBClassifier

    if trial is not None:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    else:
        params = {"max_depth": 6, "n_estimators": 500, "learning_rate": 0.1}

    return XGBClassifier(
        **params,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def _build_lightgbm(trial: Any = None) -> Any:
    """Build LightGBM classifier."""
    from lightgbm import LGBMClassifier

    if trial is not None:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    else:
        params = {"num_leaves": 63, "n_estimators": 500, "learning_rate": 0.1}

    return LGBMClassifier(
        **params,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )


def _build_catboost(trial: Any = None) -> Any:
    """Build CatBoost classifier."""
    from catboost import CatBoostClassifier

    if trial is not None:
        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    else:
        params = {"depth": 6, "iterations": 500, "learning_rate": 0.1}

    return CatBoostClassifier(
        **params,
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=0,
    )


def tune_with_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str = "xgboost",
    n_trials: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Bayesian hyperparameter optimization with Optuna."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = n_trials or OPTUNA_TRIALS

    builders = {
        "xgboost": _build_xgboost,
        "lightgbm": _build_lightgbm,
        "catboost": _build_catboost,
    }
    builder = builders[model_name]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    preprocessor = build_preprocessor()

    def objective(trial: optuna.Trial) -> float:
        model = builder(trial)
        pipeline = build_classification_pipeline(model, preprocessor)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1,
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", study_name=f"{model_name}_tuning")
    study.optimize(objective, n_trials=n_trials)

    logger.info(
        "Optuna %s: best macro_f1=%.4f, params=%s",
        model_name,
        study.best_value,
        study.best_params,
    )

    # Rebuild best model
    best_model = builder(study.best_trial)
    best_pipeline = build_classification_pipeline(best_model)
    best_pipeline.fit(X_train, y_train)

    return best_pipeline, study.best_params


def build_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Any:
    """Stacking ensemble: XGB + LGBM + CatBoost -> LogisticRegression meta-learner."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    preprocessor = build_preprocessor()

    estimators = [
        ("xgb", build_classification_pipeline(
            XGBClassifier(max_depth=6, n_estimators=500, learning_rate=0.1,
                          eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1),
            preprocessor,
        )),
        ("lgbm", build_classification_pipeline(
            LGBMClassifier(num_leaves=63, n_estimators=500, learning_rate=0.1,
                           class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
        )),
        ("catboost", build_classification_pipeline(
            CatBoostClassifier(depth=6, iterations=500, learning_rate=0.1,
                               auto_class_weights="Balanced", random_seed=RANDOM_SEED, verbose=0),
        )),
    ]

    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED),
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
    )

    logger.info("Fitting stacking ensemble (XGB + LGBM + CatBoost -> LR)...")
    stacker.fit(X_train, y_train)
    return stacker


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    borough_test: pd.Series | None = None,
) -> dict[str, Any]:
    """Train all classification models, evaluate, and save the best one."""
    feature_cols = list(X_train.columns)
    assert_no_leakage(feature_cols)

    results: dict[str, Any] = {}

    # --- 1. Baseline: Random Forest ---
    logger.info("=== Training Random Forest baseline ===")
    rf_pipeline = build_classification_pipeline(
        RandomForestClassifier(
            n_estimators=500, max_depth=20, class_weight="balanced_subsample",
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    )
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    results["random_forest"] = evaluate_classifier(y_test, rf_pred, PRICE_ZONE_LABELS)
    results["random_forest"]["pipeline"] = rf_pipeline

    # --- 2. XGBoost with Optuna ---
    logger.info("=== Tuning XGBoost with Optuna ===")
    xgb_pipeline, xgb_params = tune_with_optuna(X_train, y_train, "xgboost")
    xgb_pred = xgb_pipeline.predict(X_test)
    results["xgboost"] = evaluate_classifier(y_test, xgb_pred, PRICE_ZONE_LABELS)
    results["xgboost"]["pipeline"] = xgb_pipeline
    results["xgboost"]["best_params"] = xgb_params

    # --- 3. LightGBM with Optuna ---
    logger.info("=== Tuning LightGBM with Optuna ===")
    lgbm_pipeline, lgbm_params = tune_with_optuna(X_train, y_train, "lightgbm")
    lgbm_pred = lgbm_pipeline.predict(X_test)
    results["lightgbm"] = evaluate_classifier(y_test, lgbm_pred, PRICE_ZONE_LABELS)
    results["lightgbm"]["pipeline"] = lgbm_pipeline
    results["lightgbm"]["best_params"] = lgbm_params

    # --- 4. CatBoost with Optuna ---
    logger.info("=== Tuning CatBoost with Optuna ===")
    cb_pipeline, cb_params = tune_with_optuna(X_train, y_train, "catboost")
    cb_pred = cb_pipeline.predict(X_test)
    results["catboost"] = evaluate_classifier(y_test, cb_pred, PRICE_ZONE_LABELS)
    results["catboost"]["pipeline"] = cb_pipeline
    results["catboost"]["best_params"] = cb_params

    # --- 5. Find best model by macro F1 ---
    best_name = max(
        (k for k in results if k != "stacking"),
        key=lambda k: results[k]["macro_f1"],
    )
    best_pipeline = results[best_name]["pipeline"]

    logger.info("Best classifier: %s (macro_f1=%.4f)", best_name, results[best_name]["macro_f1"])

    # Save best model
    model_path = MODELS_DIR / "price_zone_best.joblib"
    joblib.dump(best_pipeline, model_path)
    logger.info("Saved best model to %s", model_path)

    # Fairness analysis by borough
    if borough_test is not None:
        best_pred = best_pipeline.predict(X_test)
        results["fairness"] = evaluate_fairness_by_group(y_test, best_pred, borough_test)

    return results


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    setup_logging()
    logger.info("Run train_classification.py via: python -m src.models.train_classification")
