"""End-to-end training orchestrator — load data, engineer features, train models, save artifacts.

Besides the model artefacts (local-only; DVC without a remote), every run
writes ``reports/training_metrics.json`` — the committed evidence artefact
behind the README's headline numbers. It records the metrics of the
selected models together with provenance (commit SHA, library versions,
seed, split sizes) so the README quotes a file, not a memory.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

from src.config import (
    CLEANED_DATASET,
    MODELS_DIR,
    NUMERIC_FEATURES,
    ONEHOT_FEATURES,
    RANDOM_SEED,
    TARGET_ENCODED_FEATURES,
    TEST_SIZE,
    VAL_SIZE,
)
from src.data.cleaner import clean_pipeline
from src.data.features import (
    add_numeric_features,
    add_target_variables,
    cap_categorical_cardinality,
)
from src.data.loader import load_raw
from src.models.evaluate import (
    evaluate_classifier,
    evaluate_fairness_by_group,
    evaluate_regressor,
)
from src.models.pipelines import (
    build_classification_pipeline,
    build_regression_pipeline,
)
from src.utils.geo import add_distance_features
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
    logger.info("Step 1: loading raw data and cleaning it")

    # Train from the RAW snapshot through the cleaning pipeline, rather than
    # reading a pre-cleaned CSV. The cleaned file previously had no producer
    # anywhere in the repo: it was a committed artefact whose contents no code
    # could regenerate (it held a 2,147,483,647 sentinel PRICE that this
    # pipeline's IQR cap makes impossible), so `python run_training.py` did not
    # reproduce the shipped models. Cleaning here makes the artefact derived.
    df = clean_pipeline(load_raw())

    # Written for the benchmark trainer and the EDA notebook, which both read
    # the cleaned CSV. It is an output of this script, never an input to it.
    CLEANED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATASET, index=False)
    logger.info("Wrote cleaned dataset to %s", CLEANED_DATASET)

    logger.info("Step 2: feature engineering")

    # Numeric features
    df = add_numeric_features(df)

    # Geospatial features
    df = add_distance_features(df, REFERENCE_POINTS)

    # Subway distance is BY DESIGN the Manhattan-center proxy — identical
    # semantics at train and serve time (api/main.py computes the same
    # value; see MODEL_CARD.md). Station-level data is not bundled.
    df["DIST_NEAREST_SUBWAY"] = df["DIST_MANHATTAN_CENTER"]

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
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    borough_test: pd.Series,
    class_labels: list[str],
) -> dict[str, Any]:
    """Train candidates, pick the winner on VAL, score it once on TEST.

    The val/test separation is the point of this function's shape: candidate
    macro-F1 is compared on ``X_val`` only, and ``X_test`` is touched exactly
    once, after ``best_pipeline`` is already fixed. Comparing candidates on
    test makes the winner's test score optimistic by construction — it is the
    max of several draws on the same sample.

    ``class_labels`` MUST be the label encoder's ``classes_`` (the names in
    encoded-index order), not the semantic config order — the two disagree,
    and naming report rows with the config order misattributes 3 of the 4
    per-class rows.

    Returns the selected model's record (name, metrics, fairness) for the
    committed training-metrics artefact.
    """
    logger.info("STEP 3: Training classification models")

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    models = {
        "xgboost": XGBClassifier(
            max_depth=6,
            n_estimators=500,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            num_leaves=63,
            n_estimators=500,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    best_f1 = -1.0
    best_name = ""
    best_pipeline = None
    best_val_metrics: dict[str, Any] = {}
    candidates: dict[str, Any] = {}

    for name, model in models.items():
        logger.info("--- Training %s ---", name)
        pipeline = build_classification_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        metrics = evaluate_classifier(y_val, y_pred, class_labels)
        candidates[name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "cohen_kappa": metrics["cohen_kappa"],
        }

        logger.info(
            "%s (val): accuracy=%.4f, macro_f1=%.4f, kappa=%.4f",
            name,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["cohen_kappa"],
        )

        # MLflow experiment tracking
        if _HAS_MLFLOW:
            mlflow.set_experiment("price_zone_classification")
            with mlflow.start_run(run_name=f"clf_{name}"):
                mlflow.log_params(
                    {
                        "model": name,
                        "n_features": len(X_train.columns),
                        "train_size": len(X_train),
                        "val_size": len(X_val),
                    }
                )
                mlflow.log_metrics(
                    {
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "cohen_kappa": metrics["cohen_kappa"],
                    }
                )
                mlflow.sklearn.log_model(pipeline, f"model_{name}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_name = name
            best_pipeline = pipeline
            best_val_metrics = metrics

    # The single test-set read. `best_pipeline` is already decided above, so
    # this number is a genuine hold-out estimate rather than a selected max.
    fairness: dict[str, Any] = {}
    best_metrics: dict[str, Any] = {}
    if best_pipeline is not None:
        y_best_pred = best_pipeline.predict(X_test)
        best_metrics = evaluate_classifier(y_test, y_best_pred, class_labels)
        logger.info(
            "SELECTED %s — val macro_f1=%.4f, test macro_f1=%.4f",
            best_name,
            best_f1,
            best_metrics["macro_f1"],
        )
        fairness = evaluate_fairness_by_group(y_test, y_best_pred, borough_test)
        logger.info("Fairness by borough: %s", fairness)

        path = MODELS_DIR / "price_zone_best.joblib"
        joblib.dump(best_pipeline, path)
        logger.info("Saved best classifier (%s) to %s", best_name, path)

    return {
        "selected_model": best_name,
        "metrics": best_metrics,
        "selection_metrics_val": best_val_metrics,
        "candidates_val": candidates,
        "fairness_by_borough": fairness,
    }


def train_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Train candidates, pick the winner on VAL, score it once on TEST.

    Same discipline as ``train_classification``: three candidates compared on
    R2 is three draws, and picking the max of those on test would report the
    luckiest draw as if it were a hold-out estimate.

    Returns the selected model's record (name, metrics) for the committed
    training-metrics artefact.
    """
    logger.info("STEP 4: Training regression models")

    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    models = {
        # min_samples_leaf bounds a forest that was previously unbounded:
        # 500 fully-grown trees on 2,882 rows averaged 3,583 nodes each —
        # about one leaf per training sample — for a 129 MB artifact that
        # exceeds GitHub's 100 MB file limit and so could not be committed to
        # the model registry at all. Ten is a floor with a meaning: a leaf's
        # prediction is the mean log-price of the sales in it, and a mean over
        # fewer than ten comparable sales is not an estimate worth serving.
        # This went unnoticed while candidate selection read the test split,
        # where XGBoost won and the forest was never saved.
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            max_depth=6,
            n_estimators=500,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(
            num_leaves=63,
            n_estimators=500,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    best_r2 = -999.0
    best_name = ""
    best_pipeline = None
    best_val_metrics: dict[str, Any] = {}
    candidates: dict[str, Any] = {}

    for name, model in models.items():
        logger.info("--- Training %s regressor ---", name)
        pipeline = build_regression_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        metrics = evaluate_regressor(y_val, y_pred, log_target=True)
        candidates[name] = {
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae_usd": metrics.get("mae_usd"),
        }

        logger.info(
            "%s (val): R2=%.4f, RMSE=%.4f, MAE_USD=$%.0f",
            name,
            metrics["r2"],
            metrics["rmse"],
            metrics.get("mae_usd", 0),
        )

        # MLflow experiment tracking
        if _HAS_MLFLOW:
            mlflow.set_experiment("price_regression")
            with mlflow.start_run(run_name=f"reg_{name}"):
                mlflow.log_params(
                    {
                        "model": name,
                        "target": "LOG_PRICE",
                        "train_size": len(X_train),
                        "val_size": len(X_val),
                    }
                )
                mlflow.log_metrics(
                    {
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "mae_usd": metrics.get("mae_usd", 0),
                    }
                )
                mlflow.sklearn.log_model(pipeline, f"model_{name}")

        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_name = name
            best_pipeline = pipeline
            best_val_metrics = metrics

    best_metrics: dict[str, Any] = {}
    if best_pipeline is not None:
        # The single test-set read, after selection is already fixed.
        best_metrics = evaluate_regressor(
            y_test, best_pipeline.predict(X_test), log_target=True
        )
        logger.info(
            "SELECTED %s — val R2=%.4f, test R2=%.4f",
            best_name,
            best_r2,
            best_metrics["r2"],
        )
        path = MODELS_DIR / "price_regressor_best.joblib"
        joblib.dump(best_pipeline, path)
        logger.info("Saved best regressor (%s) to %s", best_name, path)

    return {
        "selected_model": best_name,
        "metrics": best_metrics,
        "selection_metrics_val": best_val_metrics,
        "candidates_val": candidates,
    }


# Fraction of listings the served price interval is calibrated to contain.
# 0.80 rather than a tighter target because this model's honest spread is
# already wide (see the multipliers written into the artefact): a narrower
# nominal target would produce an interval that is precise-looking and no more
# truthful, which is the failure being corrected here.
PRICE_INTERVAL_TARGET = 0.80


def calibrate_price_interval(
    regressor: Any,
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    calibrate_on: str = "val",
    target: float = PRICE_INTERVAL_TARGET,
) -> dict[str, Any]:
    """Derive the served price interval from measured residuals.

    The multipliers are the empirical quantiles of ``actual / predicted`` on
    VAL — the split that exists for choosing serving-time quantities — and the
    coverage they achieve is then reported once on TEST. Choosing them on test
    would make the reported coverage the same in-sample number the old
    threshold tuning produced.

    This replaces a hardcoded +/-15%, which was not derived from anything and
    contained the true price 32% of the time while being presented to users as
    a price range.

    ``calibrate_on`` names a key of ``splits`` and both selects the data and
    labels the artefact, so the two cannot disagree. The label used to be the
    string literal "val" written next to a separate hardcoded ``X_val``
    argument: calibrating on test while still recording "val" was a one-word
    edit away, and the test guarding it asserted the literal against itself.
    Passing the splits as a mapping makes that mislabel unrepresentable rather
    than merely tested for.
    """
    if calibrate_on not in splits:
        raise KeyError(f"calibrate_on={calibrate_on!r} is not one of {sorted(splits)}")

    # Refuse to fit the served interval on the split its coverage is reported
    # against. Labelling is now honest either way, but honesty about a leak is
    # not a substitute for not shipping one: `coverage_test` is advertised as
    # an out-of-sample number, and calibrating here would make it in-sample by
    # construction. The call site was one word away from that, and no test
    # could catch it -- CI never retrains, so the artefact-reading gate only
    # notices after someone regenerates the file and commits it.
    if calibrate_on == "test":
        raise ValueError(
            "refusing to calibrate the served interval on the test split: "
            "coverage_test is reported as out-of-sample evidence"
        )

    lo_q, hi_q = (1.0 - target) / 2.0, 1.0 - (1.0 - target) / 2.0

    def ratio(X: pd.DataFrame, y_log: pd.Series) -> np.ndarray:
        predicted = np.expm1(np.asarray(regressor.predict(X), dtype=float))
        actual = np.expm1(np.asarray(y_log, dtype=float))
        return actual / predicted

    ratios = {name: ratio(X, y) for name, (X, y) in splits.items()}
    low, high = (float(q) for q in np.quantile(ratios[calibrate_on], [lo_q, hi_q]))

    def coverage(r: np.ndarray) -> float:
        return float(np.mean((r >= low) & (r <= high)))

    record = {
        "target_coverage": target,
        "low_multiplier": round(low, 4),
        "high_multiplier": round(high, 4),
        "calibrated_on": calibrate_on,
        **{f"coverage_{name}": round(coverage(r), 4) for name, r in ratios.items()},
    }
    logger.info("Price interval: %s", record)
    return record


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_working_tree_clean() -> bool | None:
    """True when `git status --porcelain` is empty — part of provenance.

    A metrics artefact generated from a dirty tree cannot be tied to its
    commit_sha's source, so the flag is recorded rather than assumed.

    MUST be sampled before the run writes anything. This training script
    produces models/, output/cleaned_house_dataset.csv and price_interval.json,
    so calling it at write time inspects a tree the script has already dirtied
    and the field records ``false`` on every run, including runs that started
    from a pristine checkout. A provenance flag that can never read ``true``
    reports nothing.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() == ""
    except Exception:
        return None


def _write_training_metrics(
    tree_clean_at_start: bool | None,
    clf_record: dict[str, Any],
    reg_record: dict[str, Any],
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    features: list[str],
) -> None:
    """Write ``reports/training_metrics.json`` — the committed artefact the
    README's headline numbers must quote.

    ``tree_clean_at_start`` is passed in rather than sampled here: see
    ``_git_working_tree_clean``. By the time this runs the script has already
    written the models and the cleaned dataset."""
    artefact = {
        "run_date": _dt.datetime.now(_dt.UTC).isoformat(),
        "commit_sha": _git_commit_sha(),
        "working_tree_clean": tree_clean_at_start,
        "provenance": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            # Every headline metric below is scored on the n_test rows by a
            # model chosen on the n_val rows. Nothing was fitted or selected
            # against the test labels.
            "selection_split": "val",
            "reported_split": "test",
            "features": features,
        },
        "classification": clf_record,
        "regression": reg_record,
        "note": (
            "Reproducible from a clean clone: the raw Kaggle CSV "
            "(Resources/NY-House-Dataset.csv) and every model artefact are "
            "committed, so `python run_training.py` regenerates this file and "
            "the models it describes."
        ),
    }
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "training_metrics.json"
    out_path.write_text(
        json.dumps(artefact, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info("Wrote training metrics artefact to %s", out_path)


def main() -> None:
    """Run the full training pipeline."""
    logger.info("NYC PRICE PREDICTION — TRAINING PIPELINE")

    # Sampled first: prepare_data writes the cleaned dataset, so anything
    # after this point sees a tree this script dirtied.
    tree_clean_at_start = _git_working_tree_clean()

    # 1. Prepare data
    df, y_zone, y_log_price, borough = prepare_data()
    X = get_feature_df(df)

    # 2. Encode zone labels
    le = LabelEncoder()
    y_zone_encoded = le.fit_transform(y_zone)

    # Save label encoder
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    # zone_classes is the ONLY valid name order for encoded class indices
    # (alphabetical, from LabelEncoder) — every report/threshold keyed by
    # class index below must use it, never the semantic config order.
    zone_classes = [str(c) for c in le.classes_]
    logger.info("Zone classes: %s", zone_classes)

    # 3. Train/val/test split (stratified for classification).
    #
    # Two splits, not one. TEST is cut first and then not read again until the
    # final scoring call; VAL is cut from what remains and absorbs every
    # decision the pipeline makes (which candidate wins, when DL stops). A
    # single train/test split forced selection to read the test labels, which
    # is what made the previously published 0.724 an in-sample figure.
    (
        X_trainval,
        X_test,
        y_zone_trainval,
        y_zone_test,
        y_price_trainval,
        y_price_test,
        borough_trainval,
        borough_test,
    ) = train_test_split(
        X,
        y_zone_encoded,
        y_log_price,
        borough,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_zone_encoded,
    )
    (
        X_train,
        X_val,
        y_zone_train,
        y_zone_val,
        y_price_train,
        y_price_val,
        borough_train,
        borough_val,
    ) = train_test_split(
        X_trainval,
        y_zone_trainval,
        y_price_trainval,
        borough_trainval,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_zone_trainval,
    )

    logger.info(
        "Train: %d samples, Val: %d samples, Test: %d samples",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    logger.info("Features: %s", list(X_train.columns))

    # 4. Train classification
    clf_record = train_classification(
        X_train,
        y_zone_train,
        X_val,
        y_zone_val,
        X_test,
        y_zone_test,
        borough_test,
        zone_classes,
    )

    # 5. Train regression
    reg_record = train_regression(
        X_train, y_price_train, X_val, y_price_val, X_test, y_price_test
    )

    # 5b. Calibrate the served price interval on val, report coverage on test.
    # Committed as a serving artefact rather than hardcoded so it cannot rot
    # away from the model it describes: a retrain that shifts the residuals
    # rewrites this file, and tests/test_price_interval.py reads the recorded
    # coverage. Constants in source would have to be updated by hand.
    interval = calibrate_price_interval(
        joblib.load(MODELS_DIR / "price_regressor_best.joblib"),
        splits={"val": (X_val, y_price_val), "test": (X_test, y_price_test)},
        calibrate_on="val",
    )
    (MODELS_DIR / "price_interval.json").write_text(
        json.dumps(interval, indent=2) + "\n", encoding="utf-8"
    )
    reg_record["price_interval"] = interval

    # 6. Generate SHAP explanations
    logger.info("STEP 5: SHAP explainability")
    try:
        best_clf = joblib.load(MODELS_DIR / "price_zone_best.joblib")
        # Get preprocessed features for SHAP
        preprocessor = best_clf.named_steps["preprocessor"]
        X_test_transformed = preprocessor.transform(X_test)
        feature_names = list(preprocessor.get_feature_names_out())

        from src.models.explain import compute_shap_values, global_feature_importance

        classifier_step = best_clf.named_steps["classifier"]
        shap_values, explainer = compute_shap_values(
            classifier_step,
            pd.DataFrame(X_test_transformed, columns=feature_names),
            max_samples=200,
        )
        importance_df = global_feature_importance(shap_values, feature_names)
        logger.info("Top 10 features by SHAP:\n%s", importance_df.head(10).to_string())
        clf_record["shap_top10"] = importance_df.head(10).to_dict("records")
    except Exception as exc:
        logger.warning("SHAP analysis failed (non-critical): %s", exc)

    # 7. Persist the committed evidence artefact behind the README numbers.
    #
    # There is no threshold-tuning step here any more. It used to fit
    # per-class thresholds against the TEST labels and publish the resulting
    # macro-F1 as a hold-out result — the +0.014 "gain" was the tuner reading
    # its own answer sheet. Measured honestly (thresholds fitted on one half
    # of the test set, scored on the other, 20 stratified splits) the effect
    # is +0.0006 mean with std 0.0106, helping 12 splits and hurting 8: noise.
    # Serving decodes with argmax.
    _write_training_metrics(
        tree_clean_at_start,
        clf_record,
        reg_record,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        features=list(X_train.columns),
    )

    # 8. Deep learning training
    logger.info("STEP 7: Deep learning (multi-task)")
    try:
        from src.dl.tabular_net import MultiTaskDenseNet, MultiTaskLoss
        from src.dl.train_dl import prepare_dl_data, train_multitask

        best_clf = joblib.load(MODELS_DIR / "price_zone_best.joblib")
        preprocessor = best_clf.named_steps["preprocessor"]

        # Transform features
        X_train_t = preprocessor.transform(X_train)
        X_val_t = preprocessor.transform(X_val)
        X_test_t = preprocessor.transform(X_test)
        n_features = X_train_t.shape[1]

        import torch

        # Build model — all numeric (no separate categorical embeddings in transformed space)
        model = MultiTaskDenseNet(
            n_numeric=n_features,
            categorical_dims=[],
            num_classes=len(zone_classes),
            hidden_dims=[256, 128, 64],
            dropout=0.3,
        )

        # Prepare data loaders
        train_loader = prepare_dl_data(
            X_train_t,
            [],
            y_zone_train,
            y_price_train.values,
            batch_size=256,
        )
        # Early stopping reads this loader every epoch, so it must not be the
        # test set: `patience=15` on test labels is model selection on test,
        # dressed as regularisation.
        val_loader = prepare_dl_data(
            X_val_t,
            [],
            y_zone_val,
            y_price_val.values,
            batch_size=256,
            shuffle=False,
        )

        # Focal loss with class weights
        from collections import Counter

        counts = Counter(y_zone_train)
        total = sum(counts.values())
        class_weights = [
            total / (len(counts) * counts.get(i, 1)) for i in range(len(zone_classes))
        ]
        loss_fn = MultiTaskLoss(alpha=0.6, focal_gamma=2.0, class_weights=class_weights)

        train_multitask(
            model,
            loss_fn,
            train_loader,
            val_loader,
            n_categorical=0,
            epochs=80,
            lr=1e-3,
            patience=15,
        )

        # Evaluate DL model
        model.eval()
        with torch.no_grad():
            x_test_tensor = torch.tensor(X_test_t, dtype=torch.float32)
            cls_logits, reg_pred = model(x_test_tensor, [])
            dl_cls_pred = cls_logits.argmax(dim=1).numpy()
            dl_reg_pred = reg_pred.numpy()

        from src.models.evaluate import evaluate_classifier, evaluate_regressor

        dl_cls_metrics = evaluate_classifier(y_zone_test, dl_cls_pred, zone_classes)
        dl_reg_metrics = evaluate_regressor(
            y_price_test.values, dl_reg_pred, log_target=True
        )
        logger.info(
            "DL Classification: accuracy=%.4f, macro_f1=%.4f",
            dl_cls_metrics["accuracy"],
            dl_cls_metrics["macro_f1"],
        )
        logger.info(
            "DL Regression: R2=%.4f, RMSE=%.4f",
            dl_reg_metrics["r2"],
            dl_reg_metrics["rmse"],
        )

        if _HAS_MLFLOW:
            mlflow.set_experiment("price_zone_classification")
            with mlflow.start_run(run_name="dl_multitask"):
                mlflow.log_metrics(
                    {
                        "accuracy": dl_cls_metrics["accuracy"],
                        "macro_f1": dl_cls_metrics["macro_f1"],
                        "reg_r2": dl_reg_metrics["r2"],
                    }
                )
    except Exception as exc:
        logger.warning("DL training failed (non-critical): %s", exc)

    # 9. Save drift baseline
    logger.info("STEP 8: Drift baseline")
    try:
        from src.models.drift import save_baseline

        save_baseline(X_train, MODELS_DIR / "drift_baseline.json")
        logger.info("Drift baseline saved")
    except Exception as exc:
        logger.warning("Drift baseline failed (non-critical): %s", exc)

    logger.info("TRAINING COMPLETE")
    logger.info("Models saved to: %s", MODELS_DIR)


if __name__ == "__main__":
    main()
