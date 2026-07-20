"""End-to-end training orchestrator — load data, engineer features, train models, save artifacts.

Besides the model artefacts (committed and MANIFEST-pinned), every run
writes ``reports/training_metrics.json`` — the committed evidence artefact
behind the README's headline numbers. It records the metrics of the
selected models together with provenance (commit SHA, library versions,
seed, split sizes) so the README quotes a file, not a memory.
"""

from __future__ import annotations

import bisect
import datetime as _dt
import hashlib
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

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
    PRICE_ZONE_LABELS,
    RANDOM_SEED,
    TARGET_ENCODED_FEATURES,
    TEST_SIZE,
    VAL_SIZE,
)
from src.data.cleaner import apply_cap, clean_pipeline, fit_cap_bounds
from src.data.features import (
    add_numeric_features,
    add_target_variables,
    apply_top_categories,
    fit_top_categories,
)
from src.data.loader import load_raw
from src.models.evaluate import (
    evaluate_classifier,
    evaluate_fairness_by_group,
    evaluate_regressor,
)
from src.models.pipelines import (
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


def prepare_data() -> pd.DataFrame:
    """Load and clean the raw snapshot; write the cleaned (uncapped) CSV.

    Everything cross-row — cap bounds, zone cut-points, category vocabulary —
    is fitted later, on the train split only, inside :func:`run_protocol`.
    """
    logger.info("Step 1: loading raw data and cleaning it")
    df = clean_pipeline(load_raw())

    # Written for the benchmark trainer and the EDA notebook, which both read
    # the cleaned CSV. It is an output of this script, never an input to it.
    CLEANED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATASET, index=False)
    logger.info("Wrote cleaned dataset to %s", CLEANED_DATASET)
    return df


def _borough_median_baseline(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    zone_bins: list[float],
) -> dict[str, Any]:
    """The naive predictor every headline number must beat: each borough's
    median train price, bucketed through the same zone decode."""
    medians = train_frame.groupby("BOROUGH")["LOG_PRICE"].median()
    fallback = float(train_frame["LOG_PRICE"].median())
    pred_log = test_frame["BOROUGH"].map(medians).fillna(fallback).to_numpy(dtype=float)
    reg = evaluate_regressor(
        test_frame["LOG_PRICE"].to_numpy(dtype=float), pred_log, log_target=True
    )
    zone_pred = pd.cut(
        np.expm1(pred_log),
        bins=zone_bins,
        labels=PRICE_ZONE_LABELS,
        include_lowest=True,
    ).astype(str)
    zones = evaluate_classifier(
        test_frame["PRICE_ZONE"].astype(str).to_numpy(), np.asarray(zone_pred)
    )
    return {
        "predictor": "per-borough median train price",
        "test_r2": round(float(reg["r2"]), 4),
        "test_zones_macro_f1": round(float(zones["macro_f1"]), 4),
    }


def build_splits(df_clean: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Deterministic data preparation for one seed, train-split-fitted.

    Split FIRST, then fit every cross-row statistic (cap bounds, zone
    cut-points, category vocabulary) on the train rows only and apply it
    everywhere. The split stratifies on pooled price quartiles purely as a
    balancing key — the served zone labels come from train-derived cut-points.
    """
    df = df_clean.reset_index(drop=True)

    strat_key = pd.qcut(df["PRICE"], 4, labels=False, duplicates="drop")
    idx_trainval, idx_test = train_test_split(
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=strat_key,
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=VAL_SIZE,
        random_state=seed,
        stratify=strat_key[idx_trainval],
    )

    bounds = fit_cap_bounds(df.loc[idx_train])
    df = apply_cap(df, bounds)

    df = add_numeric_features(df)
    df = add_distance_features(df, REFERENCE_POINTS)

    # Zone cut-points: quartiles of the TRAIN prices (post-cap), so labels for
    # every split derive from a statistic the held-out rows never touched.
    train_prices = df.loc[idx_train, "PRICE"]
    zone_bins = [
        0.0,
        float(train_prices.quantile(0.25)),
        float(train_prices.quantile(0.50)),
        float(train_prices.quantile(0.75)),
        float("inf"),
    ]
    df = add_target_variables(df, bins=zone_bins)

    top = fit_top_categories(
        df.loc[idx_train], columns=["SUBLOCALITY", "TYPE", "ZIPCODE"]
    )
    df = apply_top_categories(df, top)

    for col in ONEHOT_FEATURES + TARGET_ENCODED_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    X = get_feature_df(df)
    y_zone = df["PRICE_ZONE"].astype(str).to_numpy()
    y_log = df["LOG_PRICE"]

    return {
        "df": df,
        "idx": {"train": idx_train, "val": idx_val, "test": idx_test},
        "splits": {
            "train": (X.loc[idx_train], y_log.loc[idx_train]),
            "val": (X.loc[idx_val], y_log.loc[idx_val]),
            "test": (X.loc[idx_test], y_log.loc[idx_test]),
        },
        "y_zone": y_zone,
        "zone_bins": zone_bins,
        "cap_bounds": {k: list(v) for k, v in bounds.items()},
        "category_vocabulary": {k: sorted(v) for k, v in top.items()},
        "features": list(X.columns),
    }


def run_protocol(
    df_clean: pd.DataFrame,
    seed: int,
    save_path: Path | None = None,
) -> dict[str, Any]:
    """The full training protocol for one seed: build_splits, then candidate
    selection on val and a single test read."""
    prep = build_splits(df_clean, seed)
    df = prep["df"]
    idx_train = prep["idx"]["train"]
    idx_val = prep["idx"]["val"]
    idx_test = prep["idx"]["test"]
    splits = prep["splits"]
    y_zone = prep["y_zone"]
    zone_bins = prep["zone_bins"]
    logger.info(
        "Train: %d, Val: %d, Test: %d", len(idx_train), len(idx_val), len(idx_test)
    )

    reg_record, best_pipeline = train_regression(
        splits["train"][0],
        splits["train"][1].to_numpy(),
        splits["val"][0],
        splits["val"][1].to_numpy(),
        splits["test"][0],
        splits["test"][1].to_numpy(),
        seed=seed,
        save_path=save_path,
    )

    # Zones the service will return: the regressor's test predictions bucketed
    # through the same cut-points that labelled the training data.
    predicted_prices = np.expm1(
        np.asarray(best_pipeline.predict(splits["test"][0]), dtype=float)
    )
    interior = zone_bins[1:-1]
    zone_pred = np.array(
        [PRICE_ZONE_LABELS[bisect.bisect_left(interior, p)] for p in predicted_prices]
    )
    y_zone_test = y_zone[idx_test]
    borough_test = df.loc[idx_test, "BOROUGH"]

    clf_record: dict[str, Any] = {
        "derived_from": "regressor predictions bucketed by PRICE_ZONE_BINS",
        "metrics": evaluate_classifier(y_zone_test, zone_pred, PRICE_ZONE_LABELS),
        "fairness_by_borough": evaluate_fairness_by_group(
            y_zone_test, zone_pred, borough_test
        ),
        "borough_floor": check_borough_floor(y_zone_test, zone_pred, borough_test),
    }

    baseline = _borough_median_baseline(
        df.loc[idx_train, ["BOROUGH", "LOG_PRICE"]],
        df.loc[idx_test, ["BOROUGH", "LOG_PRICE", "PRICE_ZONE"]],
        zone_bins,
    )

    return {
        "reg_record": reg_record,
        "clf_record": clf_record,
        "baseline": baseline,
        "zone_bins": zone_bins,
        "cap_bounds": prep["cap_bounds"],
        "splits": splits,
        "best_pipeline": best_pipeline,
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
        "features": prep["features"],
    }


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


def train_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int = RANDOM_SEED,
    save_path: Path | None = None,
) -> tuple[dict[str, Any], Any]:
    """Train candidates, pick the winner on VAL, score it once on TEST.

    Three candidates compared on R2 is three draws; picking the max of those
    on test would report the luckiest draw as a hold-out estimate.

    Returns the selected model's record (name, metrics) for the committed
    training-metrics artefact.
    """
    logger.info("STEP 4: Training regression models")

    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    models = {
        # min_samples_leaf=10 both bounds the artifact under GitHub's 100 MB
        # limit and gives each leaf a mean over >=10 comparable sales; unbounded,
        # the 500 trees average one leaf per sample and produce a 129 MB file.
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=10,
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            max_depth=6,
            n_estimators=500,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(
            num_leaves=63,
            n_estimators=500,
            learning_rate=0.1,
            random_state=seed,
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
        if save_path is not None:
            joblib.dump(best_pipeline, save_path)
            logger.info("Saved best regressor (%s) to %s", best_name, save_path)

    return {
        "selected_model": best_name,
        "metrics": best_metrics,
        "selection_metrics_val": best_val_metrics,
        "candidates_val": candidates,
    }, best_pipeline


# Fraction of listings the served price interval is calibrated to contain.
# 0.80 rather than a tighter target because this model's honest spread is
# already wide (see the multipliers written into the artefact): a narrower
# nominal target would produce an interval that is precise-looking and no more
# truthful, which is the failure being corrected here.
PRICE_INTERVAL_TARGET = 0.80


class BoroughFloorError(RuntimeError):
    """A borough scored at or below its own majority-class baseline."""


def check_borough_floor(
    y_true: np.ndarray, y_pred: np.ndarray, borough: pd.Series
) -> dict[str, Any]:
    """Every borough must beat guessing its own most common zone.

    "Predicts across the 5 boroughs" is a contract, not an average. A model
    with a strong city-wide macro-F1 that fails in Queens is broken for
    everyone in Queens, and averaging is what hides it.

    The floor is derived per borough: score the constant predictor that always
    answers that borough's most common zone. A model that cannot beat that has
    learned nothing there. Deriving it is deliberate -- any fixed threshold
    would be a number chosen with the current results already in view, which is
    the defect the standard calls out by name.

    Raises so a failing borough fails the run, and therefore the build, rather
    than being written into the metrics file for someone to notice later.
    """
    frame = pd.DataFrame(
        {
            "true": np.asarray(y_true),
            "pred": np.asarray(y_pred),
            "borough": borough.to_numpy(),
        }
    )
    rows: dict[str, Any] = {}
    breaches: list[str] = []

    for name, group in frame.groupby("borough", sort=True):
        majority = group["true"].value_counts().idxmax()
        baseline = f1_score(
            group["true"],
            np.full(len(group), majority),
            average="macro",
            zero_division=0,
        )
        actual = f1_score(
            group["true"], group["pred"], average="macro", zero_division=0
        )
        rows[str(name)] = {
            "macro_f1": round(float(actual), 4),
            "majority_baseline": round(float(baseline), 4),
            "margin": round(float(actual - baseline), 4),
            "n": int(len(group)),
        }
        if actual <= baseline:
            breaches.append(
                f"{name}: macro_f1={actual:.4f} <= majority baseline={baseline:.4f}"
            )

    if breaches:
        raise BoroughFloorError(
            "borough floor breached, refusing to ship: " + "; ".join(breaches)
        )

    logger.info("Borough floor: all %d boroughs clear their baseline", len(rows))
    return rows


def calibrate_price_interval(
    regressor: Any,
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    calibrate_on: str = "val",
    target: float = PRICE_INTERVAL_TARGET,
) -> dict[str, Any]:
    """Derive the served price interval from measured residuals.

    The multipliers are the empirical quantiles of ``actual / predicted`` on the
    ``calibrate_on`` split; coverage is reported once on every split. That key
    both selects the data and labels the artefact, so the label cannot disagree
    with the data quantiled. The guard is name-based — it does not inspect the
    data to tell splits apart.
    """
    if calibrate_on not in splits:
        raise KeyError(f"calibrate_on={calibrate_on!r} is not one of {sorted(splits)}")

    # coverage_test is reported as out-of-sample, so refuse to calibrate on it.
    if calibrate_on == "test":
        raise ValueError(
            "refusing to calibrate the served interval on the test split: "
            "coverage_test is reported as out-of-sample evidence"
        )

    hi_q = 1.0 - (1.0 - target) / 2.0

    def ratio(X: pd.DataFrame, y_log: pd.Series) -> np.ndarray:
        predicted = np.expm1(np.asarray(regressor.predict(X), dtype=float))
        actual = np.expm1(np.asarray(y_log, dtype=float))
        return actual / predicted

    ratios = {name: ratio(X, y) for name, (X, y) in splits.items()}
    # Split-conformal finite-sample correction: the ceil((n+1)(1-alpha))-th
    # order statistic, which covers a fresh draw where the plain empirical
    # quantile under-covers by construction.
    n_cal = len(ratios[calibrate_on])
    corrected_hi = min(math.ceil((n_cal + 1) * hi_q) / n_cal, 1.0)
    corrected_lo = max(1.0 - corrected_hi, 0.0)
    low, high = (
        float(q)
        for q in np.quantile(ratios[calibrate_on], [corrected_lo, corrected_hi])
    )

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
    zone_bins: list[float],
    baseline: dict[str, Any],
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
            # The cut-points the zone labels were built from — derived from
            # the TRAIN prices of this run. test_config_artefact_agreement
            # fails the build if config drifts from these.
            "price_zone_bins": [b for b in zone_bins if b != float("inf")],
        },
        "baseline": baseline,
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

    # 1-4. Clean, then run the train-split-fitted protocol once with the
    # shipped seed. A dedicated classifier was measured ~1.3 SE above the
    # bucketed zones and is not shipped: two models can disagree on one
    # response with nothing to catch it.
    df_clean = prepare_data()
    result = run_protocol(
        df_clean,
        seed=RANDOM_SEED,
        save_path=MODELS_DIR / "price_regressor_best.joblib",
    )
    reg_record = result["reg_record"]
    clf_record = result["clf_record"]
    best_reg = result["best_pipeline"]
    X_val, y_price_val = result["splits"]["val"]
    X_test, y_price_test = result["splits"]["test"]
    X_train = result["splits"]["train"][0]

    # 5b. Calibrate the served price interval on val, report coverage on test.
    # Committed as an artefact rather than hardcoded so a retrain that shifts
    # the residuals rewrites it in step with the model.
    interval = calibrate_price_interval(
        best_reg,
        splits={"val": (X_val, y_price_val), "test": (X_test, y_price_test)},
        calibrate_on="val",
    )
    (MODELS_DIR / "price_interval.json").write_text(
        json.dumps(interval, indent=2) + "\n", encoding="utf-8"
    )
    reg_record["price_interval"] = interval

    logger.info(
        "Zones (test, bucketed): accuracy=%.4f macro_f1=%.4f | baseline: %s",
        clf_record["metrics"]["accuracy"],
        clf_record["metrics"]["macro_f1"],
        result["baseline"],
    )

    # 6. SHAP over the single model.
    logger.info("STEP 5: SHAP explainability")
    try:
        preprocessor = best_reg.named_steps["preprocessor"]
        X_test_transformed = preprocessor.transform(X_test)
        feature_names = list(preprocessor.get_feature_names_out())

        from src.models.explain import compute_shap_values, global_feature_importance

        shap_values, _explainer = compute_shap_values(
            best_reg.named_steps["regressor"],
            pd.DataFrame(X_test_transformed, columns=feature_names),
            max_samples=200,
        )
        importance_df = global_feature_importance(shap_values, feature_names)
        logger.info("Top 10 SHAP features:\n%s", importance_df.head(10).to_string())
        clf_record["shap_top10"] = importance_df.head(10).to_dict("records")
    except Exception as exc:
        logger.warning("SHAP analysis failed (non-critical): %s", exc)

    # 7. Persist the committed evidence artefact behind the README numbers.
    _write_training_metrics(
        tree_clean_at_start,
        clf_record,
        reg_record,
        n_train=result["n_train"],
        n_val=result["n_val"],
        n_test=result["n_test"],
        features=result["features"],
        zone_bins=result["zone_bins"],
        baseline=result["baseline"],
    )

    # 9. Save drift baseline
    logger.info("STEP 8: Drift baseline")
    try:
        from src.models.drift import save_baseline

        save_baseline(X_train, MODELS_DIR / "drift_baseline.json")
        logger.info("Drift baseline saved")
    except Exception as exc:
        logger.warning("Drift baseline failed (non-critical): %s", exc)

    # 10. Regenerate the artefact manifest LAST, over the files this run just
    # wrote, so the committed hashes always have a producer. JSON is hashed
    # LF-normalised, matching tests/test_artifact_manifest.py.
    logger.info("STEP 9: Artefact manifest")
    manifest_lines = []
    for name in sorted(
        (
            "benchmark_regressor.joblib",
            "price_regressor_best.joblib",
            "drift_baseline.json",
            "price_interval.json",
        )
    ):
        data = (MODELS_DIR / name).read_bytes()
        if name.endswith(".json"):
            data = data.replace(b"\r\n", b"\n")
        manifest_lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    (MODELS_DIR / "MANIFEST.sha256").write_bytes(
        ("\n".join(manifest_lines) + "\n").encode("ascii")
    )
    logger.info("Manifest written for %d artefacts", len(manifest_lines))

    logger.info("TRAINING COMPLETE")
    logger.info("Models saved to: %s", MODELS_DIR)


if __name__ == "__main__":
    main()
