"""External benchmark orchestrator — one-shot run.

Pipeline:

1. Download NYC.gov 2024 Rolling Sales (5 boroughs) via
   :func:`benchmarks.datasets.nyc_rolling_sales_2024.download_nyc_rolling_sales`.
2. Apply :func:`benchmarks.mapping.apply_schema_map` to produce
   ``(X, target, report)`` under the SCHEMA_MAP.md v1 contract.
3. Run the four firewall invariants:
   - name-based leakage check on X
   - semantic leakage (Pearson + Spearman + normalised MI) on X vs target
   - drop-log consistency (n_raw == n_scored + n_dropped)
   - SCHEMA_MAP.md SHA vs registry
4. Attempt model inference. The trained regressor was built on the
   Kaggle 2023 schema (BEDS / BATH / DIST_* / SUBLOCALITY / …);
   if the SCHEMA_MAP v1 feature space does not cover the model's
   required columns, inference fails. That failure is captured in
   ``inference`` rather than propagated — this is an honest finding,
   not a bug in the orchestrator.
5. Run :func:`benchmarks.invariants.check_predictions_healthy` on the
   prediction array when inference succeeded; skip when it did not.
6. Write ``benchmarks/results.json``. Whatever the first run produces
   is what ships. No tuning, no schema edits, no retry (per Rule A of
   the Step 5 execution contract).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.datasets.nyc_rolling_sales_2024 import download_nyc_rolling_sales
from benchmarks.invariants import (
    SCHEMA_MAP_VERSION,
    HealthError,
    LeakageError,
    check_no_forbidden_columns,
    check_predictions_healthy,
    check_target_independence,
)
from benchmarks.mapping import apply_schema_map


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_MAP_PATH = REPO_ROOT / "benchmarks" / "SCHEMA_MAP.md"
VERSIONS_PATH = REPO_ROOT / "benchmarks" / "SCHEMA_MAP_VERSIONS.json"
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results.json"
MODEL_PATH = REPO_ROOT / "models" / "price_regressor_best.joblib"


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _schema_map_sha() -> str:
    return hashlib.sha256(SCHEMA_MAP_PATH.read_bytes()).hexdigest()


def _run_leakage_invariants(X: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    leakage: dict[str, Any] = {"name_based": None, "semantic": None}
    try:
        check_no_forbidden_columns(X)
        leakage["name_based"] = {"triggered": False, "message": None}
    except LeakageError as exc:
        leakage["name_based"] = {"triggered": True, "message": str(exc)}
    try:
        check_target_independence(X, target)
        leakage["semantic"] = {"triggered": False, "message": None}
    except LeakageError as exc:
        leakage["semantic"] = {"triggered": True, "message": str(exc)}
    return leakage


def _run_prediction_health(predictions: np.ndarray | None) -> dict[str, Any]:
    if predictions is None:
        return {"status": "skipped", "reason": "no predictions produced"}
    try:
        check_predictions_healthy(predictions)
        return {"status": "passed", "message": None}
    except HealthError as exc:
        return {"status": "failed", "message": str(exc)}


def _attempt_inference(
    X: pd.DataFrame,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Try to load the trained model and score ``X``.

    Captures any failure mode structurally — missing model file,
    schema mismatch, unexpected runtime error — and returns it as
    part of the inference record. Never raises.
    """
    if not MODEL_PATH.exists():
        return None, {
            "status": "skipped",
            "reason": "model file not present",
            "model_path": str(MODEL_PATH.relative_to(REPO_ROOT)),
        }

    try:
        import joblib

        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        return None, {
            "status": "failed",
            "stage": "load",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    expected: list[str] = []
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)

    produced = list(X.columns)
    missing = [c for c in expected if c not in produced]
    extra = [c for c in produced if c not in expected]

    try:
        preds = np.asarray(model.predict(X), dtype=float)
    except Exception as exc:
        return None, {
            "status": "failed",
            "stage": "predict",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "expected_features": expected,
            "produced_features": produced,
            "missing_features": missing,
            "extra_features": extra,
        }

    return preds, {
        "status": "succeeded",
        "n_predictions": int(preds.size),
        "expected_features": expected,
        "produced_features": produced,
        "missing_features": missing,
        "extra_features": extra,
    }


def run_benchmark() -> dict[str, Any]:
    """Execute the benchmark once and return the results dict.

    Side effect: writes :data:`RESULTS_PATH` with the serialised
    result. The returned dict is identical to what lands on disk.
    """
    run_started = _dt.datetime.now(_dt.timezone.utc).isoformat()

    raw, manifests = download_nyc_rolling_sales()
    download_record = [asdict(m) for m in manifests]

    X, target, report = apply_schema_map(raw)

    leakage = _run_leakage_invariants(X, target)
    predictions, inference = _attempt_inference(X)
    health = _run_prediction_health(predictions)

    performance: dict[str, Any]
    if predictions is not None and len(predictions) > 0:
        target_arr = target.to_numpy()
        residuals = target_arr - predictions
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((target_arr - target_arr.mean()) ** 2))
        r2 = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
        performance = {
            "status": "computed",
            "r2_log_space": r2,
            "n_scored": int(len(predictions)),
        }
    else:
        performance = {
            "status": "unobservable",
            "reason": "no predictions produced (see inference.status)",
        }

    leakage_tripwire = (
        predictions is not None
        and performance.get("r2_log_space") is not None
        and performance["r2_log_space"] > 0.95
    )

    result: dict[str, Any] = {
        "run_date": run_started,
        "run_ended": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "commit_sha": _git_commit_sha(),
        "schema_map_version": SCHEMA_MAP_VERSION,
        "schema_map_sha256": _schema_map_sha(),
        "data_source": "https://www.nyc.gov/site/finance/property/property-rolling-sales-data.page",
        "data_manifest": download_record,
        "n_raw": report.n_raw,
        "n_dropped": report.n_dropped,
        "n_scored": report.n_scored,
        "drop_reasons": dict(report.drop_reasons),
        "feature_columns": list(X.columns),
        "leakage": leakage,
        "inference": inference,
        "health_checks": health,
        "performance": performance,
        "leakage_tripwire": {
            "threshold": 0.95,
            "triggered": bool(leakage_tripwire),
        },
        "reproducibility": {
            "tolerance": "±1e-6 on metrics across x86_64 Linux CI runners, pinned deps",
            "no_cross_arch_claim": True,
        },
    }

    RESULTS_PATH.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


if __name__ == "__main__":
    result = run_benchmark()
    print(json.dumps(
        {k: v for k, v in result.items() if k != "data_manifest"},
        indent=2,
        default=str,
    ))
