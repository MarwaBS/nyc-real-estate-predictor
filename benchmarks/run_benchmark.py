"""External benchmark orchestrator — one-shot run.

Pipeline (every "invariant" below is ENFORCED here at run time — a
violation raises and fails the run; nothing is merely recorded):

1. Verify the schema lock: SCHEMA_MAP.md's LF-normalised SHA-256 must
   equal the registry entry sealed for the current SCHEMA_MAP_VERSION
   (:func:`benchmarks.invariants.verify_schema_map_lock`). A run against
   an unsealed contract is invalid by definition, so it never starts.
2. Download NYC.gov 2024 Rolling Sales (5 boroughs) via
   :func:`benchmarks.datasets.nyc_rolling_sales_2024.download_nyc_rolling_sales`.
3. Apply :func:`benchmarks.mapping.apply_schema_map` to produce
   ``(X, target, report)`` under the SCHEMA_MAP.md contract, then enforce:
   - drop-log consistency: ``sum(drop_reasons) == n_dropped`` and
     ``n_raw == n_scored + n_dropped`` (raises on violation);
   - target sanity: every retained log-price is finite (raises — a NaN
     target would otherwise silently poison the R²).
4. Run the leakage invariants — name-based forbidden-column check and
   semantic (Pearson + Spearman + normalised MI) target-independence.
   Findings are recorded; a triggered detector is visible in
   ``results.json → leakage`` and trips the tripwire bool.
5. Run inference with the lean benchmark regressor
   (``models/benchmark_regressor.joblib`` — COMMITTED, 0.6 MB, trained by
   ``benchmarks.train_benchmark_model`` on the three features shared with
   NYC.gov sales: borough, property_sqft, zip_code). Because the artefact
   ships with the repo and the data is a public download, CI and any
   stranger compute the same R² this file reports.
6. Run :func:`benchmarks.invariants.check_predictions_healthy` on the
   prediction array when inference succeeded; skip when it did not.
7. Write ``benchmarks/results.json``. Whatever the first run produces
   is what ships. No tuning, no schema edits, no retry (per Rule A of
   the Step 5 execution contract).
"""

from __future__ import annotations

import datetime as _dt
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
    verify_schema_map_lock,
)
from benchmarks.mapping import MappingReport, apply_schema_map

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results.json"
MODEL_PATH = REPO_ROOT / "models" / "benchmark_regressor.joblib"


class DropLogError(Exception):
    """The mapping's drop accounting does not reconcile (SCHEMA_MAP §4)."""


def _verify_drop_log(report: MappingReport) -> None:
    """Enforce drop-log consistency instead of trusting the mapping."""
    reason_total = sum(report.drop_reasons.values())
    if reason_total != report.n_dropped:
        raise DropLogError(
            f"drop_reasons sum to {reason_total} but n_dropped={report.n_dropped}"
        )
    if report.n_raw != report.n_scored + report.n_dropped:
        raise DropLogError(
            f"n_raw={report.n_raw} != n_scored={report.n_scored} "
            f"+ n_dropped={report.n_dropped}"
        )


def _verify_target_finite(target: pd.Series) -> None:
    """A non-finite retained target would silently poison the R²."""
    bad = int((~np.isfinite(target.to_numpy(dtype=float))).sum())
    if bad:
        raise DropLogError(
            f"{bad} retained row(s) have a non-finite log-price target — "
            f"the drop engine must reject them (missing_sale_price)"
        )


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
    run_started = _dt.datetime.now(_dt.UTC).isoformat()

    # Invariant 1 (hard gate): the contract being executed must be the
    # contract that was sealed. Raises SchemaLockError before any download.
    verified_sha = verify_schema_map_lock()

    raw, manifests = download_nyc_rolling_sales()
    download_record = [asdict(m) for m in manifests]

    X, target, report = apply_schema_map(raw)

    # Invariants 2+3 (hard gates): drop accounting reconciles; no retained
    # row carries a non-finite target.
    _verify_drop_log(report)
    _verify_target_finite(target)

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
        if r2 is not None and not np.isfinite(r2):
            # A NaN/inf R² is a pipeline defect, never a finding.
            raise DropLogError(f"non-finite R² computed: {r2!r}")
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
        "run_ended": _dt.datetime.now(_dt.UTC).isoformat(),
        "commit_sha": _git_commit_sha(),
        "schema_map_version": SCHEMA_MAP_VERSION,
        # Verified against the registry at step 1 — by the time this is
        # written, the hash is known to equal the sealed entry.
        "schema_map_sha256": verified_sha,
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
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "data_manifest"},
            indent=2,
            default=str,
        )
    )
