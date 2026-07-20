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


class InferenceError(Exception):
    """The committed benchmark model failed to load or score.

    Since the model artefact ships WITH the repo (it is what makes the
    benchmark stranger-reproducible), any inference failure — missing
    file, unpicklable artefact (e.g. produced under unpinned numpy),
    feature mismatch — is a structural pipeline defect, never a finding.
    Raising turns the CI benchmark job red instead of recording the
    failure as data behind a green badge.
    """


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


def _run_prediction_health(predictions: np.ndarray) -> dict[str, Any]:
    """Hard gate (benchmark.yml Rule C: "structural health-check failure").

    A collapsed / NaN / Inf prediction array is a pipeline defect, never a
    finding — :class:`HealthError` propagates and fails the run, consistent
    with the lock / drop-log / inference gates. results.json therefore only
    ever records ``passed``: failed runs abort instead of shipping data.
    """
    check_predictions_healthy(predictions)
    return {"status": "passed", "message": None}


def _git_working_tree_clean() -> bool | None:
    """True when `git status --porcelain` is empty — part of provenance.

    An artefact generated from a dirty tree cannot be tied to its
    commit_sha's source, so the flag is recorded rather than assumed.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() == ""
    except Exception:
        return None


def _run_inference(X: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the COMMITTED benchmark model and score ``X``.

    Inference is load-bearing (the artefact ships with the repo), so every
    failure mode raises :class:`InferenceError` and fails the run — a
    missing or unpicklable model behind a green CI badge is exactly the
    silent-failure class this pipeline exists to prevent.
    """
    if not MODEL_PATH.exists():
        raise InferenceError(
            f"committed benchmark model missing: "
            f"{MODEL_PATH.relative_to(REPO_ROOT)} — broken checkout or "
            f"ignored artefact; run benchmarks.train_benchmark_model under "
            f"the PINNED environment (requirements.txt) and commit it"
        )

    try:
        import joblib

        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise InferenceError(
            f"committed benchmark model failed to LOAD "
            f"({type(exc).__name__}: {exc}). Most likely cause: the artefact "
            f"was pickled under an environment that violates the pins in "
            f"requirements.txt (e.g. a different numpy/sklearn). Retrain it "
            f"under the pinned environment and recommit."
        ) from exc

    expected: list[str] = []
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)

    produced = list(X.columns)
    missing = [c for c in expected if c not in produced]
    extra = [c for c in produced if c not in expected]

    try:
        preds = np.asarray(model.predict(X), dtype=float)
    except Exception as exc:
        raise InferenceError(
            f"committed benchmark model failed to PREDICT "
            f"({type(exc).__name__}: {exc}); expected features {expected}, "
            f"mapping produced {produced} (missing={missing}, extra={extra}) "
            f"— the sealed contract and the committed model must agree"
        ) from exc

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
    # Inference is a hard gate: _run_inference raises InferenceError on a
    # missing/unloadable/mismatched committed model — never a green skip.
    predictions, inference = _run_inference(X)
    health = _run_prediction_health(predictions)

    target_arr = target.to_numpy()
    residuals = target_arr - predictions
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((target_arr - target_arr.mean()) ** 2))
    r2 = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    if r2 is not None and not np.isfinite(r2):
        # A NaN/inf R² is a pipeline defect, never a finding.
        raise DropLogError(f"non-finite R² computed: {r2!r}")
    # The committed naive baseline, scored on the SAME rows: per-borough
    # median log-price of the benchmark training frame. Load-bearing — a
    # missing baseline artefact fails the run rather than skipping the
    # comparison the README advertises.
    baseline_spec = json.loads(
        (REPO_ROOT / "models" / "benchmark_baseline.json").read_text("utf-8")
    )
    medians = baseline_spec["borough_median_log_price"]
    fallback = float(baseline_spec["global_median_log_price"])
    baseline_pred = (
        X["borough"].map(medians).fillna(fallback).to_numpy(dtype=float)
    )
    b_res = target_arr - baseline_pred
    baseline_r2 = (
        None if ss_tot == 0 else 1.0 - float(np.sum(b_res**2)) / ss_tot
    )
    performance: dict[str, Any] = {
        "status": "computed",
        "r2_log_space": r2,
        "baseline_r2_log_space": (
            None if baseline_r2 is None else round(baseline_r2, 4)
        ),
        "baseline_predictor": baseline_spec["predictor"],
        "n_scored": int(len(predictions)),
    }

    leakage_tripwire = r2 is not None and r2 > 0.95

    result: dict[str, Any] = {
        "run_date": run_started,
        "run_ended": _dt.datetime.now(_dt.UTC).isoformat(),
        "commit_sha": _git_commit_sha(),
        "working_tree_clean": _git_working_tree_clean(),
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
            # Enforced by mechanism, not an asserted tolerance string.
            "enforced_by": "pinned deps + fixed seed + schema SHA gate",
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

    # Fail the run on a detected leak — AFTER results.json and the summary are
    # written, so the evidence survives the failure.
    tripped = [
        name
        for name, outcome in result["leakage"].items()
        if outcome and outcome.get("triggered")
    ]
    if result["leakage_tripwire"]["triggered"]:
        tripped.append("r2_tripwire")
    if tripped:
        raise SystemExit(
            f"LEAKAGE DETECTED ({', '.join(tripped)}) — see "
            f"{RESULTS_PATH.name} for the recorded evidence."
        )
