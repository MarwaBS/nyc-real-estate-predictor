"""Pure validation layer for the external benchmark firewall.

Implements the enforcement functions referenced by SCHEMA_MAP.md sections
6 and 10. No ML logic. No feature transforms. No I/O. Validation only.

Every function either returns None (pass) or raises ``LeakageError`` /
``HealthError`` (fail). Exceptions carry diagnostic context for CI logs.

Scope boundary: this module does not transform data, does not load
artefacts, and does not know about model internals. It receives arrays
and frames from the caller and answers one question: does this satisfy
the stated invariant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SCHEMA_MAP_VERSION = "v1"


FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "SALE PRICE",
        "SALE DATE",
        "PRICE_PER_SQFT",
        "TARGET",
        "log_price",
    }
)


class LeakageError(Exception):
    """A forbidden column or a target-correlated feature was detected in X."""


class HealthError(Exception):
    """The prediction array failed distribution-free sanity checks."""


def check_no_forbidden_columns(X: pd.DataFrame) -> None:
    """Fail if any column in ``FORBIDDEN_COLUMNS`` is present in ``X``.

    This is the first, cheapest leakage check. It complements
    :func:`check_target_independence`, which catches semantic (renamed)
    leaks by measuring statistical dependency with the target.
    """
    present = [c for c in X.columns if c in FORBIDDEN_COLUMNS]
    if present:
        raise LeakageError(f"forbidden column(s) present in X: {present}")


def check_target_independence(
    X: pd.DataFrame,
    target: pd.Series,
    *,
    max_abs_corr: float = 0.95,
    mi_threshold: float = 0.8,
) -> None:
    """Fail if any column in ``X`` shows suspicious dependency with the target.

    Three complementary detectors:

    - Pearson |r|  — linear leakage
    - Spearman |rho| — monotonic (including log-target) leakage
    - normalised mutual information — general, including non-monotonic

    MI is normalised by the target's histogram-based Shannon entropy so
    the threshold is scale-free in [0, 1]. The normalisation is a
    thresholding heuristic, not a scientific measurement.
    """
    from sklearn.feature_selection import mutual_info_regression

    numeric = X.select_dtypes(include=[np.number])
    if numeric.empty:
        return

    target_arr = target.to_numpy(dtype=float)
    target_entropy = _shannon_entropy(target_arr)
    if target_entropy <= 0:
        raise LeakageError("target is constant; MI normalisation undefined")

    violations: list[str] = []
    for col in numeric.columns:
        feat = numeric[col].to_numpy(dtype=float)
        if not np.isfinite(feat).any():
            continue

        pearson = abs(_safe_corr(feat, target_arr, method="pearson"))
        spearman = abs(_safe_corr(feat, target_arr, method="spearman"))
        mi = float(
            mutual_info_regression(feat.reshape(-1, 1), target_arr, random_state=0)[0]
        )
        mi_norm = mi / target_entropy

        if (
            pearson > max_abs_corr
            or spearman > max_abs_corr
            or mi_norm > mi_threshold
        ):
            violations.append(
                f"{col}: pearson={pearson:.3f} "
                f"spearman={spearman:.3f} mi_norm={mi_norm:.3f}"
            )

    if violations:
        raise LeakageError(
            "target-correlated features detected; likely semantic leakage:\n  "
            + "\n  ".join(violations)
        )


def check_predictions_healthy(
    predictions,
    *,
    n_min: int = 500,
    max_identical_fraction: float = 0.95,
) -> None:
    """Distribution-free collapse / degeneracy detector.

    Checks only invariants, not distribution shape:

    - enough predictions produced (``n_min``)
    - zero NaN / Inf
    - no single value covers more than ``max_identical_fraction`` of outputs

    These are collapse detectors, not performance gates. No threshold on
    R-squared; no expectation about prediction range, spread, or shape.
    A healthy model predicting garbage with high variance passes these
    checks and still fails on R-squared in the public results.
    """
    arr = np.asarray(predictions, dtype=float)

    if arr.size < n_min:
        raise HealthError(f"insufficient predictions: {arr.size} < {n_min}")

    n_nan = int(np.isnan(arr).sum())
    if n_nan > 0:
        raise HealthError(f"nan predictions: {n_nan}/{arr.size}")

    n_inf = int(np.isinf(arr).sum())
    if n_inf > 0:
        raise HealthError(f"inf predictions: {n_inf}/{arr.size}")

    _values, counts = np.unique(arr, return_counts=True)
    max_fraction = counts.max() / arr.size
    if max_fraction > max_identical_fraction:
        dominant = float(_values[counts.argmax()])
        raise HealthError(
            f"prediction collapse: {max_fraction:.3%} of outputs equal "
            f"{dominant}; threshold {max_identical_fraction:.0%}"
        )


def _safe_corr(a: np.ndarray, b: np.ndarray, *, method: str) -> float:
    """Correlation with zero-variance and NaN guards.

    Returns 0.0 when the correlation is undefined (fewer than two
    finite paired observations, or zero variance in either series).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    a, b = a[mask], b[mask]
    if a.std() == 0 or b.std() == 0:
        return 0.0

    if method == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    if method == "spearman":
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    raise ValueError(f"unknown method: {method}")


def _shannon_entropy(x: np.ndarray, *, bins: int = 64) -> float:
    """Histogram-based Shannon entropy in nats.

    Used to normalise ``mutual_info_regression`` output into a
    ratio suitable for thresholding. Precision is not critical because
    the value only drives a pass/fail gate, not a reported metric.
    """
    x = x[np.isfinite(x)]
    if x.size < 2 or x.std() == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


__all__ = [
    "SCHEMA_MAP_VERSION",
    "FORBIDDEN_COLUMNS",
    "LeakageError",
    "HealthError",
    "check_no_forbidden_columns",
    "check_target_independence",
    "check_predictions_healthy",
]
