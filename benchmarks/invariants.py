"""Flagship-specific firewall layer — backed by the schema-firewall package.

Historically this file implemented the firewall checks in-place. It has
been refactored to delegate to ``schema-firewall`` (published on PyPI,
imported as ``schema_firewall``) so the flagship's firewall and the
external library share one implementation. The flagship now *depends on*
its own published library. If the library breaks, this file breaks.
That is deliberate.

Three things remain local:

1. ``SCHEMA_MAP_VERSION`` — pinned to this repository's SCHEMA_MAP.md.
2. ``FORBIDDEN_COLUMNS`` — the flagship's concrete set, passed to the
   library via a ``SchemaContract``.
3. ``check_predictions_healthy`` + ``HealthError`` — prediction-array
   collapse detection is flagship-specific (the public library
   intentionally declines to include it, per its 3-entry-point cap).

All other names (``LeakageError``, ``check_no_forbidden_columns``,
``check_target_independence``) are preserved as thin wrappers so that
the existing adversarial test suite keeps passing without edits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from schema_firewall import (
    LeakageError,
    SchemaContract,
    SchemaError,
    check_leakage,
    check_schema,
)


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


_FLAGSHIP_CONTRACT = SchemaContract(forbidden_columns=FORBIDDEN_COLUMNS)


class HealthError(Exception):
    """The prediction array failed distribution-free sanity checks.

    Flagship-specific; not part of the schema-firewall public surface.
    """


def check_no_forbidden_columns(X: pd.DataFrame) -> None:
    """Fail if any FORBIDDEN_COLUMNS entry is present in ``X``.

    Thin wrapper over ``schema_firewall.check_schema`` with the
    flagship's forbidden set. The exception class is translated from
    ``SchemaError`` (library) to ``LeakageError`` (flagship) so the
    existing test suite, which matches on ``LeakageError``, keeps
    working unchanged.
    """
    try:
        check_schema(X, _FLAGSHIP_CONTRACT)
    except SchemaError as exc:
        raise LeakageError(str(exc)) from exc


def check_target_independence(
    X: pd.DataFrame,
    target: pd.Series,
    *,
    max_abs_corr: float = 0.95,
    mi_threshold: float = 0.8,
) -> None:
    """Fail if any column in ``X`` shows suspicious dependency with ``target``.

    Thin wrapper over ``schema_firewall.check_leakage``. Preserves the
    flagship's keyword-argument defaults.
    """
    check_leakage(
        X, target, max_abs_corr=max_abs_corr, mi_threshold=mi_threshold
    )


def check_predictions_healthy(
    predictions,
    *,
    n_min: int = 500,
    max_identical_fraction: float = 0.95,
) -> None:
    """Distribution-free collapse / degeneracy detector.

    Flagship-specific; remains in this module because the public
    schema-firewall library caps at three check functions (leakage,
    schema, statelessness). Keeps the existing behaviour exactly.
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


__all__ = [
    "SCHEMA_MAP_VERSION",
    "FORBIDDEN_COLUMNS",
    "LeakageError",
    "HealthError",
    "check_no_forbidden_columns",
    "check_target_independence",
    "check_predictions_healthy",
]
