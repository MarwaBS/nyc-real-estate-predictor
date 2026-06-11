"""Flagship-specific firewall layer — backed by the schema-firewall package.

Historically this file implemented the firewall checks in-place. It has
been refactored to delegate to ``schema-firewall`` (published on PyPI,
imported as ``schema_firewall``) so the flagship's firewall and the
external library share one implementation. The flagship now *depends on*
its own published library. If the library breaks, this file breaks.
That is deliberate.

Four things remain local:

1. ``SCHEMA_MAP_VERSION`` — pinned to this repository's SCHEMA_MAP.md.
2. ``FORBIDDEN_COLUMNS`` — the flagship's concrete set, passed to the
   library via a ``SchemaContract``.
3. ``check_predictions_healthy`` + ``HealthError`` — prediction-array
   collapse detection is flagship-specific (the public library
   intentionally declines to include it, per its 3-entry-point cap).
4. ``schema_map_sha256`` + ``verify_schema_map_lock`` — the version
   registry enforcement. The hash is computed over LF-normalised bytes
   so the lock is identical across Windows/Linux checkouts, and the
   orchestrator (not just the test suite) refuses to run against an
   unsealed contract.

All other names (``LeakageError``, ``check_no_forbidden_columns``,
``check_target_independence``) are preserved as thin wrappers so that
the existing adversarial test suite keeps passing without edits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from schema_firewall import (
    LeakageError,
    SchemaContract,
    SchemaError,
    check_leakage,
    check_schema,
)

SCHEMA_MAP_VERSION = "v3"

_BENCHMARKS_DIR = Path(__file__).resolve().parent
SCHEMA_MAP_PATH = _BENCHMARKS_DIR / "SCHEMA_MAP.md"
VERSIONS_PATH = _BENCHMARKS_DIR / "SCHEMA_MAP_VERSIONS.json"


class SchemaLockError(Exception):
    """SCHEMA_MAP.md does not match the SHA sealed for SCHEMA_MAP_VERSION.

    Raised by :func:`verify_schema_map_lock` — the run-time enforcement of
    the registry. A benchmark run against an unsealed contract is invalid
    by definition (SCHEMA_MAP.md §9), so the orchestrator hard-fails
    instead of recording the mismatching hash as an FYI.
    """


def schema_map_sha256(path: Path = SCHEMA_MAP_PATH) -> str:
    """SHA-256 of SCHEMA_MAP.md over LF-normalised bytes.

    Normalising CRLF→LF before hashing makes the lock platform-deterministic:
    a Windows checkout with ``core.autocrlf=true`` would otherwise hash
    different bytes than the Linux CI runner for an identical contract.
    (.gitattributes also pins LF checkouts; this is the belt to that braces.)
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_schema_map_lock() -> str:
    """Enforce the schema lock: current SCHEMA_MAP.md SHA must equal the
    registry entry sealed for :data:`SCHEMA_MAP_VERSION`.

    Returns the verified SHA so callers can record it. Raises
    :class:`SchemaLockError` on any mismatch or missing registry entry.
    """
    registry = json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))
    sealed = registry.get("versions", {}).get(SCHEMA_MAP_VERSION)
    if sealed is None:
        raise SchemaLockError(
            f"no registry entry for SCHEMA_MAP_VERSION={SCHEMA_MAP_VERSION!r} "
            f"in {VERSIONS_PATH.name}"
        )
    # Module-global lookup at call time (not a bound default) so tests can
    # point the lock at a tampered copy.
    actual = schema_map_sha256(SCHEMA_MAP_PATH)
    if actual != sealed:
        raise SchemaLockError(
            f"SCHEMA_MAP.md hash does not match the sealed {SCHEMA_MAP_VERSION} "
            f"registry entry: actual={actual} sealed={sealed}. Either the file "
            f"changed without a version bump, or the run is outside the chain "
            f"of custody."
        )
    return actual


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
    check_leakage(X, target, max_abs_corr=max_abs_corr, mi_threshold=mi_threshold)


def check_predictions_healthy(
    predictions: np.ndarray | list[float],
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
    "SchemaLockError",
    "check_no_forbidden_columns",
    "check_target_independence",
    "check_predictions_healthy",
    "schema_map_sha256",
    "verify_schema_map_lock",
]
