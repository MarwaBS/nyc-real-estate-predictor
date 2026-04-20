"""Adversarial tests for ``benchmarks/SCHEMA_MAP.md`` v1.

Each test injects a specific contract violation and asserts the
firewall catches it. A green suite means the firewall is enforced,
not merely documented.

Step 3 status: this suite is written against a deliberately
non-functional stub of :func:`benchmarks.mapping.apply_schema_map`
that raises :class:`NotImplementedError`. Expected outcome: every
mapping-dependent test fails with ``NotImplementedError``; the
schema-SHA check and the collapse-detector checks pass, because
those do not depend on the mapping implementation. Step 4 lands the
real mapping and the whole suite must go green without any test
modification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.invariants import (
    FORBIDDEN_COLUMNS,
    SCHEMA_MAP_VERSION,
    HealthError,
    LeakageError,
    check_no_forbidden_columns,
    check_predictions_healthy,
    check_target_independence,
)
from benchmarks.mapping import apply_schema_map

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_MAP_PATH = REPO_ROOT / "benchmarks" / "SCHEMA_MAP.md"
VERSIONS_PATH = REPO_ROOT / "benchmarks" / "SCHEMA_MAP_VERSIONS.json"


# ─────────────────────────────────────────────────────────────────────
# 1. Name-based leakage — parametrised over every forbidden column
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("forbidden_col", sorted(FORBIDDEN_COLUMNS))
def test_forbidden_column_rejected(nyc_rolling_sales_fixture, forbidden_col):
    """Smuggling any FORBIDDEN_COLUMNS entry into X must raise LeakageError."""
    x, _target, _report = apply_schema_map(nyc_rolling_sales_fixture)
    x[forbidden_col] = 1.0
    with pytest.raises(LeakageError, match=forbidden_col):
        check_no_forbidden_columns(x)


# ─────────────────────────────────────────────────────────────────────
# 2. Semantic leakage — renamed target, caught by Pearson / Spearman / MI
# ─────────────────────────────────────────────────────────────────────

def test_target_independence_catches_renamed_target(nyc_rolling_sales_fixture):
    """A renamed, nearly-identical copy of the target must be caught."""
    x, target, _report = apply_schema_map(nyc_rolling_sales_fixture)
    noise = np.random.default_rng(0).normal(0, 1e-9, len(target))
    x["neighborhood_affluence_index"] = target.to_numpy() + noise
    with pytest.raises(LeakageError, match="correlated"):
        check_target_independence(x, target)


def test_target_independence_catches_nonlinear_target_encoding(nyc_rolling_sales_fixture):
    """A non-linear transform of the target (expm1 reverses log1p) must be caught.

    Pearson may underestimate the correlation on expm1 since the
    relationship is non-linear; Spearman and normalised MI still
    flag it.
    """
    x, target, _report = apply_schema_map(nyc_rolling_sales_fixture)
    x["smoothed_signal"] = np.expm1(target.to_numpy())
    with pytest.raises(LeakageError, match="correlated"):
        check_target_independence(x, target)


# ─────────────────────────────────────────────────────────────────────
# 3. Determinism
# ─────────────────────────────────────────────────────────────────────

def test_mapping_is_deterministic(nyc_rolling_sales_fixture):
    """Same input frame must produce bitwise-identical output frames."""
    x_a, _, _ = apply_schema_map(nyc_rolling_sales_fixture)
    x_b, _, _ = apply_schema_map(nyc_rolling_sales_fixture)
    pd.testing.assert_frame_equal(x_a, x_b)


# ─────────────────────────────────────────────────────────────────────
# 4. Statelessness — subset invariance (single-row vs full-frame)
# ─────────────────────────────────────────────────────────────────────

def test_mapping_is_stateless_across_subsets(nyc_rolling_sales_fixture):
    """Mapping a kept row alone must match the same row inside the full frame.

    Catches mean-encoders, frequency encoders, rank transforms, and any
    other dataset-wide-statistic leakage channel that survives
    row-order shuffles.
    """
    full, _, report = apply_schema_map(nyc_rolling_sales_fixture)
    kept_indices = report.kept_mask[report.kept_mask].index.tolist()
    if not kept_indices:
        pytest.skip("fixture has no kept rows; test needs at least one")
    first_kept = kept_indices[0]
    single, _, _ = apply_schema_map(
        nyc_rolling_sales_fixture.loc[[first_kept]].copy()
    )
    full_first = full.loc[[first_kept]].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_first, single.reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────
# 5. Target identity — label shuffle must not change features
# ─────────────────────────────────────────────────────────────────────

def test_mapping_ignores_target_identity(nyc_rolling_sales_fixture):
    """Permuting SALE PRICE must leave X structurally identical.

    Any feature construction that touches the target will change X
    when SALE PRICE is shuffled, failing this test.
    """
    raw = nyc_rolling_sales_fixture.copy()
    shuffled = raw.copy()
    shuffled["SALE PRICE"] = np.random.default_rng(42).permutation(
        shuffled["SALE PRICE"].to_numpy()
    )
    x_a, _, _ = apply_schema_map(raw)
    x_b, _, _ = apply_schema_map(shuffled)
    # SALE PRICE may have moved rows in/out of drop sets (e.g. a row
    # was < 10,000 before the shuffle). Compare only rows kept in both.
    common = x_a.index.intersection(x_b.index)
    pd.testing.assert_frame_equal(x_a.loc[common], x_b.loc[common])


# ─────────────────────────────────────────────────────────────────────
# 6. Column-name indexing — not positional
# ─────────────────────────────────────────────────────────────────────

def test_mapping_uses_column_names_not_positions(nyc_rolling_sales_fixture):
    """Reversing column order in the input must not change X."""
    raw = nyc_rolling_sales_fixture
    permuted = raw[list(reversed(raw.columns))]
    x_a, _, _ = apply_schema_map(raw)
    x_b, _, _ = apply_schema_map(permuted)
    pd.testing.assert_frame_equal(x_a, x_b)


# ─────────────────────────────────────────────────────────────────────
# 7. Drop-log consistency
# ─────────────────────────────────────────────────────────────────────

def test_drop_reasons_equal_dropped_rows(nyc_rolling_sales_fixture):
    """Logged drop-reason counts must sum to n_dropped; raw == scored + dropped."""
    _, _, report = apply_schema_map(nyc_rolling_sales_fixture)
    assert report.n_dropped == sum(report.drop_reasons.values())
    assert report.n_raw == len(nyc_rolling_sales_fixture)
    assert report.n_raw == report.n_scored + report.n_dropped


# ─────────────────────────────────────────────────────────────────────
# 8. Filter is not target-aware
# ─────────────────────────────────────────────────────────────────────

def test_filter_independent_of_target_distribution(nyc_rolling_sales_fixture):
    """Shifting SALE PRICE into a uniform in-bounds band must not change
    drop counts for reasons unrelated to price."""
    raw = nyc_rolling_sales_fixture.copy()
    _, _, report_a = apply_schema_map(raw)

    rescaled = raw.copy()
    rng = np.random.default_rng(0)
    in_bounds = (raw["SALE PRICE"] >= 10_000) & (raw["SALE PRICE"] <= 100_000_000)
    new_prices = rng.uniform(500_000, 2_000_000, len(raw))
    rescaled.loc[in_bounds, "SALE PRICE"] = new_prices[in_bounds.to_numpy()]
    _, _, report_b = apply_schema_map(rescaled)

    def non_price(report):
        return {
            k: v for k, v in report.drop_reasons.items() if "price" not in k.lower()
        }

    assert non_price(report_a) == non_price(report_b)


# ─────────────────────────────────────────────────────────────────────
# 9. Version registry — SCHEMA_MAP.md SHA matches the pinned entry
# ─────────────────────────────────────────────────────────────────────

def test_schema_map_sha_matches_registered_version():
    """Silent edits to SCHEMA_MAP.md without a version bump must fail CI."""
    file_sha = hashlib.sha256(SCHEMA_MAP_PATH.read_bytes()).hexdigest()
    registry = json.loads(VERSIONS_PATH.read_text())["versions"]
    assert registry[SCHEMA_MAP_VERSION] == file_sha, (
        f"SCHEMA_MAP.md hash changed without bumping SCHEMA_MAP_VERSION "
        f"(current={SCHEMA_MAP_VERSION}) and updating SCHEMA_MAP_VERSIONS.json. "
        "Any prior benchmark results.json is now invalid."
    )


# ─────────────────────────────────────────────────────────────────────
# 10. Prediction health / collapse detectors — distribution-free
# ─────────────────────────────────────────────────────────────────────

def test_collapse_detector_flags_constant_predictions():
    with pytest.raises(HealthError, match="collapse"):
        check_predictions_healthy(np.full(1000, 500_000.0))


def test_collapse_detector_flags_near_constant_predictions():
    rng = np.random.default_rng(0)
    preds = np.concatenate(
        [np.full(960, 500_000.0), rng.uniform(1e5, 1e6, 40)]
    )
    with pytest.raises(HealthError, match="collapse"):
        check_predictions_healthy(preds, max_identical_fraction=0.95)


def test_collapse_detector_flags_nan_predictions():
    preds = np.full(1000, np.nan)
    with pytest.raises(HealthError, match="nan"):
        check_predictions_healthy(preds)


def test_collapse_detector_accepts_healthy_predictions():
    rng = np.random.default_rng(0)
    preds = rng.lognormal(mean=13, sigma=0.5, size=1000)
    check_predictions_healthy(preds)
