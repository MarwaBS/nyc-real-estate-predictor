"""Adversarial tests for the sealed ``benchmarks/SCHEMA_MAP.md`` contract.

Each test injects a specific contract violation and asserts the
firewall catches it. A green suite means the firewall is enforced,
not merely documented. The suite tracks whatever version is sealed in
``SCHEMA_MAP_VERSIONS.json`` (currently pinned by
``benchmarks.invariants.SCHEMA_MAP_VERSION``); the same checks run
again at benchmark time inside the orchestrator, so a violation fails
BOTH the test suite and any attempted run.
"""

from __future__ import annotations

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
    SchemaLockError,
    check_no_forbidden_columns,
    check_predictions_healthy,
    check_target_independence,
    schema_map_sha256,
    verify_schema_map_lock,
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


def test_target_independence_catches_nonlinear_target_encoding(
    nyc_rolling_sales_fixture,
):
    """A non-linear transform of the target (expm1 reverses log1p) must be caught.

    Pearson may underestimate the correlation on expm1 since the
    relationship is non-linear; Spearman and normalised MI still
    flag it.
    """
    x, target, _report = apply_schema_map(nyc_rolling_sales_fixture)
    x["smoothed_signal"] = np.expm1(target.to_numpy())
    with pytest.raises(LeakageError, match="correlated"):
        check_target_independence(x, target)


def test_target_independence_catches_low_cardinality_categorical_copy():
    """A low-cardinality OBJECT feature that DETERMINES the target must be caught.
    Regression: ``check_leakage`` only inspects numeric columns, so an object
    column that drives the target slipped past entirely. The firewall now
    label-encodes low-cardinality categoricals so the MI gate assesses them."""
    rng = np.random.default_rng(0)
    n = 300
    tier = rng.choice(["low", "mid", "high"], size=n)
    band = {"low": 11.0, "mid": 12.5, "high": 14.0}
    # Target is essentially a function of the categorical tier (tiny noise).
    target = pd.Series([band[t] for t in tier]) + rng.normal(0, 0.01, n)
    x = pd.DataFrame({"borough": tier, "sqft": rng.normal(size=n)})
    with pytest.raises(LeakageError):
        check_target_independence(x, target)


def test_high_cardinality_categorical_is_out_of_scope(nyc_rolling_sales_fixture):
    """A HIGH-cardinality categorical is deliberately NOT MI-checked: at finite n
    it is statistically indistinguishable from a strong legitimate location
    predictor. A unique-per-row object token (which trivially 'determines' the
    target) must NOT trip the gate — this pins the documented scope so it can't
    silently flip to false-positiving on legitimate ZIP/location signal."""
    x, target, _report = apply_schema_map(nyc_rolling_sales_fixture)
    x = x.copy()
    x["row_token"] = [f"tok_{i}" for i in range(len(x))]  # unique per row
    check_target_independence(x, target)  # must not raise


def test_target_independence_catches_nonmonotone_transform(
    nyc_rolling_sales_fixture,
):
    """A NON-MONOTONE transform of the target must be caught by MI alone.

    ``(target - mean)**2`` folds the target around its mean: Pearson AND
    Spearman both land near zero, so the correlation gate is blind — only
    the MI pillar can catch it. Under the old mi_threshold=0.8 (a guess
    made against 0.1.0's uncalibrated scale) this leak sailed through at
    a measured mi_norm of 0.631. Pins the leak side of the 0.45
    calibration.
    """
    x, target, _report = apply_schema_map(nyc_rolling_sales_fixture)
    x = x.copy()
    t = target.to_numpy()
    x["volatility_proxy"] = (t - t.mean()) ** 2
    with pytest.raises(LeakageError):
        check_target_independence(x, target)


def test_target_independence_passes_honest_features(nyc_rolling_sales_fixture):
    """The fixture's REAL feature frame must pass the gate as-is.

    Pins the false-positive side of the 0.45 calibration: borough is a
    genuinely predictive categorical (it sets the price level, measured
    mi_norm 0.207) and must stay under the threshold — a gate that fires
    on honest strong features would train the benchmark to bypass it.
    """
    x, target, _report = apply_schema_map(nyc_rolling_sales_fixture)
    check_target_independence(x, target)  # must not raise


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
    single, _, _ = apply_schema_map(nyc_rolling_sales_fixture.loc[[first_kept]].copy())
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


def test_nan_sale_price_is_dropped_not_poisoning_target(nyc_rolling_sales_fixture):
    """v3 regression guard: a blank SALE PRICE must become an explicit
    `missing_sale_price` drop — NaN compares False against every numeric
    threshold, so v2 silently kept these rows and produced log1p(NaN)
    targets that the hand-rolled R² propagated without complaint."""
    _, target, report = apply_schema_map(nyc_rolling_sales_fixture)
    assert report.drop_reasons.get("missing_sale_price", 0) >= 1
    assert np.isfinite(target.to_numpy(dtype=float)).all()


def test_nan_year_built_is_dropped(nyc_rolling_sales_fixture):
    """v3: YEAR BUILT blank/NaN drops alongside == 0 (record-quality rule)."""
    _, _, report = apply_schema_map(nyc_rolling_sales_fixture)
    assert report.drop_reasons.get("missing_year_built", 0) >= 2  # 0 and NaN rows


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
    """Silent edits to SCHEMA_MAP.md without a version bump must fail CI.

    The hash is LF-normalised (benchmarks.invariants.schema_map_sha256) so
    this check passes identically on a Windows CRLF checkout and the Linux
    CI runner — a determinism firewall must not itself be
    platform-nondeterministic.
    """
    file_sha = schema_map_sha256(SCHEMA_MAP_PATH)
    registry = json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))["versions"]
    assert registry[SCHEMA_MAP_VERSION] == file_sha, (
        f"SCHEMA_MAP.md hash changed without bumping SCHEMA_MAP_VERSION "
        f"(current={SCHEMA_MAP_VERSION}) and updating SCHEMA_MAP_VERSIONS.json. "
        "Any prior benchmark results.json is now invalid."
    )


def test_schema_map_sha_is_crlf_invariant(tmp_path):
    """CRLF and LF copies of the same contract must hash identically."""
    lf_copy = tmp_path / "lf.md"
    crlf_copy = tmp_path / "crlf.md"
    content_lf = SCHEMA_MAP_PATH.read_bytes().replace(b"\r\n", b"\n")
    lf_copy.write_bytes(content_lf)
    crlf_copy.write_bytes(content_lf.replace(b"\n", b"\r\n"))
    assert schema_map_sha256(lf_copy) == schema_map_sha256(crlf_copy)


def test_verify_schema_map_lock_passes_for_sealed_contract():
    """The run-time gate the orchestrator calls must accept the sealed file."""
    assert verify_schema_map_lock() == schema_map_sha256()


def test_verify_schema_map_lock_rejects_unsealed_edit(tmp_path, monkeypatch):
    """An edited contract must abort a run, not be recorded as data."""
    import benchmarks.invariants as inv

    tampered = tmp_path / "SCHEMA_MAP.md"
    tampered.write_bytes(SCHEMA_MAP_PATH.read_bytes() + b"\n<!-- tampered -->\n")
    monkeypatch.setattr(inv, "SCHEMA_MAP_PATH", tampered)
    with pytest.raises(SchemaLockError, match="does not match the sealed"):
        inv.verify_schema_map_lock()


# ─────────────────────────────────────────────────────────────────────
# 10. Prediction health / collapse detectors — distribution-free
# ─────────────────────────────────────────────────────────────────────


def test_collapse_detector_flags_constant_predictions():
    with pytest.raises(HealthError, match="collapse"):
        check_predictions_healthy(np.full(1000, 500_000.0))


def test_collapse_detector_flags_near_constant_predictions():
    rng = np.random.default_rng(0)
    preds = np.concatenate([np.full(960, 500_000.0), rng.uniform(1e5, 1e6, 40)])
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
