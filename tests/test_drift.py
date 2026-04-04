"""Tests for model drift detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.drift import (
    check_drift,
    compute_feature_stats,
    detect_drift,
    save_baseline,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "BEDS": rng.randint(1, 6, 100),
        "BATH": rng.uniform(1, 4, 100),
        "PROPERTYSQFT": rng.uniform(400, 4000, 100),
        "BOROUGH": rng.choice(["manhattan", "brooklyn"], 100),  # Non-numeric, skipped
    })


def test_compute_feature_stats_returns_expected_keys(sample_df: pd.DataFrame) -> None:
    stats = compute_feature_stats(sample_df)
    assert "BEDS" in stats
    assert "BATH" in stats
    assert "PROPERTYSQFT" in stats
    assert "BOROUGH" not in stats  # Non-numeric excluded
    assert "mean" in stats["BEDS"]
    assert "std" in stats["BEDS"]
    assert "p50" in stats["BEDS"]


def test_save_and_load_baseline(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    save_baseline(sample_df, path)
    assert path.exists()

    from src.models.drift import load_baseline
    baseline = load_baseline(path)
    assert "BEDS" in baseline
    assert baseline["BEDS"]["count"] == 100


def test_detect_drift_no_drift(sample_df: pd.DataFrame) -> None:
    baseline = compute_feature_stats(sample_df)
    drifted = detect_drift(sample_df, baseline)
    assert len(drifted) == 0


def test_detect_drift_catches_shift(sample_df: pd.DataFrame) -> None:
    baseline = compute_feature_stats(sample_df)
    # Shift BEDS mean significantly
    shifted = sample_df.copy()
    shifted["BEDS"] = shifted["BEDS"] + 10
    drifted = detect_drift(shifted, baseline, threshold=0.15)
    assert "BEDS" in drifted
    assert drifted["BEDS"]["shift_ratio"] > 0.15


def test_check_drift_raises_on_fail(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    save_baseline(sample_df, path)

    shifted = sample_df.copy()
    shifted["BEDS"] = shifted["BEDS"] + 10

    with pytest.raises(ValueError, match="DATA DRIFT DETECTED"):
        check_drift(shifted, path, threshold=0.15, fail_on_drift=True)


def test_check_drift_passes_when_clean(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    save_baseline(sample_df, path)
    result = check_drift(sample_df, path, threshold=0.15, fail_on_drift=True)
    assert len(result) == 0
