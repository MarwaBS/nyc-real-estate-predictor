"""The recorded feature baseline must describe the training data.

`save_baseline` is the whole live surface: run_training writes
models/drift_baseline.json from the train split. The comparison helpers that
once sat beside it had no caller and were deleted rather than maintained."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.drift import compute_feature_stats


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "BEDS": rng.randint(1, 6, 100),
            "BATH": rng.uniform(1, 4, 100),
            "PROPERTYSQFT": rng.uniform(400, 4000, 100),
            "BOROUGH": rng.choice(
                ["manhattan", "brooklyn"], 100
            ),  # Non-numeric, skipped
        }
    )


def test_compute_feature_stats_returns_expected_keys(sample_df: pd.DataFrame) -> None:
    stats = compute_feature_stats(sample_df)
    assert "BEDS" in stats
    assert "BATH" in stats
    assert "PROPERTYSQFT" in stats
    assert "BOROUGH" not in stats  # Non-numeric excluded
    assert "mean" in stats["BEDS"]
    assert "std" in stats["BEDS"]
    assert "p50" in stats["BEDS"]
