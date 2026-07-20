"""Every cross-row statistic must be fitted on the TRAIN split only.

Cap bounds, zone cut-points and the category vocabulary all shape the training
target or the model's input contract; fitting any of them on pooled data lets
val/test rows leak into training. Each test recomputes the statistic from the
train rows build_splits selected and requires exact agreement — a pooled fit
lands on different quantiles/counts and fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from run_training import build_splits
from src.data.cleaner import apply_cap, fit_cap_bounds
from src.data.features import fit_top_categories


@pytest.fixture(scope="module")
def clean_frame() -> pd.DataFrame:
    """A cleaned-shaped frame with heavy tails, so pooled and train-sample
    quantiles cannot coincide."""
    rng = np.random.default_rng(7)
    n = 400
    return pd.DataFrame(
        {
            "PRICE": np.concatenate(
                [rng.uniform(2e5, 2e6, n - 20), rng.uniform(5e6, 5e7, 20)]
            ),
            "PROPERTYSQFT": rng.uniform(300, 6000, n),
            "BEDS": rng.integers(1, 8, n).astype(float),
            "BATH": rng.integers(1, 6, n).astype(float),
            "LATITUDE": rng.uniform(40.5, 40.9, n),
            "LONGITUDE": rng.uniform(-74.2, -73.7, n),
            "BOROUGH": rng.choice(
                ["manhattan", "brooklyn", "queens", "the bronx", "staten island"], n
            ),
            "TYPE": rng.choice(["condo", "house", "co-op"], n),
            "ZIPCODE": rng.choice([f"1{i:04d}" for i in range(60)], n),
            "SUBLOCALITY": rng.choice([f"area {i}" for i in range(60)], n),
        }
    )


@pytest.fixture(scope="module")
def prep(clean_frame: pd.DataFrame) -> dict:
    return build_splits(clean_frame, seed=0)


def test_cap_bounds_come_from_the_train_rows(
    clean_frame: pd.DataFrame, prep: dict
) -> None:
    expected = fit_cap_bounds(
        clean_frame.reset_index(drop=True).loc[prep["idx"]["train"]]
    )
    for col, (lo, hi) in expected.items():
        assert prep["cap_bounds"][col][0] == pytest.approx(lo, abs=1e-9)
        assert prep["cap_bounds"][col][1] == pytest.approx(hi, abs=1e-9)


def test_zone_bins_are_quartiles_of_the_capped_train_prices(
    clean_frame: pd.DataFrame, prep: dict
) -> None:
    df = clean_frame.reset_index(drop=True)
    capped = apply_cap(df, fit_cap_bounds(df.loc[prep["idx"]["train"]]))
    train_prices = capped.loc[prep["idx"]["train"], "PRICE"]
    for q, bin_edge in zip((0.25, 0.50, 0.75), prep["zone_bins"][1:-1], strict=True):
        assert bin_edge == pytest.approx(float(train_prices.quantile(q)), abs=1e-9)


def test_category_vocabulary_is_counted_on_the_train_rows(
    clean_frame: pd.DataFrame, prep: dict
) -> None:
    expected = fit_top_categories(
        clean_frame.reset_index(drop=True).loc[prep["idx"]["train"]],
        columns=["SUBLOCALITY", "TYPE", "ZIPCODE"],
    )
    for col, vocab in expected.items():
        assert prep["category_vocabulary"][col] == sorted(vocab)


def test_splits_partition_the_data(clean_frame: pd.DataFrame, prep: dict) -> None:
    idx = prep["idx"]
    all_idx = set(idx["train"]) | set(idx["val"]) | set(idx["test"])
    assert all_idx == set(range(len(clean_frame)))
    assert not set(idx["train"]) & set(idx["test"])
    assert not set(idx["val"]) & set(idx["test"])
