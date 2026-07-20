"""The test split must not be readable during training decisions.

The published macro-F1 was 0.724 because per-class thresholds were fitted
against the test labels and the resulting score reported as a hold-out
number; candidate selection read the same labels. Both are the same defect —
a quantity chosen to maximise a test-set score cannot also be an unbiased
estimate of it — and both are structural, so they are pinned structurally.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import run_training
from src.config import NUMERIC_FEATURES, ONEHOT_FEATURES, TARGET_ENCODED_FEATURES


def _synthetic_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A frame carrying exactly the columns the shipped pipeline expects."""
    return pd.DataFrame(
        {
            **{c: rng.normal(size=n) for c in NUMERIC_FEATURES},
            **{c: rng.choice(["a", "b", "c"], size=n) for c in ONEHOT_FEATURES},
            **{c: rng.choice(["p", "q"], size=n) for c in TARGET_ENCODED_FEATURES},
        }
    )


def test_train_regression_reports_test_not_the_split_it_selected_on(
    tmp_path, monkeypatch
) -> None:
    """Behavioural, not a signature check: val targets are learnable from the
    features, test targets are noise. A reported R2 computed on whichever split
    drove selection would inherit val's high score instead of tracking test's
    near-zero one — which a renamed variable cannot fake."""
    monkeypatch.setattr(run_training, "MODELS_DIR", tmp_path)
    rng = np.random.default_rng(0)
    n = 240

    X_train, X_val, X_test = (_synthetic_frame(n, rng) for _ in range(3))
    signal = NUMERIC_FEATURES[0]
    y_train = X_train[signal].to_numpy()
    y_val = X_val[signal].to_numpy()
    y_test = rng.normal(size=n)

    record, _pipeline = run_training.train_regression(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    val_r2 = record["selection_metrics_val"]["r2"]
    test_r2 = record["metrics"]["r2"]
    assert val_r2 > 0.8, f"val should be learnable, got {val_r2}"
    assert test_r2 < 0.3, (
        f"reported R2 is {test_r2} on random test targets — it is not being "
        f"scored on the test split"
    )


def test_training_pipeline_has_no_threshold_tuning() -> None:
    """No step may fit a decision rule against the test labels.

    Threshold tuning did exactly that and published the result. Measured
    out-of-sample (fit on half the test set, scored on the other half, 20
    stratified splits) it was worth +0.0006 +/- 0.0106 — noise — so it is
    gone rather than moved to val.
    """
    src = inspect.getsource(run_training)
    assert "optimize_thresholds" not in src
    assert "optimal_thresholds" not in src


def test_split_sizes_are_three_way_and_disjoint() -> None:
    """train/val/test must partition the data — no row in two of them."""
    from sklearn.model_selection import train_test_split

    from src.config import RANDOM_SEED, TEST_SIZE, VAL_SIZE

    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=240)
    idx = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y[idx_tv]
    )
    assert set(idx_train) | set(idx_val) | set(idx_test) == set(idx)
    assert not set(idx_train) & set(idx_val)
    assert not set(idx_train) & set(idx_test)
    assert not set(idx_val) & set(idx_test)
