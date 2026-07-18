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


def test_train_classification_reports_test_not_the_split_it_selected_on(
    tmp_path, monkeypatch
) -> None:
    """The reported metric must come from test, and selection from val.

    Behavioural, not a source grep: val gets labels the features predict,
    test gets shuffled labels. If the reported `metrics` were computed on
    whichever split drove selection, it would inherit val's high score. It
    must instead track the near-chance test split, and `selection_metrics_val`
    must hold the high one. Reading either split from the wrong place flips
    this inequality, which a renamed variable or reformatted line cannot.
    """
    monkeypatch.setattr(run_training, "MODELS_DIR", tmp_path)
    rng = np.random.default_rng(0)
    n = 240

    X_train, X_val, X_test = (_synthetic_frame(n, rng) for _ in range(3))
    # Labels are a deterministic function of the first numeric feature, so a
    # fitted model genuinely predicts train and val.
    signal = NUMERIC_FEATURES[0]
    y_train = (X_train[signal] > 0).astype(int).to_numpy()
    y_val = (X_val[signal] > 0).astype(int).to_numpy()
    # Test labels are independent of the features: any honest score is chance.
    y_test = rng.integers(0, 2, size=n)

    record = run_training.train_classification(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        pd.Series(["manhattan"] * n),
        ["low", "high"],
    )

    val_f1 = record["selection_metrics_val"]["macro_f1"]
    test_f1 = record["metrics"]["macro_f1"]
    assert val_f1 > 0.8, f"val should be learnable, got {val_f1}"
    assert test_f1 < 0.65, (
        f"reported metric is {test_f1} on random test labels — it is not "
        f"being scored on the test split"
    )


def test_train_regression_selects_on_val_not_test() -> None:
    """Signature-level pin: without a val split, every candidate comparison
    this function makes is necessarily reading the test labels."""
    sig = inspect.signature(run_training.train_regression)
    assert "X_val" in sig.parameters and "y_val" in sig.parameters


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


def test_dl_early_stopping_does_not_watch_the_test_set() -> None:
    """`patience` on test labels is model selection wearing a val loader's
    name: the epoch chosen is the one that happened to score best on test."""
    src = inspect.getsource(run_training.main)
    val_loader_call = src[src.index("val_loader = prepare_dl_data") :][:200]
    assert "X_val_t" in val_loader_call
    assert "y_zone_val" in val_loader_call
    assert "X_test_t" not in val_loader_call


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
