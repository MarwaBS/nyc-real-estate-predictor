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

import run_training


def test_train_classification_selects_on_val_not_test() -> None:
    """Selection must consult val; test must reach only the final scoring.

    Pinned structurally rather than behaviourally: whether selection is
    biased depends on which split a comparison reads, and a run on toy data
    can pick the same winner either way, so a behavioural test would pass on
    the broken code. Reading the split out of the signature and source cannot.
    """
    sig = inspect.signature(run_training.train_classification)
    assert "X_val" in sig.parameters and "y_val" in sig.parameters, (
        "train_classification takes no validation set, so every candidate "
        "comparison it makes must be reading the test labels"
    )
    # The reported record separates the split that chose the model from the
    # split the headline number is scored on. Both keys are load-bearing for
    # the artefact's honesty, so their absence is a failure.
    src = inspect.getsource(run_training.train_classification)
    assert "selection_metrics_val" in src
    assert "predict(X_val)" in src, "candidates must be scored on val"


def test_train_regression_selects_on_val_not_test() -> None:
    sig = inspect.signature(run_training.train_regression)
    assert "X_val" in sig.parameters and "y_val" in sig.parameters
    src = inspect.getsource(run_training.train_regression)
    assert "predict(X_val)" in src, "candidates must be scored on val"


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
