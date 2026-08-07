"""Config and the shipped artefacts must describe the same model.

Changing a constant in ``src/config.py`` invalidates artefacts that were
fitted under the old value, and until now nothing connected the two: the
cut-points were shifted in config,
nothing was retrained, and all 153 tests passed. The published macro-F1 then
described zones cut one way while serving bucketed another.

These assert the agreement, so a config edit without a retrain fails the build
instead of silently rotting the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.config import (
    MODELS_DIR,
    NUMERIC_FEATURES,
    ONEHOT_FEATURES,
    PRICE_ZONE_BINS,
    RANDOM_SEED,
    TARGET_ENCODED_FEATURES,
    TEST_SIZE,
    VAL_SIZE,
)

METRICS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "reports" / "training_metrics.json"
    ).read_text(encoding="utf-8")
)


def test_zone_cut_points_match_the_ones_the_model_was_trained_with() -> None:
    """The zone labels are built from these bins; serving buckets with them too.

    A change here re-labels every training row AND re-buckets every served
    price, so the shipped macro-F1 stops describing what the service returns.
    """
    finite = [b for b in PRICE_ZONE_BINS if b != float("inf")]
    assert METRICS["provenance"]["price_zone_bins"] == finite, (
        "src/config.py PRICE_ZONE_BINS differs from the bins recorded in "
        "reports/training_metrics.json, the shipped zone metrics were "
        "computed under different cut-points. Retrain."
    )


def test_feature_set_matches_the_fitted_model() -> None:
    """A feature added or removed in config changes the model's input contract.

    The fitted pipeline carries the columns it was trained on; if config drifts
    from them, training and serving build different frames and the mismatch
    surfaces as a runtime error at predict time rather than here.
    """
    configured = sorted(NUMERIC_FEATURES + ONEHOT_FEATURES + TARGET_ENCODED_FEATURES)
    model = joblib.load(MODELS_DIR / "price_regressor_best.joblib")

    assert sorted(model.feature_names_in_) == configured
    assert sorted(METRICS["provenance"]["features"]) == configured


def test_split_and_seed_match_the_recorded_run() -> None:
    """Split sizes and the seed determine every number in the artefact."""
    prov = METRICS["provenance"]
    assert prov["random_seed"] == RANDOM_SEED
    assert prov["test_size"] == TEST_SIZE
    assert prov["val_size"] == VAL_SIZE
