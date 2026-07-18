"""The served price interval must be calibrated, and must say what it covers.

The shipped interval was a hardcoded +/-15% derived from nothing. Measured
against the real test split it contained the true price 32% of the time while
being presented to users as a price range — a precise-looking number with no
evidence behind it, which is the same defect as the threshold-tuned macro F1
this repo already corrected.

These tests read the committed artefact rather than recomputing from the raw
dataset, because CI has the artefact and does not have the dataset. A gate
that skips when data is absent is not a gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.predict import price_range

ARTEFACT = Path(__file__).resolve().parents[1] / "models" / "price_interval.json"


@pytest.fixture(scope="module")
def interval() -> dict:
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


def test_interval_is_calibrated_on_val_not_test(interval: dict) -> None:
    """Choosing the multipliers on test would make coverage_test in-sample."""
    assert interval["calibrated_on"] == "val"


def test_measured_coverage_is_close_to_the_target_it_advertises(
    interval: dict,
) -> None:
    """The interval's claim is its coverage, so that is what must hold.

    Fails on the pre-fix +/-15% band, whose coverage was 0.32 against any
    sensible target. The tolerance is one-sided-ish by intent: badly
    over-covering is also a broken claim, just a less harmful one.
    """
    target = interval["target_coverage"]
    assert abs(interval["coverage_test"] - target) <= 0.05, (
        f"interval advertises {target:.0%} coverage but measured "
        f"{interval['coverage_test']:.1%} on the test split"
    )


def test_multipliers_bracket_the_prediction(interval: dict) -> None:
    """low < 1 < high — an interval that excludes its own point estimate is
    incoherent, and a symmetric one would misdescribe a log-target model whose
    residuals are asymmetric in dollar space."""
    assert 0.0 < interval["low_multiplier"] < 1.0 < interval["high_multiplier"]


def test_serving_uses_the_artefact_not_a_hardcoded_band(interval: dict) -> None:
    """price_range must derive from the artefact.

    Pins the actual regression: three surfaces each hardcoded 0.85/1.15, so a
    recalibration would have silently failed to reach any of them.
    """
    band = price_range(1_000_000.0)
    assert band["low"] == round(1_000_000.0 * interval["low_multiplier"], -2)
    assert band["high"] == round(1_000_000.0 * interval["high_multiplier"], -2)
    # The old hardcoded band, explicitly rejected.
    assert (band["low"], band["high"]) != (850_000, 1_150_000)


def test_missing_artefact_raises_rather_than_guessing(monkeypatch) -> None:
    """A missing calibration must fail loudly.

    Falling back to a default would reinstate an interval nothing measured,
    which is precisely what is being removed.
    """
    import src.models.predict as m

    monkeypatch.setattr(m, "_price_interval", None)
    monkeypatch.setattr(m, "MODELS_DIR", Path("/nonexistent-models-dir"))
    with pytest.raises(FileNotFoundError, match="price_interval"):
        m.get_price_interval()
