"""The served price interval must be calibrated, and must say what it covers.

An interval presented to users as a price range while containing the true
price only some of the time is a precise-looking number with no evidence
behind it. These tests assert the shipped multipliers and their measured
coverage.

The artefact-reading tests below assert the shipped numbers; the calibration
tests drive ``calibrate_price_interval`` directly on synthetic residuals, so
they check the mechanism rather than re-asserting the file against itself.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_training import calibrate_price_interval
from src.models.predict import price_range

ARTEFACT = Path(__file__).resolve().parents[1] / "models" / "price_interval.json"


@pytest.fixture(scope="module")
def interval() -> dict:
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


def test_interval_is_calibrated_on_val_not_test(interval: dict) -> None:
    """Choosing the multipliers on test would make coverage_test in-sample."""
    assert interval["calibrated_on"] == "val"


def test_calibrate_price_interval_labels_the_split_it_actually_used() -> None:
    """The label must follow the data, not a literal written beside it.

    Previously ``calibrated_on`` was the constant "val" regardless of which
    frame was quantiled, so the artefact could claim val-calibration while
    holding test-calibrated multipliers and every gate stayed green. Calibrating
    on a deliberately different split here must change BOTH the label and the
    multipliers; asserting only the label would restore the vacuous check.
    """
    rng = np.random.default_rng(0)

    class _StubRegressor:
        """Predicts a flat $1, so ratio == actual and the quantiles are the inputs.

        log1p(1.0), not 0.0: the function un-logs predictions with expm1, and
        expm1(0) is 0, which makes every actual/predicted ratio infinite.
        """

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), np.log1p(1.0))

    # Disjoint ranges, so the two splits' quantiles cannot coincide. Named
    # "val"/"holdout" rather than "val"/"test" because calibrating on a split
    # named "test" is refused outright -- see the leak test below.
    val = pd.Series(np.log1p(rng.uniform(1.0, 2.0, 500)))
    holdout = pd.Series(np.log1p(rng.uniform(10.0, 20.0, 500)))
    splits = {
        "val": (pd.DataFrame({"f": np.zeros(500)}), val),
        "holdout": (pd.DataFrame({"f": np.zeros(500)}), holdout),
    }

    on_val = calibrate_price_interval(_StubRegressor(), splits, calibrate_on="val")
    on_test = calibrate_price_interval(_StubRegressor(), splits, calibrate_on="holdout")

    assert on_val["calibrated_on"] == "val"
    assert on_test["calibrated_on"] == "holdout"
    # The stub predicts a flat $1, so each multiplier is a quantile of that
    # split's own actuals: val must land inside [1, 2] and test inside
    # [10, 20], the ranges the two splits were drawn from. Asserting the
    # constructed ranges beats asserting a ratio threshold, which would be a
    # slack factor picked to pass rather than a derived expectation.
    assert 1.0 <= on_val["high_multiplier"] <= 2.0
    assert 10.0 <= on_test["high_multiplier"] <= 20.0


def test_calibrate_price_interval_refuses_to_calibrate_on_test() -> None:
    """The leak must be refused at the call, not merely labelled afterwards.

    Changing the call site to calibrate_on="test" previously left all 140
    tests green: the artefact-reading gate above only inspects the committed
    file, and CI never retrains, so the leak would ship and be detected only
    once someone regenerated and committed the artefact.
    """
    splits = {
        "val": (pd.DataFrame({"f": [0.0]}), pd.Series([1.0])),
        "test": (pd.DataFrame({"f": [0.0]}), pd.Series([1.0])),
    }
    with pytest.raises(ValueError, match="out-of-sample"):
        calibrate_price_interval(object(), splits, calibrate_on="test")


def test_calibrate_price_interval_rejects_an_unknown_split() -> None:
    with pytest.raises(KeyError):
        calibrate_price_interval(
            object(), {"val": (pd.DataFrame(), pd.Series())}, "holdout"
        )


class _FlatDollarRegressor:
    """Predicts a flat $1 so each ratio is just the actual. log1p(1.0), not
    0.0, expm1(0) is 0 and every ratio would be infinite."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), np.log1p(1.0))


def test_the_interval_uses_the_conformal_level_not_the_plain_quantile() -> None:
    """The finite-sample correction is what the 2 SE coverage bound rests on.

    The expected level is found by search, not by re-evaluating ceil((n+1)q) -
    that would put the same expression on both sides of the assertion.
    """
    rng = np.random.default_rng(0)
    n = 500
    ratios = np.sort(rng.uniform(1.0, 3.0, n))
    splits = {"val": (pd.DataFrame({"f": np.zeros(n)}), pd.Series(np.log1p(ratios)))}

    # Written out, not imported, so this test does not move with the constant.
    hi_q = 0.9
    k = next(i for i in range(1, n + 2) if i / (n + 1) >= hi_q)
    expected_level = min(k / n, 1.0)

    result = calibrate_price_interval(
        _FlatDollarRegressor(), splits, calibrate_on="val", target=0.80
    )

    assert result["high_multiplier"] == pytest.approx(
        round(float(np.quantile(ratios, expected_level)), 4), abs=1e-9
    )
    # If the corrected level coincided with the plain one, the assertion above
    # would pass on the mutation it exists to catch.
    assert float(np.quantile(ratios, expected_level)) != pytest.approx(
        float(np.quantile(ratios, hi_q)), abs=1e-9
    )


def test_the_shipped_interval_targets_the_coverage_the_code_asks_for(
    interval: dict,
) -> None:
    """CI never retrains, so the constant must be compared to the artefact."""
    from run_training import PRICE_INTERVAL_TARGET

    assert PRICE_INTERVAL_TARGET == 0.80
    assert interval["target_coverage"] == PRICE_INTERVAL_TARGET, (
        "run_training.PRICE_INTERVAL_TARGET differs from the target recorded "
        "in models/price_interval.json, the shipped interval was calibrated "
        "to a different coverage than the code now asks for. Retrain."
    )


def test_measured_coverage_is_close_to_the_target_it_advertises(
    interval: dict,
) -> None:
    """Coverage must sit within 2 binomial standard errors of the target.

    2 SE, not 3. The previous bound was 3 SE for one reason: at 2 SE the
    shipped interval failed, because the calibration used the plain empirical
    quantile instead of split conformal's finite-sample correction. Picking
    the multiplier that let the number pass, then writing a derivation around
    it, is the defect this file exists to catch -- so the math was fixed and
    the bound tightened rather than the reverse.

    sqrt(0.8*0.2/906) = 0.0133 on the 906-row test split, so 2 SE = 0.0266.
    """
    target = interval["target_coverage"]
    n_test = 906
    tolerance = 2 * math.sqrt(target * (1 - target) / n_test)
    assert abs(interval["coverage_test"] - target) <= tolerance, (
        f"interval advertises {target:.0%} coverage but measured "
        f"{interval['coverage_test']:.1%} on the test split -- "
        f"{abs(interval['coverage_test'] - target) / (tolerance / 2):.2f} SE away"
    )


def test_multipliers_bracket_the_prediction(interval: dict) -> None:
    """low < 1 < high, an interval that excludes its own point estimate is
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
