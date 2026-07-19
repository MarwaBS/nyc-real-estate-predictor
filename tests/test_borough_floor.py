"""Every borough must beat its own majority-class baseline, or the build fails.

"Predicts across the 5 boroughs" is a contract, not an average. A model with a
strong city-wide macro-F1 that fails in Queens is broken for everyone in
Queens, and a single headline number hides exactly that.

The floor is derived per borough rather than fixed: score the constant
predictor that always answers that borough's most common zone. A fixed
threshold would have been a number chosen with the current results already in
view -- and the previous run's Queens figure (0.601) sat close enough to any
round number that picking one would have been exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_training import BoroughFloorError, check_borough_floor

METRICS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "reports" / "training_metrics.json"
    ).read_text(encoding="utf-8")
)


def test_every_shipped_borough_clears_its_baseline() -> None:
    """The gate's own record, as shipped."""
    floor = METRICS["classification"]["borough_floor"]
    assert len(floor) == 5, f"expected 5 boroughs, got {sorted(floor)}"
    for name, row in floor.items():
        assert row["macro_f1"] > row["majority_baseline"], (
            f"{name}: {row['macro_f1']} <= baseline {row['majority_baseline']}"
        )


def test_a_borough_below_its_baseline_raises() -> None:
    """The gate must stop the run, not annotate the metrics file.

    Queens here is given predictions that are always the rarest zone, so it
    scores below the constant-majority predictor.
    """
    zones = ["Low", "Medium", "High", "Very High"]
    rng = np.random.default_rng(0)

    y_true = np.array(["Low"] * 40 + ["Medium"] * 30 + list(rng.choice(zones, 30)))
    borough = pd.Series(["queens"] * len(y_true))
    y_pred = np.array(["Very High"] * len(y_true))

    with pytest.raises(BoroughFloorError, match="queens"):
        check_borough_floor(y_true, y_pred, borough)


def test_a_passing_borough_returns_its_margin() -> None:
    """A clearing borough is recorded with the margin it cleared by."""
    y_true = np.array(["Low"] * 50 + ["High"] * 50)
    y_pred = y_true.copy()
    borough = pd.Series(["brooklyn"] * 100)

    result = check_borough_floor(y_true, y_pred, borough)

    assert result["brooklyn"]["macro_f1"] == 1.0
    assert result["brooklyn"]["margin"] > 0
    assert result["brooklyn"]["n"] == 100


def test_the_floor_is_per_borough_not_global() -> None:
    """A borough failing on its OWN distribution must fail even when the
    city-wide score is strong -- which is the averaging the gate exists to
    defeat."""
    strong = ["Low"] * 100
    y_true = np.array(strong + ["Low"] * 40 + ["Medium"] * 30)
    y_pred = np.array(strong + ["Very High"] * 70)
    borough = pd.Series(["manhattan"] * 100 + ["queens"] * 70)

    with pytest.raises(BoroughFloorError, match="queens"):
        check_borough_floor(y_true, y_pred, borough)
