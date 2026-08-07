"""zone_for_price must bucket a price the way pd.cut labelled it for training.

The two disagree on exactly the three cut-points, so only a test that hits
them can tell bisect_left from bisect_right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PRICE_ZONE_BINS, PRICE_ZONE_LABELS
from src.models.decode import zone_for_price

CUT_POINTS = list(PRICE_ZONE_BINS[1:-1])


def _pandas_zone(prices: list[float]) -> list[str]:
    """The training labeller itself, as the reference, not a reimplementation."""
    return [
        str(z)
        for z in pd.cut(
            pd.Series(prices),
            bins=PRICE_ZONE_BINS,
            labels=PRICE_ZONE_LABELS,
            include_lowest=True,
        )
    ]


@pytest.mark.parametrize("cut", CUT_POINTS)
def test_a_price_on_a_cut_point_lands_in_the_lower_zone(cut: float) -> None:
    """pd.cut closes intervals on the right; bisect_right would serve the zone above."""
    assert zone_for_price(cut) == _pandas_zone([cut])[0]


def test_boundaries_agree_with_the_training_labeller() -> None:
    prices = [p for cut in CUT_POINTS for p in (cut - 1.0, cut, cut + 1.0)]
    assert [zone_for_price(p) for p in prices] == _pandas_zone(prices)


def test_the_whole_price_range_agrees() -> None:
    """Catches a wrong bin count or mis-ordered labels, which boundaries alone miss."""
    prices = list(np.linspace(1.0, 5_000_000.0, 2_000))
    assert [zone_for_price(p) for p in prices] == _pandas_zone(prices)


def test_a_price_above_the_top_cut_point_is_the_highest_zone() -> None:
    assert zone_for_price(500_000_000.0) == PRICE_ZONE_LABELS[-1]


def test_zero_is_the_lowest_zone() -> None:
    assert zone_for_price(0.0) == PRICE_ZONE_LABELS[0]
