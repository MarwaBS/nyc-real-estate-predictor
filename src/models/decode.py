"""The single conversion from a predicted price to a served zone.

The zone is a deterministic function of price -- ``cut(PRICE,
PRICE_ZONE_BINS)`` -- and training scores zones through this same function,
so the reported macro-F1 describes exactly what serving returns.
"""

from __future__ import annotations

import bisect

from src.config import PRICE_ZONE_BINS, PRICE_ZONE_LABELS

# Interior cut-points only: the bins are [0, q1, q2, q3, inf] and bisect needs
# the three thresholds that separate the four labels.
_CUTS = list(PRICE_ZONE_BINS[1:-1])


def zone_for_price(price: float) -> str:
    """The zone a price falls in.

    ``bisect_left`` places a price exactly on a cut-point in the LOWER zone,
    matching ``pd.cut``'s right-closed ``(a, b]`` intervals used to build the
    training labels. The cut-points are quantiles of the training prices, so
    exact ties do occur and train and serve must break them identically.
    """
    return PRICE_ZONE_LABELS[bisect.bisect_left(_CUTS, float(price))]
