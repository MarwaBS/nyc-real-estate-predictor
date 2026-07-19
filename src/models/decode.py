"""The single decode from class probabilities to a served zone.

Both serving surfaces (the API and the Streamlit dashboard) call
``served_zone`` rather than each doing their own ``argmax``. They diverged
once before — the API decoded one way and the dashboard another, so the two
disagreed on the same input — and one shared function is what stops that.
"""

from __future__ import annotations

import numpy as np


def served_zone(proba: np.ndarray, labels: list[str]) -> tuple[str, float]:
    """Return ``(zone_name, confidence)`` for a single-row probability vector.

    ``labels`` is the zone name per class index, in the model's ``classes_``
    order; indexing it with anything else silently renames the prediction.
    """
    row = np.asarray(proba, dtype=float).ravel()
    zone_idx = int(np.argmax(row))
    return labels[zone_idx], float(row[zone_idx])
