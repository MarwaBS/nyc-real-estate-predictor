"""Model drift detection, compare input feature distributions against baseline.

Tracks per-feature statistics (mean, std, min, max, percentiles) from training
data (``run_training.py`` writes ``models/drift_baseline.json`` at the end of
each run). ``check_drift`` is an OFFLINE utility for comparing a batch of
candidate data against that baseline, it is not wired into the serving path,
so nothing here runs at inference time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_feature_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute distribution statistics for each numeric feature."""
    stats: dict[str, dict[str, float]] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "count": int(len(series)),
        }

    return stats


def save_baseline(df: pd.DataFrame, path: Path) -> None:
    """Save feature distribution baseline from training data."""
    stats = compute_feature_stats(df)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Drift baseline saved to %s (%d features)", path, len(stats))
