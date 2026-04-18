"""Model drift detection — compare input feature distributions against baseline.

Tracks per-feature statistics (mean, std, min, max, percentiles) from training data.
At inference time, flag drift when current distributions shift beyond thresholds.
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


def load_baseline(path: Path) -> dict[str, dict[str, float]]:
    """Load saved baseline."""
    with open(path, encoding="utf-8") as f:
        data: dict[str, dict[str, float]] = json.load(f)
    return data


def detect_drift(
    current_df: pd.DataFrame,
    baseline: dict[str, dict[str, float]],
    threshold: float = 0.15,
) -> dict[str, dict[str, float]]:
    """Compare current data against baseline. Returns drifted features.

    A feature is flagged as drifted if its mean shifts by more than
    `threshold` relative to the baseline standard deviation.

    Returns:
        dict of {feature: {"baseline_mean", "current_mean", "shift_ratio"}}
    """
    current_stats = compute_feature_stats(current_df)
    drifted: dict[str, dict[str, float]] = {}

    for feature, base in baseline.items():
        if feature not in current_stats:
            continue
        curr = current_stats[feature]
        base_std = base.get("std", 1.0)
        if base_std < 1e-8:
            base_std = 1.0  # Avoid division by zero

        shift = abs(curr["mean"] - base["mean"]) / base_std
        if shift > threshold:
            drifted[feature] = {
                "baseline_mean": base["mean"],
                "current_mean": curr["mean"],
                "shift_ratio": round(shift, 4),
            }

    if drifted:
        logger.warning("Drift detected in %d features: %s", len(drifted), list(drifted.keys()))
    else:
        logger.info("No drift detected (threshold=%.2f)", threshold)

    return drifted


def check_drift(
    current_df: pd.DataFrame,
    baseline_path: Path,
    threshold: float = 0.15,
    fail_on_drift: bool = False,
) -> dict[str, dict[str, float]]:
    """Load baseline and check for drift. Optionally raise on drift."""
    baseline = load_baseline(baseline_path)
    drifted = detect_drift(current_df, baseline, threshold)

    if drifted and fail_on_drift:
        raise ValueError(
            f"DATA DRIFT DETECTED in {len(drifted)} features: {list(drifted.keys())}"
        )

    return drifted
