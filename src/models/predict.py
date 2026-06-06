"""Inference — load a saved model and predict on new data."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PRICE_ZONE_LABELS

logger = logging.getLogger(__name__)

_classifier_cache: Any = None
_regressor_cache: Any = None


def _load_model(path: Path) -> Any:
    """Load a joblib-serialized model/pipeline."""
    logger.info("Loading model from %s", path)
    return joblib.load(path)


def get_classifier(path: Path | None = None) -> Any:
    """Load the best classifier (cached after first call)."""
    global _classifier_cache
    if _classifier_cache is None:
        _classifier_cache = _load_model(path or MODELS_DIR / "price_zone_best.joblib")
    return _classifier_cache


def get_regressor(path: Path | None = None) -> Any:
    """Load the best regressor (cached after first call)."""
    global _regressor_cache
    if _regressor_cache is None:
        _regressor_cache = _load_model(path or MODELS_DIR / "price_regressor_best.joblib")
    return _regressor_cache


def predict_price_zone(
    features: pd.DataFrame,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict price zone + probabilities for one or more properties.

    Returns a single dict when a single row is passed, a list of dicts when
    multiple rows are passed. This dual return shape is historical; new
    callers should prefer the single-row form.
    """
    clf = get_classifier()
    proba = clf.predict_proba(features)
    predicted_class = clf.predict(features)

    results = [
        {
            "price_zone": PRICE_ZONE_LABELS[int(zone_idx)],
            "confidence": round(float(row_proba.max()), 3),
            "probabilities": {
                label: round(float(p), 3) for label, p in zip(PRICE_ZONE_LABELS, row_proba)
            },
        }
        for zone_idx, row_proba in zip(predicted_class, proba)
    ]
    return results[0] if len(results) == 1 else results


def predict_price(
    features: pd.DataFrame,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict actual price (in USD) for one or more properties.

    Mirrors `predict_price_zone` — single dict for a single row, list for
    multiple. New callers should prefer the single-row form.
    """
    reg = get_regressor()
    prices = np.expm1(np.asarray(reg.predict(features), dtype=float))

    results = [
        {
            "predicted_price": round(price, -2),  # Round to nearest $100
            "price_range": {
                "low": round(price * 0.85, -2),
                "high": round(price * 1.15, -2),
            },
        }
        for price in prices.tolist()
    ]
    return results[0] if len(results) == 1 else results
