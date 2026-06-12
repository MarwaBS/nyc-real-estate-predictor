"""Inference — load a saved model and predict on new data.

Loading is version-guarded: a model trained under a different
scikit-learn version than the one running is REFUSED, not loaded with a
warning. This is the runtime fix for the postmortem'd incident where
sklearn 1.5.2 deserialised a 1.8.0-trained pipeline into garbage and the
pipeline kept serving ($2 Manhattan condos) — the failure mode is silent
corruption, so the guard must be a hard stop, not a log line.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from src.config import MODELS_DIR, PRICE_ZONE_LABELS

logger = logging.getLogger(__name__)

_classifier_cache: Any = None
_regressor_cache: Any = None


class ModelVersionError(RuntimeError):
    """A model artefact was produced by a different scikit-learn version.

    Raised instead of serving potentially-corrupt predictions. Retrain the
    artefact under the pinned scikit-learn version (requirements.txt) or
    align the runtime to the version that trained it.
    """


def _load_model(path: Path) -> Any:
    """Load a joblib-serialized model/pipeline, refusing version mismatches.

    ``InconsistentVersionWarning`` is promoted to an error: scikit-learn
    emits it when unpickling an estimator trained under another version,
    which is exactly the silent-corruption precondition documented in the
    MODEL_CARD postmortem.
    """
    logger.info("Loading model from %s", path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except InconsistentVersionWarning as exc:
            raise ModelVersionError(
                f"refusing to load {path.name}: {exc}. The artefact must be "
                f"retrained under the pinned scikit-learn version "
                f"(see requirements.txt) — loading across versions can "
                f"silently corrupt predictions."
            ) from exc


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


def predict_price_zone(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Predict price zone + probabilities for one or more properties.

    Always returns a list with one entry per input row — callers index
    ``[0]`` for the single-row case. (The historical single-row-returns-
    a-bare-dict shape made every caller branch on type.)
    """
    clf = get_classifier()
    proba = clf.predict_proba(features)
    predicted_class = clf.predict(features)

    return [
        {
            "price_zone": PRICE_ZONE_LABELS[int(zone_idx)],
            "confidence": round(float(row_proba.max()), 3),
            "probabilities": {
                label: round(float(p), 3)
                for label, p in zip(PRICE_ZONE_LABELS, row_proba, strict=False)
            },
        }
        for zone_idx, row_proba in zip(predicted_class, proba, strict=True)
    ]


def predict_price(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Predict actual price (in USD) for one or more properties.

    Always returns a list with one entry per input row, mirroring
    :func:`predict_price_zone`.
    """
    reg = get_regressor()
    prices = np.expm1(np.asarray(reg.predict(features), dtype=float))

    return [
        {
            "predicted_price": round(price, -2),  # Round to nearest $100
            "price_range": {
                "low": round(price * 0.85, -2),
                "high": round(price * 1.15, -2),
            },
        }
        for price in prices.tolist()
    ]
