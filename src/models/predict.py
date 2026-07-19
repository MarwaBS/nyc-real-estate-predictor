"""Inference — load a saved model and predict on new data.

Loading is version-guarded: a model trained under a different
scikit-learn version than the one running is REFUSED, not loaded with a
warning. This is the runtime fix for the postmortem'd incident where
sklearn 1.5.2 deserialised a 1.8.0-trained pipeline into garbage and the
pipeline kept serving ($2 Manhattan condos) — the failure mode is silent
corruption, so the guard must be a hard stop, not a log line.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

_classifier_cache: Any = None
_regressor_cache: Any = None
_label_encoder_cache: Any = None


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
        _regressor_cache = _load_model(
            path or MODELS_DIR / "price_regressor_best.joblib"
        )
    return _regressor_cache


def get_label_encoder(path: Path | None = None) -> Any:
    """Load the label encoder fitted at training time (cached after first call).

    This is the single source of truth for decoding class indices into zone
    names: the classifier was fit on ``LabelEncoder``-transformed targets, so
    class index ``i`` means ``label_encoder.classes_[i]`` — an ALPHABETICAL
    ordering ('High', 'Low', 'Medium', 'Very High'), not the semantic
    ``PRICE_ZONE_LABELS`` config order. Decoding through any other list is
    exactly the bug that served "Low" for luxury Manhattan condos.
    """
    global _label_encoder_cache
    if _label_encoder_cache is None:
        _label_encoder_cache = _load_model(path or MODELS_DIR / "label_encoder.joblib")
    return _label_encoder_cache


def get_zone_classes() -> list[str]:
    """Zone names in the classifier's class-index order (encoder ``classes_``)."""
    return [str(c) for c in get_label_encoder().classes_]


_price_interval: dict[str, Any] | None = None


def get_price_interval() -> dict[str, Any]:
    """The calibrated price-interval multipliers (cached after first call).

    Load-bearing, so a missing artefact raises rather than falling back to a
    guess: the previous behaviour was a hardcoded +/-15% that contained the
    true price 32% of the time, and silently substituting any default here
    would reintroduce an interval nothing measured.
    """
    global _price_interval
    if _price_interval is None:
        path = MODELS_DIR / "price_interval.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing — the served price interval is calibrated "
                f"during training. Run: python run_training.py"
            )
        _price_interval = json.loads(path.read_text(encoding="utf-8"))
    return _price_interval


def price_range(price: float) -> dict[str, float]:
    """The interval served alongside ``price``, from the calibrated artefact.

    One implementation for every surface (API, predict module, dashboard) so
    the three cannot drift — they previously each hardcoded the same literal.
    """
    interval = get_price_interval()
    return {
        "low": round(price * float(interval["low_multiplier"]), -2),
        "high": round(price * float(interval["high_multiplier"]), -2),
    }


def predict_price_zone(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Predict price zone + probabilities for one or more properties.

    Always returns a list with one entry per input row — callers index
    ``[0]`` for the single-row case. (The historical single-row-returns-
    a-bare-dict shape made every caller branch on type.)
    """
    clf = get_classifier()
    proba = clf.predict_proba(features)
    predicted_class = clf.predict(features)
    # Decode through the SHIPPED label encoder, never the config list: the
    # model's class indices follow le.classes_ (alphabetical), and the two
    # orders disagree for 3 of the 4 zones.
    classes = get_zone_classes()

    return [
        {
            "price_zone": classes[int(zone_idx)],
            "confidence": round(float(row_proba.max()), 3),
            "probabilities": {
                label: round(float(p), 3)
                for label, p in zip(classes, row_proba, strict=True)
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
            "price_range": price_range(price),
        }
        for price in prices.tolist()
    ]
