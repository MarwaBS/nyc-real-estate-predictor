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
from src.models.decode import zone_for_price

logger = logging.getLogger(__name__)

_regressor_cache: Any = None
_price_interval: dict[str, Any] | None = None


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


def get_regressor(path: Path | None = None) -> Any:
    """Load the best regressor (cached after first call)."""
    global _regressor_cache
    if _regressor_cache is None:
        _regressor_cache = _load_model(
            path or MODELS_DIR / "price_regressor_best.joblib"
        )
    return _regressor_cache


def get_price_interval() -> dict[str, Any]:
    """The calibrated price-interval multipliers (cached after first call).

    Load-bearing, so a missing artefact raises rather than falling back to a
    guess that would serve an interval nothing measured.
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

    One implementation for the API, predict module and dashboard, so the three
    cannot drift.
    """
    interval = get_price_interval()
    return {
        "low": round(price * float(interval["low_multiplier"]), -2),
        "high": round(price * float(interval["high_multiplier"]), -2),
    }


def predict_price_zone(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Zone per row, derived from the predicted price.

    There is no classifier. The zone is a bucketing of the price the regressor
    already predicts, so a second model would have been fitting the same
    features to the same signal -- and could disagree with the served price on
    the same listing. Training scores zones through this same decode, so the
    published macro-F1 describes what a caller actually receives.

    No ``probabilities`` key: a bucketed point estimate has no class posterior,
    and inventing one from the interval would be a confidence number nothing
    measured.
    """
    prices = np.expm1(np.asarray(get_regressor().predict(features), dtype=float))
    return [{"price_zone": zone_for_price(float(p))} for p in prices]


def predict_price(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Predict actual price (in USD) for one or more properties.

    Always returns a list with one entry per input row, mirroring
    :func:`predict_price_zone`.
    """
    reg = get_regressor()
    prices = np.expm1(np.asarray(reg.predict(features), dtype=float))

    # Derive the band from the rounded price, so low/high reproduce from the
    # figure shown beside them. They were multiplied from the unrounded price.
    out = []
    for price in prices.tolist():
        rounded = round(price, -2)
        out.append({"predicted_price": rounded, "price_range": price_range(rounded)})
    return out
