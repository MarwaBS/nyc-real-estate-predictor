"""Inference, load a saved model and predict on new data.

Loading is version-guarded: a model trained under a different
scikit-learn version than the one running is REFUSED, not loaded with a
warning. This is the runtime fix for the postmortem'd incident where
sklearn 1.5.2 deserialised a 1.8.0-trained pipeline into garbage and the
pipeline kept serving ($2 Manhattan condos), the failure mode is silent
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
from src.data.features import apply_serving_cap, learned_capped_categories
from src.models.decode import zone_for_price

logger = logging.getLogger(__name__)

_regressor_cache: Any = None
_price_interval: dict[str, Any] | None = None


class ModelVersionError(RuntimeError):
    """A model artefact was produced by a different library version.

    Raised instead of serving potentially-corrupt predictions. Retrain the
    artefact under the pinned versions (requirements.txt) or align the
    runtime to the versions that trained it.
    """


# XGBoost has no InconsistentVersionWarning: unpickling a booster serialized
# by another version emits a plain UserWarning starting with this text. The
# trained-with version is not recoverable post-unpickle, so matching the
# warning is the only hook; if upstream rewords it the guard degrades to a
# warning again, the pinned CI never emits it either way.
_XGB_CROSS_VERSION = r".*If you are loading a serialized model"


def _load_model(path: Path) -> Any:
    """Load a joblib-serialized model/pipeline, refusing version mismatches.

    scikit-learn's ``InconsistentVersionWarning`` and XGBoost's serialized-
    model warning are both promoted to errors: each fires when unpickling an
    estimator trained under another version, which is exactly the
    silent-corruption precondition documented in the MODEL_CARD postmortem.
    """
    logger.info("Loading model from %s", path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        warnings.filterwarnings(
            "error", category=UserWarning, message=_XGB_CROSS_VERSION
        )
        try:
            return joblib.load(path)
        except (InconsistentVersionWarning, UserWarning) as exc:
            raise ModelVersionError(
                f"refusing to load {path.name}: {exc}. The artefact must be "
                f"retrained under the pinned library versions "
                f"(see requirements.txt), loading across versions can "
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
                f"{path} is missing, the served price interval is calibrated "
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


def predict_listings(features: pd.DataFrame) -> list[dict[str, Any]]:
    """The single inference path, the API and dashboard both call this.

    Applies the serving cap (so an unseen category gets the trained "other"
    encoding, not the encoder's unseen default), predicts once, and returns per
    row the rounded price, its calibrated band, and the zone that price falls
    in. One implementation, so no serving surface can drift on capping,
    rounding or zoning. The band is derived from the rounded price so low/high
    reproduce from the figure shown beside them; the zone is derived from the
    unrounded price. There is no classifier, the zone is the predicted price
    bucketed through the shared decode, so it cannot disagree with that price.
    """
    reg = get_regressor()
    features = apply_serving_cap(features, learned_capped_categories(reg))
    prices = np.expm1(np.asarray(reg.predict(features), dtype=float))
    records = []
    for price in prices.tolist():
        rounded = round(price, -2)
        records.append(
            {
                "predicted_price": rounded,
                "price_range": price_range(rounded),
                "price_zone": zone_for_price(price),
            }
        )
    return records
