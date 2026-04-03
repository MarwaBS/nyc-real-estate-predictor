"""FastAPI prediction service — /predict, /health, /docs."""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    HealthResponse,
    PredictionResponse,
    PricePrediction,
    PropertyInput,
    ZonePrediction,
)
from src.config import CENTRAL_PARK, MANHATTAN_CENTER, PRICE_ZONE_LABELS
from src.utils.geo import haversine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NYC Real Estate Price Prediction API",
    version="1.0.0",
    description="Predict price zones and property values for NYC real estate.",
)

# Lazy-loaded models (avoid startup crash if models not yet trained)
_classifier = None
_regressor = None


def _get_classifier():  # type: ignore[no-untyped-def]
    global _classifier
    if _classifier is None:
        from src.models.predict import get_classifier
        _classifier = get_classifier()
    return _classifier


def _get_regressor():  # type: ignore[no-untyped-def]
    global _regressor
    if _regressor is None:
        from src.models.predict import get_regressor
        _regressor = get_regressor()
    return _regressor


def _build_features(prop: PropertyInput) -> pd.DataFrame:
    """Transform a PropertyInput into the feature DataFrame the model expects."""
    total_rooms = prop.beds + prop.bath
    bed_bath_ratio = prop.beds / max(prop.bath, 1.0)
    log_sqft = math.log1p(prop.propertysqft)
    rooms_per_sqft = total_rooms / max(prop.propertysqft, 1.0)
    dist_manhattan = haversine(prop.latitude, prop.longitude, *MANHATTAN_CENTER)
    dist_central_park = haversine(prop.latitude, prop.longitude, *CENTRAL_PARK)

    row = {
        "BEDS": prop.beds,
        "BATH": prop.bath,
        "PROPERTYSQFT": prop.propertysqft,
        "TOTAL_ROOMS": total_rooms,
        "BED_BATH_RATIO": bed_bath_ratio,
        "LOG_SQFT": log_sqft,
        "ROOMS_PER_SQFT": rooms_per_sqft,
        "DIST_MANHATTAN_CENTER": dist_manhattan,
        "DIST_CENTRAL_PARK": dist_central_park,
        "DIST_NEAREST_SUBWAY": dist_manhattan,  # Proxy until subway data available
        "BOROUGH": prop.borough.lower(),
        "TYPE": prop.type.lower(),
        "PROPERTY_CATEGORY": "residential",  # Default
        "ZIPCODE": prop.zipcode,
        "SUBLOCALITY": prop.sublocality.lower(),
    }
    return pd.DataFrame([row])


@app.post("/predict", response_model=PredictionResponse)
def predict(prop: PropertyInput) -> PredictionResponse:
    """Predict price zone and estimated price for a property."""
    try:
        features = _build_features(prop)

        clf = _get_classifier()
        proba = clf.predict_proba(features)[0]
        zone_idx = int(np.argmax(proba))

        reg = _get_regressor()
        log_price = float(reg.predict(features)[0])
        price = math.expm1(log_price)

        return PredictionResponse(
            zone=ZonePrediction(
                price_zone=PRICE_ZONE_LABELS[zone_idx],
                confidence=round(float(proba.max()), 3),
                probabilities={
                    label: round(float(p), 3)
                    for label, p in zip(PRICE_ZONE_LABELS, proba, strict=False)
                },
            ),
            price=PricePrediction(
                predicted_price=round(price, -2),
                price_range={"low": round(price * 0.85, -2), "high": round(price * 1.15, -2)},
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(503, "Models not yet trained. Run: make train") from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"Prediction error: {exc}") from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — reports model availability."""
    models_ok = False
    try:
        _get_classifier()
        _get_regressor()
        models_ok = True
    except Exception:
        pass
    return HealthResponse(status="ok", models_loaded=models_ok)
