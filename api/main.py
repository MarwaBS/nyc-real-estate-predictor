"""FastAPI prediction service — /predict, /health, /docs."""
from __future__ import annotations

import hmac
import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.schemas import (
    HealthResponse,
    PredictionResponse,
    PricePrediction,
    PropertyInput,
    ZonePrediction,
)
from api.settings import get_settings
from src.config import CENTRAL_PARK, MANHATTAN_CENTER, PRICE_ZONE_LABELS
from src.utils.geo import haversine

logger = logging.getLogger(__name__)

# Settings are constructed (and validated) at import time. A prod deploy with
# ALLOWED_ORIGINS="*" or unset raises ValueError here and the app fails to
# start — which is what we want. Dev/staging keep permissive defaults.
_settings = get_settings()

app = FastAPI(
    title="NYC Real Estate Price Prediction API",
    version="1.0.0",
    description="Predict price zones and property values for NYC real estate.",
)

# CORS — env-driven. Wildcard in prod is rejected at settings load-time.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.origins_list or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*", "X-API-Key"],
)

# Rate limiting — slowapi is now a hard dependency; no conditional import.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Auth dependency — optional X-API-Key header, enabled only when API_KEY is set
# ---------------------------------------------------------------------------
async def verify_api_key(
    x_api_key: str | None = Header(default=None),
) -> None:
    """Timing-safe X-API-Key check.

    No-op when the process is started without an `API_KEY` env var (dev /
    portfolio mode). In that mode /predict is open, which is the existing
    behaviour and what tests expect. Set API_KEY in any non-dev deploy.
    """
    configured = _settings.api_key
    if not configured:
        return

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not hmac.compare_digest(x_api_key.encode(), configured.encode()):
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


# ---------------------------------------------------------------------------
# Lazy model loading — keeps startup cheap for /health and /docs
# ---------------------------------------------------------------------------
_classifier: Any = None
_regressor: Any = None


def _get_classifier() -> Any:
    global _classifier
    if _classifier is None:
        from src.models.predict import get_classifier
        _classifier = get_classifier()
    return _classifier


def _get_regressor() -> Any:
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
        "DIST_NEAREST_SUBWAY": dist_manhattan,  # proxy — see MODEL_CARD.md
        "BOROUGH": prop.borough.lower(),
        "TYPE": prop.type.lower(),
        "PROPERTY_CATEGORY": "residential",
        "ZIPCODE": prop.zipcode,
        "SUBLOCALITY": prop.sublocality.lower(),
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not yet trained. Run: make train",
        ) from exc
    except Exception:
        # Do NOT leak the exception message — it can disclose internal paths,
        # model-file names, or SQL fragments. Log the full trace server-side
        # (captured by the structured logger / request-id pipeline), return
        # a generic client-facing message. `from None` suppresses the
        # "During handling of the above exception" chain for clean
        # serialization (the original is captured by logger.exception).
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. See server logs.",
        ) from None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — reports model availability. Not auth-gated."""
    models_ok = False
    try:
        _get_classifier()
        _get_regressor()
        models_ok = True
    except Exception:
        pass
    return HealthResponse(status="ok", models_loaded=models_ok)
