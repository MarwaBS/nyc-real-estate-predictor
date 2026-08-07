"""FastAPI prediction service, /predict, /health, /docs."""

from __future__ import annotations

import hmac
import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
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
from src.config import CENTRAL_PARK, MANHATTAN_CENTER
from src.utils.geo import haversine

logger = logging.getLogger(__name__)

# Settings are constructed (and validated) at import time. A prod deploy with
# ALLOWED_ORIGINS="*" or unset raises ValueError here and the app fails to
# start, which is what we want. Dev/staging keep permissive defaults.
_settings = get_settings()

app = FastAPI(
    title="NYC Real Estate Price Prediction API",
    version="1.0.0",
    description="Predict price zones and property values for NYC real estate.",
)

# CORS, env-driven. Wildcard in prod is rejected at settings load-time.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.origins_list or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*", "X-API-Key"],
)

# Rate limit from settings (PREDICT_RATE_LIMIT env), wired to /predict below.
# /health stays unlimited so uptime probes are never throttled into flapping.
PREDICT_RATE_LIMIT = _settings.predict_rate_limit
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
# starlette types the handler's exc param as the base Exception; slowapi's
# handler narrows it to RateLimitExceeded, which is correct at runtime but
# trips mypy's invariant arg-type check. Scoped ignore, not blanket.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Auth dependency, optional X-API-Key header, enabled only when API_KEY is set
# ---------------------------------------------------------------------------
def _verify_api_key(x_api_key: str | None) -> None:
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
# Lazy model loading, keeps startup cheap for /health and /docs
# ---------------------------------------------------------------------------
_regressor: Any = None


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
    rooms_per_sqft = total_rooms / max(prop.propertysqft, 1.0)
    dist_manhattan = haversine(prop.latitude, prop.longitude, *MANHATTAN_CENTER)
    dist_central_park = haversine(prop.latitude, prop.longitude, *CENTRAL_PARK)

    row = {
        "BEDS": prop.beds,
        "BATH": prop.bath,
        "PROPERTYSQFT": prop.propertysqft,
        "TOTAL_ROOMS": total_rooms,
        "BED_BATH_RATIO": bed_bath_ratio,
        "ROOMS_PER_SQFT": rooms_per_sqft,
        "DIST_MANHATTAN_CENTER": dist_manhattan,
        "DIST_CENTRAL_PARK": dist_central_park,
        "BOROUGH": prop.borough.lower(),
        "TYPE": prop.type.lower(),
        "ZIPCODE": prop.zipcode,
        "SUBLOCALITY": prop.sublocality.lower(),
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(PREDICT_RATE_LIMIT)
def predict(
    request: Request,
    prop: PropertyInput,
    x_api_key: str | None = Header(default=None),
) -> PredictionResponse:
    """Predict price zone and estimated price for a property.

    Rate-limited per client IP at the configured ``PREDICT_RATE_LIMIT``;
    requests beyond it receive HTTP 429 from slowapi's handler. The rate is
    deployment-configurable, so no specific request count is quoted here -
    this docstring is rendered into the public /docs page, where a hardcoded
    number would misdescribe every deployment that overrides the default.
    The ``request`` parameter is required by slowapi's decorator contract.

    The API key is checked HERE, not via ``dependencies=[...]``: FastAPI
    resolves route dependencies before the limiter that decorates the endpoint,
    which would leave rejected keys uncounted and key brute-force unbounded.
    """
    _verify_api_key(x_api_key)
    try:
        # One inference path, shared with the dashboard: capping, rounding and
        # zoning live in src.models.predict so the surfaces cannot disagree.
        from src.models.predict import predict_listings

        record = predict_listings(_build_features(prop))[0]
        return PredictionResponse(
            zone=ZonePrediction(price_zone=record["price_zone"]),
            price=PricePrediction(
                predicted_price=record["predicted_price"],
                price_range=record["price_range"],
            ),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not yet trained. Run: make train",
        ) from exc
    except Exception:
        # Do NOT leak the exception message, it can disclose internal paths,
        # model-file names, or SQL fragments. logger.exception records the full
        # trace server-side; the client gets a generic message. `from None`
        # suppresses the exception chain for clean serialization.
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. See server logs.",
        ) from None


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """Health check, reports serving-stack availability. Not auth-gated.

    A failed probe returns **503**, not 200 with a false flag in the body.
    The status code must track whether the service can actually predict:
    every consumer (Dockerfile HEALTHCHECK, docker-compose, the Space start
    script, CI smoke test) checks the status and none reads the body. The
    price interval is probed alongside the model because a deployment missing
    ``price_interval.json`` would otherwise report healthy while every
    /predict returned 500.
    """
    models_loaded = False
    try:
        from src.models.predict import get_price_interval

        _get_regressor()
        get_price_interval()
        models_loaded = True
    except Exception:
        # Logged, not swallowed: a silent probe failure is how an unhealthy
        # container looks identical to a healthy one in the logs.
        logger.exception("Health probe: model or price interval failed to load")

    if not models_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if models_loaded else "degraded",
        models_loaded=models_loaded,
    )
