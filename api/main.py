"""FastAPI prediction service — /predict, /health, /docs."""

from __future__ import annotations

import hmac
import logging
import math
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
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

# Rate limiting — slowapi is a hard dependency, and the limit is actually
# WIRED to the prediction route below (a Limiter with no decorated routes
# enforces nothing). The limit comes from settings (DAILY_RATE_LIMIT env), not a
# hardcoded literal — previously `settings.daily_rate_limit` was dead config while
# this constant silently overrode it. /health stays unlimited: k8s/uptime probes
# must never be throttled into flapping.
PREDICT_RATE_LIMIT = _settings.daily_rate_limit
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
# starlette types the handler's exc param as the base Exception; slowapi's
# handler narrows it to RateLimitExceeded, which is correct at runtime but
# trips mypy's invariant arg-type check. Scoped ignore, not blanket.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


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


_label_encoder: Any = None


def _get_label_encoder() -> Any:
    """Load the shipped label encoder (cached). It is the source of truth for
    decoding class indices into zone names; /health verifies it can load
    because a missing encoder means predictions would be mislabeled, not
    merely absent."""
    global _label_encoder
    if _label_encoder is None:
        from src.models.predict import get_label_encoder

        _label_encoder = get_label_encoder()
    return _label_encoder


_zone_classes: list[str] | None = None


def _get_zone_classes() -> list[str]:
    """Zone names in the classifier's class-index order, from the SHIPPED
    label encoder (``models/label_encoder.joblib``). The encoder — not the
    config's semantic ``PRICE_ZONE_LABELS`` order — defines what class index
    ``i`` means; the two orders disagree for 3 of the 4 zones, so decoding
    via the config list mislabels most predictions."""
    global _zone_classes
    if _zone_classes is None:
        from src.models.predict import get_zone_classes

        _zone_classes = get_zone_classes()
    return _zone_classes


_thresholds: dict[str, float] | None = None
_thresholds_loaded: bool = False


def _get_thresholds() -> dict[str, float] | None:
    """Tuned per-class serving thresholds (cached), or ``None`` when the
    artifact is absent. The headline macro-F1 is measured under threshold
    decoding, so the API must serve the same decision rule — decoding with
    bare argmax here would mean the metric users read is never the metric
    they are served (and would diverge from the dashboard, which already
    decodes through ``served_zone``)."""
    global _thresholds, _thresholds_loaded
    if not _thresholds_loaded:
        from src.models.predict import get_thresholds

        _thresholds = get_thresholds()
        _thresholds_loaded = True
    return _thresholds


_capped_categories: dict[str, set] | None = None


def _get_capped_categories() -> dict[str, set]:
    """Cached map of {column: learned category set} for columns the model
    frequency-capped at train time, derived from the shipped classifier. Both the
    classifier and regressor are fit on the same capped frame, so one is the source
    of truth. Used to mirror the training cap at serve time (train/serve parity)."""
    global _capped_categories
    if _capped_categories is None:
        from src.data.features import learned_capped_categories

        _capped_categories = learned_capped_categories(_get_classifier())
    return _capped_categories


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
@limiter.limit(PREDICT_RATE_LIMIT)
def predict(request: Request, prop: PropertyInput) -> PredictionResponse:
    """Predict price zone and estimated price for a property.

    Rate-limited per client IP (``PREDICT_RATE_LIMIT``); the 61st request
    in a minute receives HTTP 429 from slowapi's handler. The ``request``
    parameter is required by slowapi's decorator contract.
    """
    try:
        features = _build_features(prop)
        # Mirror the train-time frequency cap: map rare/unseen SUBLOCALITY/ZIPCODE
        # to "other" so they get the trained "other" encoding rather than the
        # encoder's unseen default (train/serve parity).
        from src.data.features import apply_serving_cap

        features = apply_serving_cap(features, _get_capped_categories())

        clf = _get_classifier()
        proba = clf.predict_proba(features)[0]
        zone_classes = _get_zone_classes()

        # served_zone is the single serving decode for BOTH surfaces (API +
        # dashboard): per-class threshold logic when the tuned artifact is
        # shipped, argmax otherwise — and confidence is the probability of
        # the class actually served, not proba.max(), which in threshold
        # mode can describe a different class than the one returned.
        from src.models.threshold import served_zone

        zone_name, confidence = served_zone(proba, zone_classes, _get_thresholds())

        reg = _get_regressor()
        log_price = float(reg.predict(features)[0])
        price = math.expm1(log_price)

        return PredictionResponse(
            zone=ZonePrediction(
                price_zone=zone_name,
                confidence=round(confidence, 3),
                probabilities={
                    label: round(float(p), 3)
                    for label, p in zip(zone_classes, proba, strict=True)
                },
            ),
            price=PricePrediction(
                predicted_price=round(price, -2),
                price_range={
                    "low": round(price * 0.85, -2),
                    "high": round(price * 1.15, -2),
                },
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not yet trained. Run: make train",
        ) from exc
    except Exception:
        # Do NOT leak the exception message — it can disclose internal paths,
        # model-file names, or SQL fragments. logger.exception below records
        # the full trace server-side (there is no request-id pipeline in
        # this service — an earlier revision of this comment claimed one),
        # and the client gets a generic message. `from None` suppresses the
        # "During handling of the above exception" chain for clean
        # serialization (the original is captured by logger.exception).
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. See server logs.",
        ) from None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — reports serving-stack availability. Not auth-gated.

    The label encoder is verified EXPLICITLY, not just the classifier and
    regressor: it is the source of truth for decoding class indices into zone
    names, so a loadable clf/reg with a missing or unloadable encoder would
    still serve mislabeled zones. ``models_loaded`` is therefore true only when
    all three load; ``label_encoder_loaded`` surfaces the encoder on its own."""
    clf_reg_ok = False
    try:
        _get_classifier()
        _get_regressor()
        clf_reg_ok = True
    except Exception:
        pass

    encoder_ok = False
    try:
        _get_label_encoder()
        encoder_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        models_loaded=clf_reg_ok and encoder_ok,
        label_encoder_loaded=encoder_ok,
    )
