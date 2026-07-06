"""Pydantic request/response models for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PropertyInput(BaseModel):
    """Input schema for a property prediction request."""

    beds: int = Field(ge=0, le=20, description="Number of bedrooms")
    bath: float = Field(ge=0, le=15, description="Number of bathrooms")
    propertysqft: float = Field(gt=0, le=50_000, description="Property size in sqft")
    borough: str = Field(
        description="NYC borough (manhattan, brooklyn, queens, the bronx, staten island)"
    )
    type: str = Field(
        description="Property type (condo, house, co-op, townhouse, etc.)"
    )
    zipcode: str = Field(pattern=r"^\d{5}$", description="5-digit ZIP code")
    latitude: float = Field(ge=40.4, le=40.95, description="Latitude (NYC range)")
    longitude: float = Field(ge=-74.3, le=-73.6, description="Longitude (NYC range)")
    sublocality: str = Field(
        default="unknown", description="Neighborhood / sublocality"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "beds": 2,
                    "bath": 2.0,
                    "propertysqft": 1200.0,
                    "borough": "manhattan",
                    "type": "condo",
                    "zipcode": "10022",
                    "latitude": 40.7580,
                    "longitude": -73.9855,
                    "sublocality": "midtown east",
                }
            ],
        }
    }


class ZonePrediction(BaseModel):
    """Price zone classification result."""

    price_zone: str
    confidence: float
    probabilities: dict[str, float]


class PricePrediction(BaseModel):
    """Price regression result."""

    predicted_price: float
    price_range: dict[str, float]


class PredictionResponse(BaseModel):
    """Combined prediction response.

    Explainability is intentionally **not** served per request: SHAP is a
    training-only dependency (kept out of the inference image to keep it
    lean, ~300 MB vs ~3 GB), and per-request explainers add latency for a
    signal that is stable globally. Global SHAP feature importance is computed
    at training time and documented in ``MODEL_CARD.md``. (A previous
    ``top_factors`` field defaulted to an empty list on every response and is
    removed rather than left as a hollow promise.)
    """

    zone: ZonePrediction
    price: PricePrediction


class HealthResponse(BaseModel):
    """Health check response.

    ``models_loaded`` is true only when the FULL serving stack loads: the
    classifier, the regressor, AND the label encoder. ``label_encoder_loaded``
    surfaces the encoder specifically — it is the source of truth for decoding
    class indices into zone names, so a loadable classifier/regressor with a
    missing encoder would still serve mislabeled zones.
    """

    status: str
    models_loaded: bool
    label_encoder_loaded: bool
