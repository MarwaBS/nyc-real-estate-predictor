"""Pydantic request/response models for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# The system's contract is these five boroughs. An unknown borough would
# one-hot encode to all zeros and still return a confident price.
VALID_BOROUGHS = frozenset(
    {"manhattan", "brooklyn", "queens", "the bronx", "staten island"}
)


class PropertyInput(BaseModel):
    """Input schema for a property prediction request."""

    beds: int = Field(ge=0, le=20, description="Number of bedrooms")
    bath: float = Field(ge=0, le=15, description="Number of bathrooms")
    propertysqft: float = Field(gt=0, le=50_000, description="Property size in sqft")
    borough: str = Field(
        description="NYC borough (manhattan, brooklyn, queens, the bronx, staten island)"
    )

    @field_validator("borough")
    @classmethod
    def _borough_must_be_one_of_the_five(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in VALID_BOROUGHS:
            raise ValueError(
                f"borough must be one of {sorted(VALID_BOROUGHS)}, got {v!r}"
            )
        return normalized

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
    """Price segment, derived from the predicted price.

    No ``confidence`` or ``probabilities``: the zone is a bucketing of a point
    estimate, not a classifier output, so there is no class posterior to
    report. Deriving a number from the interval and calling it confidence
    would be a figure nothing measured. Uncertainty is served where it was
    actually calibrated -- ``PricePrediction.price_range``.
    """

    price_zone: str


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
    at training time and documented in ``MODEL_CARD.md``.
    """

    zone: ZonePrediction
    price: PricePrediction


class HealthResponse(BaseModel):
    """Health check response.

    ``models_loaded`` is true only when the full serving stack loads: the
    regressor AND the calibrated price interval. Both are required to answer a
    prediction, so either one missing means the service cannot serve.
    """

    status: str
    models_loaded: bool
