"""NYC Real Estate Price Prediction — Streamlit Dashboard."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from src.config import CENTRAL_PARK, MANHATTAN_CENTER
from src.utils.geo import haversine

st.set_page_config(
    page_title="NYC Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("NYC Real Estate Price Prediction")
st.markdown(
    "Predict NYC property prices with a calibrated range, plus the price zone that estimate falls in."
)


# ---------------------------------------------------------------------------
# Load models once (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> Any:
    """Load the regressor through the API's guarded loader, so a cross-version
    artefact is refused here (as the API refuses it) instead of loading and
    raising at predict time. Returns None when the model is absent or rejected;
    the caller shows one "unavailable" message for both.
    """
    from src.models.predict import ModelVersionError, get_regressor

    try:
        return get_regressor()
    except (FileNotFoundError, ModelVersionError):
        return None


def build_features(
    beds: float,
    bath: float,
    sqft: float,
    borough: str,
    prop_type: str,
    zipcode: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """Build feature DataFrame from user input."""
    total_rooms = beds + bath
    bed_bath_ratio = beds / max(bath, 1.0)
    rooms_per_sqft = total_rooms / max(sqft, 1.0)
    dist_manhattan = haversine(lat, lon, *MANHATTAN_CENTER)
    dist_central_park = haversine(lat, lon, *CENTRAL_PARK)

    return pd.DataFrame(
        [
            {
                "BEDS": beds,
                "BATH": bath,
                "PROPERTYSQFT": float(sqft),
                "TOTAL_ROOMS": total_rooms,
                "BED_BATH_RATIO": bed_bath_ratio,
                "ROOMS_PER_SQFT": rooms_per_sqft,
                "DIST_MANHATTAN_CENTER": dist_manhattan,
                "DIST_CENTRAL_PARK": dist_central_park,
                "BOROUGH": borough.lower(),
                "TYPE": prop_type.lower(),
                "ZIPCODE": zipcode,
                # No dashboard field for it: the encoder folds an unseen value
                # into "other", so this surface predicts without a shipped
                # feature the API accepts.
                "SUBLOCALITY": "unknown",
            }
        ]
    )


# ---------------------------------------------------------------------------
# Sidebar: property input form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Property Details")

    beds = st.number_input("Bedrooms", min_value=0, max_value=20, value=2)
    bath = st.number_input(
        "Bathrooms", min_value=0.0, max_value=15.0, value=2.0, step=0.5
    )
    sqft = st.number_input(
        "Square Footage", min_value=100, max_value=50_000, value=1_200
    )
    borough = st.selectbox(
        "Borough",
        [
            "manhattan",
            "brooklyn",
            "queens",
            "the bronx",
            "staten island",
        ],
    )
    prop_type = st.selectbox(
        "Property Type",
        [
            "condo",
            "house",
            "co-op",
            "townhouse",
            "multi-family home",
        ],
    )
    zipcode = st.text_input("ZIP Code", value="10022", max_chars=5)
    latitude = st.number_input(
        "Latitude", min_value=40.4, max_value=40.95, value=40.758, format="%.6f"
    )
    longitude = st.number_input(
        "Longitude", min_value=-74.3, max_value=-73.6, value=-73.985, format="%.6f"
    )

    predict_btn = st.button("Predict", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Location")
    map_data = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
    st.map(map_data, zoom=12)

with col2:
    st.subheader("Prediction Results")

    if predict_btn:
        reg = load_model()

        if reg is None:
            st.error("Model unavailable — train it first: `python run_training.py`")
        else:
            features = build_features(
                beds, bath, sqft, borough, prop_type, zipcode, latitude, longitude
            )

            # One inference path, shared with the API (src.models.predict):
            # capping, rounding and zoning live in one place so the surfaces
            # cannot disagree. No confidence figure -- the zone is a bucketed
            # point estimate; uncertainty is the calibrated range below.
            from src.models.predict import get_price_interval, predict_listings

            record = predict_listings(features)[0]
            st.metric("Price Zone", record["price_zone"])
            st.metric("Estimated Price", f"${record['predicted_price']:,.0f}")
            band = record["price_range"]
            interval = get_price_interval()
            target = interval["target_coverage"]
            measured = interval["coverage_test"]
            st.caption(
                f"Calibrated for {target:.0%} coverage; measured "
                f"{measured:.1%} on the test split. "
                f"${band['low']:,.0f} - ${band['high']:,.0f}"
            )

    else:
        st.info("Enter property details and click **Predict**.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")


def _footer_facts() -> str:
    """Model name and cleaned-listing count read from the committed training
    artefact, so the footer tracks the shipped run instead of being hand-kept."""
    try:
        metrics = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "reports"
                / "training_metrics.json"
            ).read_text(encoding="utf-8")
        )
        model = str(metrics["regression"]["selected_model"]).upper()
        p = metrics["provenance"]
        n = p["n_train"] + p["n_val"] + p["n_test"]
        return f"Model: one {model} regressor | Data: NYC Housing Dataset ({n:,} cleaned listings)"
    except Exception:
        return "Model: one regressor over the NYC Housing Dataset"


st.caption(_footer_facts())
