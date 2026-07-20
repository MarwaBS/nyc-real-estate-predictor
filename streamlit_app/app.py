"""NYC Real Estate Price Prediction — Streamlit Dashboard."""

from __future__ import annotations

import math
import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from src.config import CENTRAL_PARK, MANHATTAN_CENTER, MODELS_DIR
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
def load_model():
    """Load the single regressor. Returns None on failure.

    One model: the zone is the predicted price bucketed through the same
    decode the API uses, so there is no classifier and no label encoder to
    keep in step with it.
    """
    import joblib

    try:
        return joblib.load(MODELS_DIR / "price_regressor_best.joblib")
    except FileNotFoundError:
        return None


def build_features(beds, bath, sqft, borough, prop_type, zipcode, lat, lon):
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
            st.error("Models not found. Train them first: `python run_training.py`")
        else:
            features = build_features(
                beds, bath, sqft, borough, prop_type, zipcode, latitude, longitude
            )

            # Mirror the train-time frequency cap (train/serve parity, same
            # as the API): rare/unseen SUBLOCALITY/ZIPCODE values map to
            # "other" so they get the trained "other" encoding instead of
            # the encoder's unseen default.
            from src.data.features import apply_serving_cap, learned_capped_categories

            features = apply_serving_cap(features, learned_capped_categories(reg))

            log_price = float(reg.predict(features)[0])
            price = math.expm1(log_price)

            # zone_for_price is the same decode the API uses, so both surfaces
            # answer identically for identical input.
            from src.models.decode import zone_for_price

            # No confidence figure: the zone is a bucketed point estimate, not
            # a classifier output, so there is no posterior to report.
            # Uncertainty is shown where it was calibrated -- the range below.
            st.metric("Price Zone", zone_for_price(price))
            # Round to $100 and band from the rounded figure, matching the API
            # so the displayed range reproduces from the displayed price.
            rounded = round(price, -2)
            st.metric("Estimated Price", f"${rounded:,.0f}")
            # Same calibrated interval the API serves, and labelled with the
            # coverage it was measured to achieve — a range without its
            # coverage invites the reader to assume a precision it lacks.
            from src.models.predict import get_price_interval, price_range

            band = price_range(rounded)
            target = get_price_interval()["target_coverage"]
            st.caption(
                f"{target:.0%} of listings fall in ${band['low']:,.0f} - "
                f"${band['high']:,.0f}"
            )

    else:
        st.info("Enter property details and click **Predict**.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Model: one Random Forest regressor | Data: NYC Housing Dataset (4,526 cleaned listings)"
)
