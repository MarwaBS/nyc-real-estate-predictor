"""NYC Real Estate Price Prediction — Streamlit Dashboard."""
from __future__ import annotations

import math
import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st

from src.config import CENTRAL_PARK, MANHATTAN_CENTER, MODELS_DIR, PRICE_ZONE_LABELS
from src.utils.geo import haversine

st.set_page_config(
    page_title="NYC Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("NYC Real Estate Price Prediction")
st.markdown("Predict price zones and estimated values for NYC properties using ML + DL models.")


# ---------------------------------------------------------------------------
# Load models once (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load classifier + regressor + thresholds. Returns None on failure."""
    import joblib
    try:
        clf = joblib.load(MODELS_DIR / "price_zone_best.joblib")
        reg = joblib.load(MODELS_DIR / "price_regressor_best.joblib")
        thresholds = None
        thresh_path = MODELS_DIR / "optimal_thresholds.joblib"
        if thresh_path.exists():
            thresholds = joblib.load(thresh_path)
        return clf, reg, thresholds
    except FileNotFoundError:
        return None, None, None


def build_features(beds, bath, sqft, borough, prop_type, zipcode, lat, lon):
    """Build feature DataFrame from user input."""
    total_rooms = beds + bath
    bed_bath_ratio = beds / max(bath, 1.0)
    log_sqft = math.log1p(sqft)
    rooms_per_sqft = total_rooms / max(sqft, 1.0)
    dist_manhattan = haversine(lat, lon, *MANHATTAN_CENTER)
    dist_central_park = haversine(lat, lon, *CENTRAL_PARK)

    return pd.DataFrame([{
        "BEDS": beds,
        "BATH": bath,
        "PROPERTYSQFT": float(sqft),
        "TOTAL_ROOMS": total_rooms,
        "BED_BATH_RATIO": bed_bath_ratio,
        "LOG_SQFT": log_sqft,
        "ROOMS_PER_SQFT": rooms_per_sqft,
        "DIST_MANHATTAN_CENTER": dist_manhattan,
        "DIST_CENTRAL_PARK": dist_central_park,
        "DIST_NEAREST_SUBWAY": dist_manhattan,
        "BOROUGH": borough.lower(),
        "TYPE": prop_type.lower(),
        "PROPERTY_CATEGORY": "residential",
        "ZIPCODE": zipcode,
        "SUBLOCALITY": "unknown",
    }])


# ---------------------------------------------------------------------------
# Sidebar: property input form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Property Details")

    beds = st.number_input("Bedrooms", min_value=0, max_value=20, value=2)
    bath = st.number_input("Bathrooms", min_value=0.0, max_value=15.0, value=2.0, step=0.5)
    sqft = st.number_input("Square Footage", min_value=100, max_value=50_000, value=1_200)
    borough = st.selectbox("Borough", [
        "manhattan", "brooklyn", "queens", "the bronx", "staten island",
    ])
    prop_type = st.selectbox("Property Type", [
        "condo", "house", "co-op", "townhouse", "multi-family home",
    ])
    zipcode = st.text_input("ZIP Code", value="10022", max_chars=5)
    latitude = st.number_input("Latitude", min_value=40.4, max_value=40.95, value=40.758, format="%.6f")
    longitude = st.number_input("Longitude", min_value=-74.3, max_value=-73.6, value=-73.985, format="%.6f")

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
        clf, reg, thresholds = load_models()

        if clf is None or reg is None:
            st.error("Models not found. Train them first: `python run_training.py`")
        else:
            features = build_features(beds, bath, sqft, borough, prop_type, zipcode, latitude, longitude)

            # Classification
            proba = clf.predict_proba(features)[0]

            if thresholds is not None:
                from src.models.threshold import predict_with_thresholds
                zone_idx = int(predict_with_thresholds(
                    proba.reshape(1, -1), thresholds, PRICE_ZONE_LABELS,
                )[0])
            else:
                zone_idx = int(np.argmax(proba))

            zone_name = PRICE_ZONE_LABELS[zone_idx]
            confidence = float(proba.max())

            # Regression
            log_price = float(reg.predict(features)[0])
            price = math.expm1(log_price)

            # Display results
            st.metric("Price Zone", zone_name, f"{confidence:.0%} confidence")
            st.metric("Estimated Price", f"${price:,.0f}")
            st.caption(f"Range: ${price * 0.85:,.0f} - ${price * 1.15:,.0f}")

            # Probability chart — plotly preserves zone ordering (st.bar_chart sorts alphabetically and truncates "Very High")
            import plotly.express as px
            chart_df = pd.DataFrame({
                "Zone": PRICE_ZONE_LABELS,
                "Probability": [round(float(p), 3) for p in proba],
            })
            fig = px.bar(
                chart_df, x="Zone", y="Probability",
                category_orders={"Zone": PRICE_ZONE_LABELS},
            )
            fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 10, "b": 10}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enter property details and click **Predict**.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Models: XGBoost + LightGBM + CatBoost + Multi-Task DL | Data: NYC Housing Dataset (4,500+ listings)")
