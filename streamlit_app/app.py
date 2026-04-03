"""NYC Real Estate Price Prediction — Streamlit Dashboard."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="NYC Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("NYC Real Estate Price Prediction")
st.markdown("Predict price zones and estimated values for NYC properties using ML + DL models.")

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
        try:
            import requests
            response = requests.post(
                "http://localhost:8000/predict",
                json={
                    "beds": beds, "bath": bath, "propertysqft": sqft,
                    "borough": borough, "type": prop_type, "zipcode": zipcode,
                    "latitude": latitude, "longitude": longitude,
                },
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                zone = data["zone"]
                price = data["price"]

                st.metric("Price Zone", zone["price_zone"], f"{zone['confidence']:.0%} confidence")
                st.metric("Estimated Price", f"${price['predicted_price']:,.0f}")
                st.caption(f"Range: ${price['price_range']['low']:,.0f} - ${price['price_range']['high']:,.0f}")

                # Probability chart
                proba_df = pd.DataFrame([zone["probabilities"]])
                st.bar_chart(proba_df.T.rename(columns={0: "Probability"}))
            else:
                st.error(f"API error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.warning("API not running. Start it with: `make api`")
            st.info("Showing placeholder results.")
            st.metric("Price Zone", "Medium", "Demo mode")
            st.metric("Estimated Price", "$750,000")
    else:
        st.info("Enter property details and click **Predict**.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Models: XGBoost + LightGBM + CatBoost + Multi-Task DL | Data: NYC Housing Dataset (4,500+ listings)")
