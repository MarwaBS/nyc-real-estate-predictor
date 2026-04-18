---
title: NYC Real Estate Predictor
emoji: 🏙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: NYC price-zone classification + regression with XGBoost
---

# NYC Real Estate Price Predictor

[![GitHub](https://img.shields.io/badge/GitHub-MarwaBS/nyc--real--estate--predictor-181717?logo=github)](https://github.com/MarwaBS/nyc-real-estate-predictor)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](https://github.com/MarwaBS/nyc-real-estate-predictor/blob/main/MODEL_CARD.md)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://github.com/MarwaBS/nyc-real-estate-predictor/blob/main/api/main.py)

Live demo of an end-to-end ML service for NYC real estate. Pick a property profile in the **Predictor** tab and the dashboard POSTs to a co-located FastAPI service that returns a **price zone** (Low / Medium / High / Very High) plus an **estimated price** with a ±15% range.

**Two processes, one container:**
- **FastAPI** on `localhost:8000` — `/predict`, `/health`. Loads XGBoost classifier + regressor lazily on first request.
- **Streamlit** on `:7860` — the dashboard you see above. The Predictor tab calls `POST /predict` on the local API so behaviour stays consistent with direct API calls.

> **This is a portfolio demo, not a deployable real-estate predictor.** The model is trained on a 4,504-row Kaggle snapshot of NYC listings; it will drift against current-market reality. See the `MODEL_CARD.md` in the GitHub repo for the full framing, fairness analysis (per-borough disparity), and the data-leakage story (R²=0.997 → 0.815 honest).

## How the prediction works

1. Predictor tab collects property inputs (beds, bath, sqft, borough, type, zipcode, lat/long, sublocality).
2. Builds a `PropertyInput` JSON payload.
3. POSTs to `http://localhost:8000/predict` (the co-located FastAPI service).
4. The API runs `pipeline.engineer_features` → XGBoost zone classifier (with per-class threshold tuning) → XGBoost regressor on `LOG_PRICE`.
5. Returns the zone + confidence + per-class probabilities + predicted price + ±15% range.

## Honest about what this can and can't do

| ✅ Demonstrates | ❌ Does NOT |
|---|---|
| End-to-end ML pipeline (clean → feature-engineer → model → serve → UI) | Predict 2026 NYC prices accurately (data is a snapshot) |
| 4-model comparison (XGBoost / LightGBM / Random Forest / Multi-Task DL) | Beat Zillow Zestimate at scale |
| Data-leakage detection + ADR-001 documentation | Provide loan-grade pricing |
| Fairness-by-borough analysis (Manhattan F1=0.619 → Staten Island 0.795) | Mitigate the documented disparity |
| Per-class threshold tuning (+0.020 macro F1) | Online learning |

## Links

- **GitHub repo**: <https://github.com/MarwaBS/nyc-real-estate-predictor>
- **Model card**: <https://github.com/MarwaBS/nyc-real-estate-predictor/blob/main/MODEL_CARD.md>
- **Architecture decisions** (3 ADRs): <https://github.com/MarwaBS/nyc-real-estate-predictor/tree/main/docs/decisions>

## Notes on the live environment

- First load may take ~30s while uvicorn + Streamlit + XGBoost model files come up.
- HF Spaces free tier — no persistent state, no Redis, no rate-limit backend (slowapi falls back to in-memory).
- Models are baked into the image (see `deploy/huggingface/Dockerfile`); refresh requires rebuild.
