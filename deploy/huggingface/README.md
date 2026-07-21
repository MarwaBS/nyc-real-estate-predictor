---
title: NYC Real Estate Predictor
emoji: 🏙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: NYC price prediction + derived zones, one XGBoost model
---

# NYC Real Estate Price Predictor

[![GitHub](https://img.shields.io/badge/GitHub-MarwaBS/nyc--real--estate--predictor-181717?logo=github)](https://github.com/MarwaBS/nyc-real-estate-predictor)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](https://github.com/MarwaBS/nyc-real-estate-predictor/blob/main/MODEL_CARD.md)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://github.com/MarwaBS/nyc-real-estate-predictor/blob/main/api/main.py)

**Shipped model** — `XGBoost` — matched against the trained artefact by a CI gate.

Live demo of an end-to-end ML service for NYC real estate. Pick a property profile in the sidebar and the dashboard returns a **price zone** (Low / Medium / High / Very High) plus an **estimated price** with a range calibrated to contain 80% of listings (measured 77.9% on the held-out test split).

**Two processes, one container:**
- **Streamlit** on `:7860` — the dashboard you see above. It runs the prediction **in-process**, importing the same `src/` serving code the API uses.
- **FastAPI** on `localhost:8000` — `/predict`, `/health`. Started alongside the dashboard (the entrypoint waits for its `/health` before Streamlit comes up), but HF Spaces only exposes port 7860, so the API is **not reachable from outside this container**. To call the API yourself, run the repo's Docker image locally (see the GitHub README).

> **This is a portfolio demo, not a deployable real-estate predictor.** The model is trained on a 4,526-row Kaggle snapshot of NYC listings; it will drift against current-market reality. See the `MODEL_CARD.md` in the GitHub repo for the full framing, fairness analysis (per-borough disparity), and the data-leakage story (R²=0.997 leaked → 0.835 honest).

## How the prediction works

1. The sidebar collects property inputs (beds, bath, sqft, borough, type, zipcode, lat/long).
2. `build_features` derives the model's feature frame (room ratios, log-sqft, distances to Manhattan Center / Central Park), then the train-time frequency cap is mirrored via `apply_serving_cap` (train/serve parity).
3. The XGBoost regressor (trained on `LOG_PRICE`) produces the price estimate. The dashboard shows the calibrated interval from `models/price_interval.json` (0.677x-1.457x the point estimate), labelled with the coverage it was measured to achieve.
4. The zone is that estimate bucketed through `PRICE_ZONE_BINS`. There is one model and no classifier.

Both the dashboard and the API bucket through the same `zone_for_price` function, so what you see here is the decision rule the API serves.

## Honest about what this can and can't do

| ✅ Demonstrates | ❌ Does NOT |
|---|---|
| End-to-end ML pipeline (clean → feature-engineer → model → serve → UI) | Predict 2026 NYC prices accurately (data is a snapshot) |
| Multi-model comparison (XGBoost / LightGBM / Random Forest) | Beat Zillow Zestimate at scale |
| Data-leakage detection + ADR-001 documentation | Provide loan-grade pricing |
| Fairness-by-borough analysis (Staten Island F1=0.529 → Manhattan 0.696) | Mitigate the documented disparity |
| Val/test separation — selection on val, test scored once | Online learning |

## Links

- **GitHub repo**: <https://github.com/MarwaBS/nyc-real-estate-predictor>
- **Model card**: <https://github.com/MarwaBS/nyc-real-estate-predictor/blob/main/MODEL_CARD.md>
- **Architecture decisions** (3 ADRs): <https://github.com/MarwaBS/nyc-real-estate-predictor/tree/main/docs/decisions>

## Notes on the live environment

- **This Space is deployed automatically from `main`** by the repo's Deploy workflow; a weekly drift guard fails CI if the Space ever stops matching `main`. (It was previously hand-deployed — and served a 3-month-stale revision. Never again.)
- First load may take ~30s while uvicorn + Streamlit + the model artefacts come up.
- HF Spaces free tier — no persistent state, no Redis, no rate-limit backend (slowapi falls back to in-memory).
- **Served model provenance:** the artifacts in `models/` are the canonical 2026-07-19 training run (`run_date` in the repo's `reports/training_metrics.json` is authoritative; if it disagrees with this line, believe it), committed to the GitHub repo and pinned byte-for-byte by `models/MANIFEST.sha256` — the deploy workflow syncs them here and the weekly drift guard fails if this Space's code **or models** ever diverge from `main`. The metrics quoted above describe exactly these artifacts.
