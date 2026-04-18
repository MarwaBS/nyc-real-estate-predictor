# ── Hugging Face Spaces single-container image ──────────────────────────────
#
# Runs BOTH the FastAPI service (uvicorn on :8000, localhost-only) and the
# Streamlit dashboard (on :7860, the HF Spaces default port) inside one
# container. The Streamlit Predictor tab calls the local API via
# http://localhost:8000/predict so cache-aware behaviour stays consistent
# whether you hit the dashboard or the API directly.
#
# HF Spaces constraints respected:
#   1. Runs as non-root user (uid 1000 = "user", home /home/user)
#   2. Listens on $PORT (default 7860)
#   3. Writes only to /home/user or /tmp
#   4. No network calls at build time beyond pip install
#
# This Dockerfile is INDEPENDENT from the project-root multi-stage Dockerfile
# (used by docker-compose + GHCR). The project-root one is for production-
# grade local stack; this one is for the slim HF demo.

FROM python:3.12-slim-bookworm

# ── System deps + OS security patches ──────────────────────────────────────
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (HF Spaces convention) ────────────────────────────────────
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# ── Python deps ──────────────────────────────────────────────────────────────
# requirements.txt is the runtime-only set (pandas, numpy, sklearn, xgboost,
# lightgbm, fastapi, slowapi, streamlit, etc.) — produced by the Phase H
# split that separated runtime from training. Training-only deps (torch,
# mlflow, optuna, shap, catboost) are in requirements-train.txt and are
# NOT installed here — the Streamlit demo doesn't train.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Source + models ──────────────────────────────────────────────────────────
# api/, src/, streamlit_app/ are runtime code. models/ contains the trained
# .joblib + .json artifacts the API loads at first /predict. Models are
# baked INTO the image because HF Spaces free tier doesn't support volumes.
# To refresh models: re-run scripts.train_classification + train_regression
# locally, copy the new artifacts into ./models/, rebuild the Space.
COPY --chown=user:user api/             ./api/
COPY --chown=user:user src/             ./src/
COPY --chown=user:user streamlit_app/   ./streamlit_app/
COPY --chown=user:user models/          ./models/
COPY --chown=user:user deploy/huggingface/start.sh ./start.sh

RUN chmod +x ./start.sh

USER user

EXPOSE 7860

# Healthcheck targets the Streamlit port (Spaces routes traffic there).
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["./start.sh"]
