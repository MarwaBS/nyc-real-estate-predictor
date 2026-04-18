# Hugging Face Spaces deployment guide

Mirrors the `high-pay-salary-predictor` deploy pattern. ~10 minutes total.

## Prerequisites

1. A Hugging Face account. Sign up free at <https://huggingface.co/join> if you don't have one.
2. Git installed locally.
3. A local clone of this project (referred to below as `<PROJECT_DIR>`).
4. Trained model artifacts in `<PROJECT_DIR>/models/` (`price_zone_best.joblib`, `price_regressor_best.joblib`, `optimal_thresholds.joblib`, `label_encoder.joblib`, `drift_baseline.json`). Run `make train` if missing.

## Step 1 — Create a Hugging Face access token

You need a **write-scope** token to push to the Space repo.

1. Go to <https://huggingface.co/settings/tokens>.
2. Click **New token** → name `nyc-real-estate-deploy` → role `Write` → **Generate**.
3. Copy the token (starts with `hf_`). Save somewhere private — you'll only see it once.

## Step 2 — Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Owner: `MarwaBS` (your username).
3. Space name: `nyc-real-estate-predictor` (matches the GitHub repo name exactly).
4. License: `mit`.
5. Space SDK: **Docker**.
6. Hardware: **CPU basic (free)** — more than enough for XGBoost inference.
7. Visibility: **Public**.
8. Click **Create Space**.

## Step 3 — Clone the empty Space repo

```bash
git clone https://huggingface.co/spaces/MarwaBS/nyc-real-estate-predictor hf-space-nyc
cd hf-space-nyc
```

The clone has only `.gitattributes` and `README.md` (the default HF README — about to be replaced).

## Step 4 — Copy the deployment files into the Space

From `hf-space-nyc/`, copy from your local project clone:

```bash
# Paths assume you're in the hf-space-nyc directory
PROJECT="<PROJECT_DIR>"   # substitute the absolute path to your local clone

# 1. HF-specific files — these replace the Space defaults
cp "$PROJECT/deploy/huggingface/Dockerfile"      ./Dockerfile
cp "$PROJECT/deploy/huggingface/README.md"       ./README.md
cp "$PROJECT/deploy/huggingface/start.sh"        ./start.sh
chmod +x ./start.sh

# 2. Python source + config
cp -r "$PROJECT/api"            ./api
cp -r "$PROJECT/src"            ./src
cp -r "$PROJECT/streamlit_app"  ./streamlit_app

# 3. Trained models — required at runtime
mkdir -p ./models
cp "$PROJECT/models/"*.joblib   ./models/
cp "$PROJECT/models/"*.json     ./models/

# 4. Runtime requirements (the Phase-H slim set, NOT requirements-train.txt)
cp "$PROJECT/requirements.txt"  ./requirements.txt
```

Verify the file tree:

```bash
ls -la
# expected: Dockerfile, README.md, start.sh, api/, src/, streamlit_app/, models/, requirements.txt
```

## Step 5 — Push to the Space

```bash
git add .
git commit -m "Initial deployment: FastAPI + Streamlit single container"
git push
# When prompted for password, paste the HF token from Step 1
```

## Step 6 — Watch the build

1. Go to <https://huggingface.co/spaces/MarwaBS/nyc-real-estate-predictor>.
2. Click the **Logs** tab. The build takes ~3-5 min on first push (pip install + Docker layers).
3. Once green, the **App** tab shows the Streamlit dashboard.

## Step 7 — Update the GitHub repo README

Add the live demo badge to the GitHub repo's main `README.md`:

```markdown
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-on%20Hugging%20Face-yellow)](https://huggingface.co/spaces/MarwaBS/nyc-real-estate-predictor)
```

Commit + push to GitHub.

## Updating the Space after code changes

```bash
cd hf-space-nyc
# Sync the changed files from the main project
cp "<PROJECT_DIR>/api/main.py" ./api/main.py
# (or whatever changed)
git commit -am "Update X"
git push
# HF rebuilds automatically (~30 s for code-only changes, ~3 min if deps changed)
```

If you re-trained models, copy the new `models/` artifacts over and push.

## Rollback

Each push is a Space revision. Visit the Space's **Settings → Revisions** tab to roll back to a prior commit SHA.

## Cost

Free tier (CPU basic). No charge for the demo. If you upgrade to GPU or A10G later for higher throughput, costs apply per the HF pricing page.
