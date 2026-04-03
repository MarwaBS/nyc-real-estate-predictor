# Price Prediction: Senior Data Scientist / ML+DL Engineer — Complete Build Plan

**Status:** Reference document — full architecture, models, algorithms, and implementation roadmap.
**Date:** 2026-04-03
**Current state:** 6 Jupyter notebooks, 4,504 cleaned rows, no .py modules, no tests, no deployment.
**Target state:** Production-grade ML+DL pipeline with API, explainability, monitoring, and CI/CD.

---

## 1. PROJECT DEFINITION

### 1.1 Problem Statement

Predict NYC real estate **price zones** (classification) and **actual prices** (regression) from property features + geospatial data. Build a production-grade ML system that a hiring manager would evaluate as Senior/Staff-level.

### 1.2 Target Variables

| Task | Target | Type | Classes / Range |
|---|---|---|---|
| **Primary: Price Zone Classification** | PRICE_ZONE | Multi-class (4) | Low ($0-500K), Medium ($500K-1M), High ($1M-2M), Very High ($2M+) |
| **Secondary: Price Regression** | PRICE | Continuous | $50K - $5M+ |
| **Tertiary: Property Size Classification** | SQFT_CATEGORY | Multi-class (3) | Small (<1000), Medium (1000-2000), Large (2000+) |

### 1.3 Current Data

| File | Rows | Columns | Notes |
|---|---|---|---|
| `Resources/NY-House-Dataset.csv` | 4,801 | 16 | Raw Kaggle dataset |
| `output/cleaned_house_dataset.csv` | 4,504 | 13 | Cleaned: LATITUDE, LONGITUDE, ADDRESS, ZIPCODE, SUBLOCALITY, TYPE, BEDS, BATH, PROPERTYSQFT, PRICE, BROKERTITLE, HOUSE_UNIT_LATLON_KEY, BOROUGH |
| `Resources/housing_geocode_extraction.csv` | 4,504 | — | Geoapify reverse geocoding results |
| `address_sublocality_map.csv` | 4,504 | — | Address to ZIP/sublocality mapping |

---

## 2. TARGET ARCHITECTURE

```
Price_Prediction/
├── src/
│   ├── __init__.py
│   ├── config.py                    # Paths, hyperparams, env vars, constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # Load raw + cleaned CSVs
│   │   ├── cleaner.py               # Cleaning pipeline (dedupe, impute, normalize)
│   │   ├── geocoder.py              # Geoapify/Google Maps with error handling + caching
│   │   └── features.py              # All feature engineering (geospatial, interactions, bins)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pipelines.py             # sklearn Pipeline + ColumnTransformer definitions
│   │   ├── train_classification.py  # Train price zone classifier (ML + DL)
│   │   ├── train_regression.py      # Train price regressor (ML + DL)
│   │   ├── evaluate.py              # Metrics, confusion matrix, reports, plots
│   │   ├── explain.py               # SHAP explainability
│   │   └── predict.py               # Load model + run inference
│   ├── dl/
│   │   ├── __init__.py
│   │   ├── tabular_net.py           # PyTorch TabNet / FNN with embeddings
│   │   ├── multitask_net.py         # Multi-task head (regression + classification)
│   │   └── train_dl.py              # Training loop, early stopping, LR scheduler
│   └── utils/
│       ├── __init__.py
│       ├── geo.py                   # Haversine, H3 hexgrid, subway proximity
│       ├── validation.py            # Data quality checks, schema enforcement
│       └── logging_config.py        # Structured logging
│
├── notebooks/                       # Thin orchestrators — import from src/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_ml_models.ipynb
│   ├── 05_dl_models.ipynb
│   └── 06_explainability.ipynb
│
├── api/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app — /predict, /health, /docs
│   ├── schemas.py                   # Pydantic request/response models
│   └── dependencies.py              # Load model at startup
│
├── streamlit_app/
│   └── app.py                       # Interactive dashboard — map + prediction form
│
├── models/                          # Serialized model artifacts (gitignored)
│   ├── price_zone_xgb_v1.joblib
│   ├── price_regressor_rf_v1.joblib
│   ├── preprocessor_v1.joblib
│   └── dl_multitask_v1.pt
│
├── tests/
│   ├── conftest.py                  # Fixtures, sample data
│   ├── test_data_cleaner.py
│   ├── test_features.py
│   ├── test_no_leakage.py           # Assert PRICE_PER_SQFT NOT in feature set
│   ├── test_model_quality.py        # Accuracy > threshold gates
│   ├── test_pipeline.py             # End-to-end: raw data -> prediction
│   └── test_api.py                  # FastAPI endpoint tests
│
├── data/                            # Raw data (gitignored, DVC-tracked)
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── architecture.md              # System design + data flow diagram
│   └── decisions/
│       ├── 001-xgboost-over-rf.md
│       ├── 002-no-price-per-sqft.md # Why we removed the leaky feature
│       └── 003-multi-task-dl.md
│
├── .github/workflows/
│   ├── ci.yml                       # lint + test + model quality gate
│   └── deploy.yml                   # Build + push Docker image on merge
│
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml                   # ruff + mypy + pytest config
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## 3. DATA PIPELINE (Phase 1)

### 3.1 Data Cleaning (`src/data/cleaner.py`)

| Step | Action | Detail |
|---|---|---|
| 1. Load | `pd.read_csv` with dtype enforcement | Explicit dtypes: BEDS=int, BATH=float, PRICE=float, ZIPCODE=str |
| 2. Deduplicate | Drop exact dupes + near-dupes by lat/lon | Haversine < 10m + same price = duplicate |
| 3. Missing values | Impute BEDS/BATH by median per BOROUGH | NOT global median — borough-aware |
| 4. Outlier removal | IQR x 3 on PRICE, PROPERTYSQFT, BEDS, BATH | Cap, don't drop (preserve sample size) |
| 5. Type standardize | Lowercase + strip all text columns | "Condo for sale" -> "condo" |
| 6. Borough normalize | Map county names to boroughs | "Kings County" -> "Brooklyn", "Richmond County" -> "Staten Island" |
| 7. ZIP normalize | Extract 5-digit ZIP, zero-pad | "10022.0" -> "10022" |
| 8. Validate | Assert no nulls in critical columns | PRICE > 0, PROPERTYSQFT > 0, valid lat/lon range |
| 9. Export | `data/processed/cleaned.csv` + manifest JSON | Row count, column list, timestamp, hash |

### 3.2 Geocoding (`src/data/geocoder.py`)

```python
# Key design decisions:
# 1. Cache API responses to JSON (avoid re-calling on rerun)
# 2. Async batch with aiohttp (100x faster than sequential)
# 3. Retry with exponential backoff on 5xx errors
# 4. API key from environment variable, NEVER hardcoded

import os
from dotenv import load_dotenv

load_dotenv()
GEOAPIFY_KEY = os.environ["GEOAPIFY_API_KEY"]
GOOGLE_MAPS_KEY = os.environ["GOOGLE_MAPS_API_KEY"]
```

### 3.3 Feature Engineering (`src/data/features.py`)

**CRITICAL: No PRICE_PER_SQFT in any feature set.** It is derived from the target variable and causes data leakage. The R2=0.997 from the current regression is fake because of this.

#### Numerical Features (no leakage)

| Feature | Formula | Rationale |
|---|---|---|
| BEDS | raw | Room count signal |
| BATH | raw | Luxury indicator |
| PROPERTYSQFT | raw | Primary size metric |
| TOTAL_ROOMS | BEDS + BATH | Combined room density |
| BED_BATH_RATIO | BEDS / max(BATH, 1) | Layout balance |
| LOG_SQFT | log1p(PROPERTYSQFT) | Normalize right-skewed distribution |
| ROOMS_PER_SQFT | TOTAL_ROOMS / PROPERTYSQFT | Density metric |

#### Geospatial Features (new — senior differentiator)

| Feature | Method | Library |
|---|---|---|
| DIST_MANHATTAN_CENTER | Haversine from (lat, lon) to (40.7580, -73.9855) | `geopy.distance` |
| DIST_CENTRAL_PARK | Haversine to (40.7829, -73.9654) | `geopy.distance` |
| DIST_NEAREST_SUBWAY | Min distance to subway station coordinates | NYC Open Data subway CSV + `scipy.spatial.cKDTree` |
| H3_INDEX_RES9 | Uber H3 hexagonal grid cell at resolution 9 | `h3` library |
| H3_INDEX_RES7 | Coarser grid for neighborhood-level signal | `h3` library |
| NEIGHBORHOOD_CLUSTER | KMeans(k=15) on (lat, lon) | `sklearn.cluster.KMeans` |

#### Categorical Features

| Feature | Encoding | Cardinality |
|---|---|---|
| BOROUGH | OneHot (5 values) | Low |
| TYPE | OneHot (top 8, rest "other") | Low |
| PROPERTY_CATEGORY | OneHot (4 values) | Low |
| ZIPCODE | Target encoding (mean price per ZIP on train fold only) | High (~150 unique) |
| SUBLOCALITY | Target encoding | High (~80 unique) |
| H3_INDEX_RES7 | Target encoding | High |

#### Target Variables

| Target | Derivation | Notes |
|---|---|---|
| PRICE_ZONE | pd.cut(PRICE, bins=[0, 500K, 1M, 2M, inf], labels=[Low, Medium, High, Very High]) | 4-class classification |
| LOG_PRICE | log1p(PRICE) | Regression target (log-transform stabilizes variance) |
| SQFT_CATEGORY | Small/Medium/Large by PROPERTYSQFT thresholds | 3-class classification |

---

## 4. MACHINE LEARNING MODELS (Phase 2)

### 4.1 Preprocessing Pipeline (`src/models/pipelines.py`)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder

NUMERIC_FEATURES = [
    "BEDS", "BATH", "PROPERTYSQFT", "TOTAL_ROOMS",
    "BED_BATH_RATIO", "LOG_SQFT", "ROOMS_PER_SQFT",
    "DIST_MANHATTAN_CENTER", "DIST_CENTRAL_PARK",
    "DIST_NEAREST_SUBWAY",
]

ONEHOT_FEATURES = ["BOROUGH", "TYPE", "PROPERTY_CATEGORY"]

TARGET_ENCODED_FEATURES = ["ZIPCODE", "SUBLOCALITY", "H3_INDEX_RES7"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat_onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT_FEATURES),
        ("cat_target", TargetEncoder(smoothing=10), TARGET_ENCODED_FEATURES),
    ],
    remainder="drop",
)
```

### 4.2 Classification Models — Price Zone

| Model | Library | Key Hyperparameters | Tuning Method |
|---|---|---|---|
| **XGBoost** (primary) | `xgboost.XGBClassifier` | max_depth=[3,5,7,10], n_estimators=[200,500,1000], learning_rate=[0.01,0.05,0.1], subsample=[0.7,0.8,1.0], colsample_bytree=[0.7,0.8,1.0], scale_pos_weight=auto | Optuna (50 trials, 5-fold stratified CV) |
| **LightGBM** | `lightgbm.LGBMClassifier` | num_leaves=[31,63,127], n_estimators=[200,500,1000], learning_rate=[0.01,0.05,0.1], min_child_samples=[5,10,20], class_weight="balanced" | Optuna (50 trials) |
| **CatBoost** | `catboost.CatBoostClassifier` | depth=[4,6,8,10], iterations=[500,1000], learning_rate=[0.01,0.05,0.1], l2_leaf_reg=[1,3,5,7] | Optuna (50 trials) |
| **Random Forest** (baseline) | `sklearn.ensemble.RandomForestClassifier` | n_estimators=[200,500], max_depth=[10,20,None], class_weight="balanced_subsample" | GridSearchCV |
| **Stacking Ensemble** | `sklearn.ensemble.StackingClassifier` | Base: XGB + LGBM + CatBoost. Meta: LogisticRegression(C=1.0) | 5-fold CV for base predictions |

**Evaluation metrics:**
- Primary: **Macro F1-score** (handles class imbalance)
- Secondary: Cohen's Kappa, per-class recall (especially "Very High" class)
- Confusion matrix + classification report

**Class imbalance strategy:**
1. SMOTE-ENN (better than SMOTE alone — removes noisy synthetic samples)
2. class_weight="balanced" in tree models
3. Focal loss in DL model (down-weights easy examples)
4. Threshold optimization post-training (per-class optimal thresholds)

### 4.3 Regression Models — Actual Price

| Model | Target | Key Difference from Current |
|---|---|---|
| **XGBoost Regressor** | LOG_PRICE | NO PRICE_PER_SQFT feature (fixes leakage) |
| **LightGBM Regressor** | LOG_PRICE | Native categorical support |
| **CatBoost Regressor** | LOG_PRICE | Best for mixed feature types |
| **Random Forest** (baseline) | LOG_PRICE | Same as current but without leaky feature |
| **Stacking Ensemble** | LOG_PRICE | Meta-learner: Ridge(alpha=1.0) |

**Expected honest R2 (without leakage):** 0.70-0.85 (not 0.997)

**Evaluation metrics:**
- RMSE, MAE, MAPE, R2
- Residual analysis (plot residuals vs predicted)
- Per-borough performance breakdown

### 4.4 Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Classification: Stratified 5-fold (preserves class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Regression: Standard 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# CRITICAL: Target encoding MUST be fit inside each fold (prevent leakage)
# Use sklearn Pipeline to ensure this automatically
```

**Hold-out test set:** 20% stratified split, NEVER used during tuning. Only evaluated once at the end.

### 4.5 Experiment Tracking

```python
import mlflow

mlflow.set_tracking_uri("mlruns")  # Local tracking (no server needed for portfolio)
mlflow.set_experiment("price_zone_classification")

with mlflow.start_run(run_name="xgb_v2_geospatial"):
    mlflow.log_params(best_params)
    mlflow.log_metrics({"macro_f1": 0.87, "accuracy": 0.92, "kappa": 0.84})
    mlflow.sklearn.log_model(pipeline, "model")
    mlflow.log_artifact("reports/confusion_matrix.png")
```

---

## 5. DEEP LEARNING MODELS (Phase 3 — Senior Differentiator)

### 5.1 Architecture: Multi-Task TabNet (`src/dl/multitask_net.py`)

```
Input Features
      │
      ├── Numeric (10 features) ──► BatchNorm ──► Dense(128, ReLU)
      │
      ├── BOROUGH (5) ──────────► Embedding(5, 8) ──┐
      ├── TYPE (8) ─────────────► Embedding(8, 12) ──┤
      ├── ZIPCODE (150) ────────► Embedding(150, 16) ┤
      ├── H3_INDEX (200) ───────► Embedding(200, 16) ┘
      │                                               │
      └── Concatenate all ◄────────────────────────────┘
                │
          Dense(256, ReLU) + Dropout(0.3)
          Dense(128, ReLU) + Dropout(0.2)
          Dense(64, ReLU)
                │
      ┌─────────┴──────────┐
      │                    │
  Classification Head   Regression Head
  Dense(4, Softmax)     Dense(1, Linear)
      │                    │
  CrossEntropy Loss    MSE Loss (on LOG_PRICE)
      │                    │
      └─────── Combined Loss = 0.6 * CE + 0.4 * MSE ───────┘
```

**Why multi-task:**
- Shared representation learns price structure that benefits both tasks
- Classification regularizes regression (prevents overfitting to outliers)
- Looks impressive in portfolio — shows DL architecture design skills

### 5.2 Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW (weight_decay=1e-4) | Better generalization than Adam |
| LR scheduler | CosineAnnealingWarmRestarts(T_0=10) | Smooth decay with periodic restarts |
| Batch size | 256 | Fits in memory, stable gradients |
| Epochs | 100 (early stopping patience=15) | Prevent overtraining |
| Loss weights | 0.6 classification + 0.4 regression | Classification is primary task |
| Focal loss alpha | [0.15, 0.25, 0.30, 0.30] per class | Higher weight for rare classes |
| Framework | PyTorch 2.x | Industry standard for custom architectures |

### 5.3 Alternative DL: TabNet

```python
from pytorch_tabnet.tab_model import TabNetClassifier

tabnet = TabNetClassifier(
    n_d=32, n_a=32,           # Decision/attention width
    n_steps=5,                 # Number of sequential attention steps
    gamma=1.5,                 # Coefficient for feature reusage
    lambda_sparse=1e-4,        # Sparsity regularization
    optimizer_fn=torch.optim.AdamW,
    scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    scheduler_params={"T_0": 10},
    mask_type="entmax",        # Sparse attention (better than softmax)
)
```

**Why TabNet:** Built-in feature importance via attention masks — directly comparable to SHAP for tree models. Shows DL interpretability awareness.

---

## 6. MODEL EXPLAINABILITY (Phase 4 — Senior Differentiator)

### 6.1 SHAP (for tree models)

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Global importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Per-prediction explanation
shap.force_plot(explainer.expected_value[0], shap_values[0][idx], X_test.iloc[idx])

# Dependence plot (feature interaction)
shap.dependence_plot("DIST_MANHATTAN_CENTER", shap_values[0], X_test)
```

### 6.2 What to Show in README

- SHAP summary plot (global feature importance — better than `.feature_importances_`)
- SHAP waterfall for a single prediction (shows reasoning per property)
- SHAP dependence: DIST_MANHATTAN_CENTER vs PRICE_ZONE (geographic price gradient)
- TabNet attention masks (which features the DL model focuses on per prediction)

### 6.3 Fairness Analysis

```python
# Check if model is biased by borough
for borough in ["Manhattan", "Brooklyn", "Queens", "The Bronx", "Staten Island"]:
    mask = X_test["BOROUGH"] == borough
    borough_f1 = f1_score(y_test[mask], y_pred[mask], average="macro")
    print(f"{borough}: F1 = {borough_f1:.3f}")
```

---

## 7. DEPLOYMENT (Phase 5)

### 7.1 FastAPI Prediction Service (`api/main.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI(title="NYC Price Prediction API", version="1.0.0")
pipeline = joblib.load("models/price_zone_xgb_v1.joblib")
regressor = joblib.load("models/price_regressor_rf_v1.joblib")

class PropertyInput(BaseModel):
    beds: int = Field(ge=0, le=20)
    bath: float = Field(ge=0, le=15)
    propertysqft: float = Field(gt=0, le=50000)
    borough: str
    type: str
    zipcode: str
    latitude: float = Field(ge=40.4, le=40.95)
    longitude: float = Field(ge=-74.3, le=-73.6)

class PredictionResponse(BaseModel):
    price_zone: str              # "Low" / "Medium" / "High" / "Very High"
    zone_confidence: float       # max probability
    zone_probabilities: dict     # all 4 class probabilities
    predicted_price: float       # USD
    price_range: dict            # {"low": ..., "high": ...} 80% confidence interval
    top_factors: list            # Top 3 SHAP features driving this prediction

@app.post("/predict", response_model=PredictionResponse)
def predict(property: PropertyInput):
    features = preprocess(property)
    zone_proba = pipeline.predict_proba([features])[0]
    price_log = regressor.predict([features])[0]
    price = math.expm1(price_log)
    shap_explanation = explain_single(features)
    return PredictionResponse(
        price_zone=ZONE_LABELS[zone_proba.argmax()],
        zone_confidence=round(float(zone_proba.max()), 3),
        zone_probabilities=dict(zip(ZONE_LABELS, zone_proba.round(3).tolist())),
        predicted_price=round(price, -2),
        price_range={"low": round(price * 0.85, -2), "high": round(price * 1.15, -2)},
        top_factors=shap_explanation[:3],
    )

@app.get("/health")
def health():
    return {"status": "ok"}
```

### 7.2 Streamlit Dashboard (`streamlit_app/app.py`)

**Features:**
- Interactive NYC map (Plotly Mapbox) — colored by price zone, click to predict
- Property input form — beds, bath, sqft, borough, type, address
- Prediction result — zone + confidence + price estimate
- SHAP waterfall — per-prediction explanation
- Model comparison table — XGBoost vs LightGBM vs CatBoost vs DL
- Feature importance (global SHAP summary)

### 7.3 Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.4 Docker Compose (full stack)

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  streamlit:
    build: .
    command: streamlit run streamlit_app/app.py --server.port=8501
    ports: ["8501:8501"]
    depends_on: [api]
```

---

## 8. TESTING & CI/CD (Phase 6)

### 8.1 Test Suite

| Test File | What It Tests | Key Assertions |
|---|---|---|
| `test_data_cleaner.py` | Cleaning pipeline | No nulls in critical cols, PRICE > 0, valid ZIP format |
| `test_features.py` | Feature engineering | Correct feature count, no NaN, derived features mathematically correct |
| `test_no_leakage.py` | **DATA LEAKAGE PREVENTION** | `assert "PRICE_PER_SQFT" not in pipeline.feature_names_in_`, `assert "price_per_sqft" not in X_train.columns` |
| `test_model_quality.py` | Model performance gates | Classification: macro_f1 > 0.80. Regression: R2 > 0.65 (honest, no leakage) |
| `test_pipeline.py` | End-to-end | Raw row -> prediction (no crash, valid output) |
| `test_api.py` | FastAPI endpoints | POST /predict returns valid schema, GET /health returns 200 |
| `test_geo.py` | Geospatial features | Haversine distance > 0, H3 index valid format |

### 8.2 CI Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff mypy
      - run: ruff check src/ api/ tests/
      - run: mypy src/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: "pip" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-fail-under=70

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install pip-audit
      - run: pip-audit -r requirements.txt
```

### 8.3 Model Quality Gate

```python
# tests/test_model_quality.py
import joblib
import pytest

@pytest.fixture
def trained_pipeline():
    return joblib.load("models/price_zone_xgb_v1.joblib")

def test_classification_macro_f1_above_threshold(trained_pipeline, test_data):
    X_test, y_test = test_data
    y_pred = trained_pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    assert f1 > 0.80, f"Macro F1 = {f1:.3f}, expected > 0.80"

def test_regression_r2_above_threshold(trained_regressor, test_data):
    X_test, y_test = test_data
    y_pred = trained_regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.65, f"R2 = {r2:.3f}, expected > 0.65 (honest, no leakage)"

def test_no_price_per_sqft_leakage(trained_pipeline):
    feature_names = trained_pipeline.feature_names_in_
    assert "price_per_sqft" not in [f.lower() for f in feature_names]
    assert "PRICE_PER_SQFT" not in feature_names
```

---

## 9. DEPENDENCIES (`requirements.txt`)

```
# Core
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
scipy>=1.13

# Gradient Boosting
xgboost>=2.1
lightgbm>=4.3
catboost>=1.2

# Deep Learning
torch>=2.3
pytorch-tabnet>=4.1

# Explainability
shap>=0.45

# Geospatial
geopy>=2.4
h3>=3.7
folium>=0.16

# Encoding
category-encoders>=2.6

# Imbalanced Learning
imbalanced-learn>=0.12

# Experiment Tracking
mlflow>=2.12

# API
fastapi>=0.111
uvicorn>=0.29
pydantic>=2.7

# UI
streamlit>=1.35
plotly>=5.22

# Testing
pytest>=8.2
pytest-cov>=5.0

# Linting
ruff>=0.4

# Config
python-dotenv>=1.0

# Hyperparameter Tuning
optuna>=3.6
```

---

## 10. README STRUCTURE (Target)

```markdown
# NYC Real Estate Price Prediction

[![CI](badge)](link)

> Predict NYC property price zones and values using gradient boosting ensembles
> and multi-task deep learning on 4,500+ listings with geospatial features.

## Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Price Zone (4-class) | XGBoost + SMOTE-ENN | Macro F1 | 0.87 |
| Price Zone (4-class) | Stacking Ensemble | Macro F1 | 0.89 |
| Price Zone (4-class) | Multi-Task TabNet | Macro F1 | 0.85 |
| Price Regression | XGBoost | R2 (no leakage) | 0.82 |
| Price Regression | Multi-Task TabNet | R2 (no leakage) | 0.79 |

## Architecture
[diagram]

## Quick Start
3 commands to run

## Feature Engineering
Table of all features with rationale

## Model Explainability
SHAP plots, TabNet attention, fairness analysis

## API Documentation
Endpoint table + example curl

## Tech Stack
Table with categories

## Testing
Commands + what's tested

## Deployment
Docker + cloud instructions
```

---

## 11. EXECUTION TIMELINE

| Phase | Tasks | Hours | Deliverables |
|---|---|---|---|
| **1. Emergency** | Rotate API keys, add .env, requirements.txt, .gitignore update | 1-2h | Secure repo |
| **2. Data Pipeline** | Extract cleaner.py, features.py, geocoder.py from notebooks. Add geospatial features (haversine, subway, H3). Remove PRICE_PER_SQFT from all features. | 6-8h | `src/data/`, processed data, manifest |
| **3. ML Models** | sklearn Pipeline, XGBoost/LightGBM/CatBoost, Optuna tuning, stacking, SMOTE-ENN. Stratified CV. MLflow tracking. | 8-10h | `src/models/`, serialized artifacts, MLflow runs |
| **4. DL Models** | PyTorch multi-task net with embeddings, TabNet, focal loss, training loop. | 6-8h | `src/dl/`, .pt checkpoint |
| **5. Explainability** | SHAP global + local, TabNet attention, fairness by borough, residual analysis. | 4-5h | `notebooks/06_explainability.ipynb`, plots |
| **6. Deployment** | FastAPI service, Streamlit dashboard, Docker, docker-compose. | 4-6h | `api/`, `streamlit_app/`, Dockerfile |
| **7. Testing + CI** | pytest suite (7 test files), GitHub Actions CI, coverage gate 70%. | 4-5h | `tests/`, `.github/workflows/ci.yml` |
| **8. Documentation** | README rewrite, architecture diagram, ADRs, clean git history. | 3-4h | README.md, `docs/` |
| **Total** | | **36-48h** | Production-grade ML+DL project |

---

## 12. KEY DECISIONS LOG

| Decision | Choice | Rejected Alternative | Why |
|---|---|---|---|
| Remove PRICE_PER_SQFT | Yes, from ALL feature sets | Keep it (current) | Data leakage — derived from target. R2=0.997 is fake. |
| Primary metric | Macro F1 | Accuracy | Accuracy misleads with imbalanced classes (Very High = 18 samples) |
| Regression target | LOG_PRICE | Raw PRICE | Log-transform stabilizes variance, improves tree splits |
| High-cardinality encoding | Target encoding (with CV) | OneHot | 150 ZIP codes -> 150 columns with OneHot. Target encoding = 1 column. |
| DL architecture | Multi-task (classification + regression) | Separate models | Shared representation, regularization, portfolio differentiator |
| Hyperparameter tuning | Optuna (Bayesian) | GridSearchCV | 10x fewer trials needed for same result |
| Class imbalance | SMOTE-ENN + focal loss | SMOTE only | SMOTE-ENN removes noisy synthetic samples; focal loss handles DL |
| Geospatial features | Haversine + H3 + subway distance | ZIP code only | Geographic granularity is the #1 predictor after removing leaky feature |
| Experiment tracking | MLflow (local) | None | Proves reproducibility; logs every run with params + metrics |
| Validation | Stratified 5-fold CV | Random 80/20 split | More robust estimate, preserves class distribution |
