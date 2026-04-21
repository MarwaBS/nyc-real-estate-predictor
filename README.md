# NYC Real Estate Price Prediction

[![CI](https://github.com/MarwaBS/nyc-real-estate-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/MarwaBS/nyc-real-estate-predictor/actions/workflows/ci.yml)

**Classify NYC properties into price zones and predict actual values using gradient boosting ensembles and multi-task deep learning on 4,500+ listings with geospatial features.**

> Every model is trained without data leakage. Previous R2=0.997 results were caused by PRICE_PER_SQFT (derived from target) — this has been removed and [documented as ADR-001](docs/decisions/001-remove-price-per-sqft.md).

> **Built on a published library we own.** The leakage-firewall logic that catches PRICE_PER_SQFT — and the broader bug classes documented in JAMA, *Nature Communications*, and the Kaggle Santander 2019 reveal — is extracted as a standalone package: [**`schema-firewall`** v0.1.0 on PyPI](https://pypi.org/project/schema-firewall/0.1.0/) ([source](https://github.com/MarwaBS/schema-firewall)). This repo pins `schema-firewall==0.1.0` in [`requirements.txt`](requirements.txt) and re-validates the integration in its `External Benchmark` CI job on every push. `pip install schema-firewall` works globally.

This repository contains **two separate evaluation surfaces** that should not be conflated:

| Surface | Data | Purpose | Primary result |
|---|---|---|---|
| Trained-model evaluation | Kaggle 2023 listings (4,504 rows; BEDS, BATH, LAT/LON, SUBLOCALITY) | Model quality on matched distribution | R² = 0.815 on 20% holdout |
| External benchmark | NYC.gov 2024 Rolling Sales (81,567 rows; no BEDS / BATH / LAT/LON) | Leakage-firewall + schema-lock behaviour under real-world distribution shift | Zero predictions by design — see [§External Benchmark](#external-benchmark--nycgov-2024) |

The external benchmark **is not a prediction system**. It is a controlled, versioned firewall that exposes what the trained model can and cannot be evaluated against once the data source changes. It is scoped this way deliberately — see that section for the full information-boundary statement.

---

## Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Price Zone (4-class) | **XGBoost + threshold tuning** | Macro F1 | **0.724** |
| Price Zone (4-class) | XGBoost (argmax) | Macro F1 | 0.704 |
| Price Zone (4-class) | LightGBM | Macro F1 | 0.693 |
| Price Zone (4-class) | Multi-Task DL (PyTorch) | Macro F1 | 0.670 |
| Price Regression | **XGBoost** | R2 (honest, no leakage) | **0.815** |
| Price Regression | Random Forest | R2 (honest, no leakage) | 0.804 |
| Price Regression | LightGBM | R2 (honest, no leakage) | 0.796 |
| Price Regression | Multi-Task DL (PyTorch) | R2 (honest, no leakage) | 0.724 |

All scores on held-out 20% stratified test set (3,603 train / 901 test). No data leakage.

Threshold tuning optimized per-class probability thresholds (Low=0.165, Medium=0.704, High=0.5, Very High=0.5), improving macro F1 from 0.704 to 0.724 (+0.020).

### SHAP feature importance (top 10)

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | DIST_MANHATTAN_CENTER | 0.212 |
| 2 | PROPERTYSQFT | 0.184 |
| 3 | BATH | 0.166 |
| 4 | SUBLOCALITY (target-encoded) | 0.148 |
| 5 | DIST_CENTRAL_PARK | 0.115 |
| 6 | ROOMS_PER_SQFT | 0.086 |
| 7 | TOTAL_ROOMS | 0.064 |
| 8 | ZIPCODE (target-encoded) | 0.044 |
| 9 | BEDS | 0.044 |
| 10 | TYPE_condo (one-hot) | 0.037 |

### Fairness by borough

| Borough | Macro F1 |
|---|---|
| Staten Island | 0.795 |
| Bronx | 0.680 |
| Brooklyn | 0.664 |
| Queens | 0.625 |
| Manhattan | 0.619 |

---

## External Benchmark — NYC.gov 2024

**What this is.** A schema-constrained, leakage-proof evaluation pipeline that runs the trained regressor against the City of New York's public Rolling Sales dataset.
**What this is not.** A production housing-price predictor. Not a model comparison. Not an accuracy claim under shift.
**Why results are structurally limited.** NYC.gov Rolling Sales 2024 is transaction data, not listings. It does not publish BEDS, BATH, LAT/LON, or SUBLOCALITY. The trained model (§Results) was built on 15 features that depend on those fields; they cannot be reconstructed from this data source.

### Layer separation (the whole point)

The benchmark is composed of three independent layers. Each has its own success condition. Conflating them is the most common misread.

**A. Data validity layer** — [SCHEMA_MAP.md](benchmarks/SCHEMA_MAP.md) v1 + [`benchmarks/mapping.py`](benchmarks/mapping.py)
Row-wise, stateless column mapping. Residential-only filter (`BUILDING CLASS CATEGORY` prefix `R*`). Structural drop rules (non-positive price, out-of-range price, missing sqft, missing year). Success = deterministic, auditable, target-independent; verified by the ten-test adversarial suite in [`tests/benchmarks/test_schema_firewall.py`](tests/benchmarks/test_schema_firewall.py) (18 tests incl. parametrisations, all green).

**B. Evaluation layer** — [`benchmarks/invariants.py`](benchmarks/invariants.py)
Four firewall checks: name-based leakage (`FORBIDDEN_COLUMNS`), semantic leakage (Pearson + Spearman + normalised MI), drop-log consistency, schema-SHA lock vs [`SCHEMA_MAP_VERSIONS.json`](benchmarks/SCHEMA_MAP_VERSIONS.json). Prediction-health checks (NaN, Inf, constant collapse) run only when the model produces output. Success = these invariants hold regardless of model performance.

**C. Benchmark layer** — [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py) + [`benchmarks/results.json`](benchmarks/results.json) + [`benchmarks/POSTMORTEMS.md`](benchmarks/POSTMORTEMS.md)
One-shot orchestration: download → map → invariant checks → attempt inference → write results. No tuning, no retry, no schema edits after seeing results. Success = reproducible, version-stamped output; the first-run number is the shipped number.

### Run 1 results (2026-04-20, SCHEMA_MAP v1, commit `00edb77`)

| | Value |
|---|---|
| Raw rows | 81,567 |
| Scored | **0** |
| Dropped | 81,567 |
| Leakage — name-based | not triggered |
| Leakage — semantic (Pearson + Spearman + MI) | not triggered |
| Schema SHA vs registry | match |
| Determinism | ✓ |
| Model inference | failed (schema mismatch: 15 expected vs 6 produced) |
| Prediction health | skipped (no predictions) |
| Performance | unobservable |

Drop reasons:

| Reason | Count | Share |
|---|---:|---:|
| `non_residential_class` | 50,799 | 62.3% |
| `sale_price_non_positive` | 27,704 | 34.0% |
| `sale_price_out_of_range` | 2,037 | 2.5% |
| `missing_gross_sqft` | 1,027 | 1.3% |

### Information-boundary statement

**Model performance is bounded by the observable features in NYC.gov 2024, not by model quality.** The trained regressor requires BEDS, BATH, distance-to-Manhattan-centre, distance-to-Central-Park, distance-to-nearest-subway, and SUBLOCALITY (target-encoded). NYC.gov Rolling Sales publishes none of these. Price prediction on this data source is partially underdetermined by information theory — cleaning, feature engineering, or model retraining cannot recover information that the data source does not contain.

The 62% `non_residential_class` rate reflects an **intentional domain restriction in SCHEMA_MAP v1** (condominiums only, `R*` prefix), not a modelling failure. Broadening scope to coops (`09`/`10`) and family dwellings (`01`/`02`/`03`) is a deliberate v2 decision under the versioning rule — it would invalidate current `results.json` and require a fresh benchmark run.

### What the benchmark validated on Run 1

| Claim | Status |
|---|---|
| Pipeline runs end-to-end on real public data | ✓ |
| No leakage detected (name-based + semantic) | ✓ |
| Drop-log consistent with row counts | ✓ |
| Schema-SHA lock held through the run | ✓ |
| Determinism preserved (same raw → same output) | ✓ |
| Failures surface as structured output, not crashes | ✓ |
| Model output free of NaN / Inf / collapse | not exercised (no predictions) |

Full narrative: [`benchmarks/POSTMORTEMS.md`](benchmarks/POSTMORTEMS.md). Reproduce locally: `python -m benchmarks.run_benchmark` (requires `pip install -r requirements.txt` + `openpyxl`).

### What will NOT change to "improve" this number

- `SCHEMA_MAP.md` v1 (locked by SHA in `SCHEMA_MAP_VERSIONS.json` and enforced by CI).
- Drop rules (changing them post-run = outcome shaping, not benchmarking).
- Model features (the trained model is the trained model; retraining is new-version work).
- Prediction-health thresholds (they are invariants, not performance gates).

These constraints are the point of the firewall. Loosening any of them converts this from a verifiable system into a tunable demo.

---

## Architecture

```
Raw CSV (4,800 rows)
    |
    v
src/data/cleaner.py         Dedupe, impute (borough-aware), cap outliers, normalize
    |
    v
src/data/features.py        Geospatial (haversine, H3, subway), numeric, target encoding
    |                       CRITICAL: no PRICE_PER_SQFT (data leakage guard)
    v
src/models/pipelines.py     sklearn Pipeline + ColumnTransformer (reproducible preprocessing)
    |
    +---> src/models/train_classification.py   XGBoost / LightGBM / CatBoost / Stacking
    |         Optuna Bayesian tuning (50 trials) + SMOTE-ENN for class imbalance
    |
    +---> src/models/train_regression.py       Same models, LOG_PRICE target
    |
    +---> src/dl/tabular_net.py                Multi-task PyTorch: shared trunk
    |         Entity embeddings + Focal Loss + CosineAnnealing + early stopping
    |
    +---> src/models/explain.py                SHAP (global + per-prediction) + fairness
    |
    v
api/main.py                 FastAPI — POST /predict, GET /health
streamlit_app/app.py        Interactive dashboard — NYC map + prediction form
```

### Key layers

| Layer | Files | Purpose |
|---|---|---|
| **Data pipeline** | `src/data/` | Load, clean, feature-engineer with validation gates |
| **Geospatial** | `src/utils/geo.py` | Haversine distances, H3 hexgrid, KMeans clusters, subway proximity |
| **ML models** | `src/models/` | sklearn Pipelines, Optuna tuning, stacking ensemble, SMOTE-ENN |
| **Deep learning** | `src/dl/` | Multi-task TabNet (PyTorch): classification + regression heads, Focal Loss |
| **Explainability** | `src/models/explain.py` | SHAP TreeExplainer, per-prediction explanations, fairness by borough |
| **API** | `api/` | FastAPI with Pydantic v2 schemas, health checks |
| **UI** | `streamlit_app/` | Interactive NYC map, prediction form, probability charts |
| **Validation** | `src/utils/validation.py` | Schema checks, `assert_no_leakage()` (enforced in CI) |

---

## Quick start

### Docker Compose (recommended)

```bash
cp .env.example .env          # fill in API keys
docker compose up --build     # starts API + Streamlit
```

- FastAPI API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`
- Swagger docs: `http://localhost:8000/docs`

### Manual setup

```bash
pip install -r requirements.txt

# Train models
python -m src.models.train_classification
python -m src.models.train_regression

# Start API
uvicorn api.main:app --reload --port 8000

# Start Streamlit
streamlit run streamlit_app/app.py
```

---

## API endpoints

```
POST /predict       Predict price zone + estimated price for a property
GET  /health        Liveness probe (reports model availability)
GET  /docs          Swagger UI (auto-generated)
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "beds": 2, "bath": 2.0, "propertysqft": 1200,
    "borough": "manhattan", "type": "condo", "zipcode": "10022",
    "latitude": 40.758, "longitude": -73.985
  }'
```

---

## Feature engineering

### Numerical (no target-derived features)

| Feature | Formula | Rationale |
|---|---|---|
| BEDS, BATH, PROPERTYSQFT | Raw | Core property attributes |
| TOTAL_ROOMS | BEDS + BATH | Combined room signal |
| BED_BATH_RATIO | BEDS / max(BATH, 1) | Layout balance |
| LOG_SQFT | log1p(SQFT) | Normalize right-skewed distribution |
| ROOMS_PER_SQFT | TOTAL_ROOMS / SQFT | Density metric |

### Geospatial (senior differentiator)

| Feature | Method | Library |
|---|---|---|
| DIST_MANHATTAN_CENTER | Haversine to (40.758, -73.985) | `geopy` / vectorized numpy |
| DIST_CENTRAL_PARK | Haversine to (40.783, -73.965) | `geopy` |
| DIST_NEAREST_SUBWAY | scipy cKDTree nearest-neighbor | NYC Open Data subway stations |
| H3_RES7 | Uber H3 hexagonal grid | `h3` |
| NEIGHBORHOOD_CLUSTER | KMeans(k=15) on lat/lon | `sklearn` |

### Categorical encoding

| Feature | Method | Why |
|---|---|---|
| BOROUGH, TYPE | OneHotEncoder | Low cardinality (5-8 values) |
| ZIPCODE, SUBLOCALITY | TargetEncoder (smoothing=10, fit per CV fold) | High cardinality (~150 ZIPs) — OneHot would create 150 sparse columns |

---

## Models

### Classification: Price Zone (Low / Medium / High / Very High)

| Model | Tuning | Class Imbalance |
|---|---|---|
| XGBoost | Optuna (50 trials, Bayesian) | scale_pos_weight |
| LightGBM | Optuna (50 trials) | class_weight="balanced" |
| CatBoost | Optuna (50 trials) | auto_class_weights="Balanced" |
| Random Forest | Baseline (no tuning) | class_weight="balanced_subsample" |
| Stacking Ensemble | XGB + LGBM + CatBoost -> LogisticRegression | Inherits from base models |

**Imbalance strategy:** SMOTE-ENN (oversampling + cleaning noisy synthetic samples) + Focal Loss in DL model.

### Regression: Actual Price

Same models as classification, predicting LOG_PRICE (log-transform stabilizes variance). Predictions converted back via `expm1()`.

### Deep Learning: Multi-Task TabNet

```
Numeric (10 feats) -> BatchNorm -> Dense(128)
Categorical        -> Entity Embeddings -> Dense(128)
                         |
                    Shared Trunk: 256 -> 128 -> 64 (BatchNorm + Dropout)
                         |
              +----------+----------+
              |                     |
     Classification Head      Regression Head
     Dense(4, Softmax)        Dense(1, Linear)
     Focal Loss               MSE Loss
              |                     |
              +-- Combined: 0.6*CE + 0.4*MSE --+
```

- Optimizer: AdamW (weight_decay=1e-4)
- LR scheduler: CosineAnnealingWarmRestarts(T_0=10)
- Early stopping: patience=15 epochs
- Gradient clipping: max_norm=1.0

---

## Explainability

- **SHAP summary plot**: Global feature importance (mean |SHAP|) — replaces `.feature_importances_`
- **SHAP waterfall**: Per-prediction explanation (which features drove this specific prediction)
- **SHAP dependence**: DIST_MANHATTAN_CENTER vs PRICE_ZONE (geographic price gradient)
- **Fairness analysis**: Macro F1 computed per borough to detect geographic bias
- **TabNet attention masks**: Which features the DL model focuses on per sample

---

## Testing

```bash
# Full test suite with coverage (gate: 70%)
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-fail-under=70

# Run only leakage prevention tests
pytest tests/test_no_leakage.py -v

# Run only API tests
pytest tests/test_api.py -v
```

**14 test files** covering:
- Data cleaning pipeline correctness
- Feature engineering (derived features, target creation, cardinality capping)
- **Data leakage prevention** — PRICE_PER_SQFT blocked in config AND validated at runtime
- Geospatial utilities (haversine, clusters)
- FastAPI endpoints (health, predict, validation errors)

CI runs 4 jobs: `lint` (ruff + mypy + bandit), `test` (pytest + 80% coverage gate), `security` (pip-audit + CycloneDX SBOM emission), `docker-build` (multi-stage build + Trivy HIGH/CRITICAL scan + `/health` smoke-run).

---

## Data leakage: why R2=0.997 was wrong

The original regression model used `PRICE_PER_SQFT` (= PRICE / PROPERTYSQFT) as a feature. Since the target is PRICE, this gives the model a near-perfect answer:

```
PRICE = PRICE_PER_SQFT * PROPERTYSQFT   # trivial algebra
```

R2=0.997 was not a real prediction — it was circular computation. After removing this feature:
- Honest R2 = **0.815** (XGBoost, best) — a real result, not inflated
- This is enforced by `test_no_leakage.py` in CI
- Documented in [ADR-001](docs/decisions/001-remove-price-per-sqft.md)

---

## Project structure

```
nyc-real-estate-predictor/
├── src/                          Core ML pipeline
│   ├── config.py                 All paths, constants, feature lists
│   ├── data/
│   │   ├── loader.py             Data I/O with dtype enforcement
│   │   ├── cleaner.py            Dedupe, impute, cap, normalize
│   │   └── features.py           Feature engineering + leakage guard
│   ├── models/
│   │   ├── pipelines.py          sklearn Pipeline + ColumnTransformer
│   │   ├── train_classification.py  XGBoost/LGBM/CatBoost + Optuna + stacking
│   │   ├── train_regression.py   Same models for LOG_PRICE
│   │   ├── evaluate.py           Metrics, confusion matrix, fairness
│   │   ├── explain.py            SHAP values + global importance
│   │   └── predict.py            Load model + inference
│   ├── dl/
│   │   ├── tabular_net.py        Multi-task PyTorch net + Focal Loss
│   │   └── train_dl.py           Training loop, early stopping, LR scheduler
│   └── utils/
│       ├── geo.py                Haversine, H3, KMeans, subway proximity
│       └── validation.py         Schema checks, assert_no_leakage()
│
├── api/                          FastAPI prediction service
│   ├── main.py                   POST /predict, GET /health
│   └── schemas.py                Pydantic v2 request/response models
│
├── streamlit_app/
│   └── app.py                    Interactive NYC map + prediction form
│
├── tests/                        14 test files, 70% coverage gate
│   ├── test_data_cleaner.py
│   ├── test_features.py
│   ├── test_no_leakage.py        DATA LEAKAGE PREVENTION (critical)
│   ├── test_geo.py
│   └── test_api.py
│
├── docs/decisions/               Architecture Decision Records
│   ├── 001-remove-price-per-sqft.md
│   ├── 002-xgboost-primary-model.md
│   └── 003-multi-task-deep-learning.md
│
├── notebooks/                    EDA + analysis (import from src/)
├── models/                       Serialized artifacts (gitignored)
├── .github/workflows/ci.yml      4-job CI: lint + test + security + docker-build
├── Dockerfile                    Non-root, health check
├── docker-compose.yml            API + Streamlit stack
├── requirements.txt
├── pyproject.toml                ruff + mypy + pytest config
└── Makefile                      make test / make train / make api
```

---

## Technology stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| DL | PyTorch 2.x (multi-task TabNet, Focal Loss, entity embeddings) |
| Tuning | Optuna (Bayesian optimization) |
| Explainability | SHAP |
| Geospatial | geopy, h3, scipy (cKDTree), KMeans |
| Encoding | category-encoders (TargetEncoder), sklearn OneHotEncoder |
| Imbalanced learning | imbalanced-learn (SMOTE-ENN) |
| API | FastAPI, Pydantic v2, Uvicorn |
| UI | Streamlit, Plotly |
| Testing | pytest (14 test files, 80% coverage gate) |
| Linting | ruff, mypy (strict), bandit |
| Infra | Docker (multi-stage, bookworm-tagged), docker-compose |
| CI | GitHub Actions: lint (ruff + mypy + bandit) + test (coverage gate) + security (pip-audit + CycloneDX SBOM) + docker-build (multi-stage build + Trivy HIGH/CRITICAL scan + smoke-run) |
| Supply chain | Dependabot (pip + docker + actions), Trivy, CycloneDX SBOM |

---

## Reproducibility

The training environment and the serving environment MUST run the **same** `scikit-learn` line. A silent prediction-corruption incident on 2026-04-19 (Manhattan condo predicted at $2) traced to a `scikit-learn==1.5.2` runtime loading a pickle produced under 1.8.0 — sklearn emitted `InconsistentVersionWarning` but the pipeline continued with corrupted internal state. Postmortem in [`MODEL_CARD.md`](MODEL_CARD.md#production-incidents-postmortem).

Exact pins — training + runtime are now identical:

| Library | Pinned version | Notes |
|---|---|---|
| Python | 3.12 | |
| scikit-learn | `==1.8.0` | MUST match; 1.5.x loads pickles as garbage without erroring |
| xgboost | `==2.1.2` | `.ubj` format is stable across 2.1.x patch releases |
| lightgbm | `==4.6.0` | bumped from 4.3.0 (PYSEC-2024-231) |
| category-encoders | `==2.8.1` | 2.6.x is incompatible with sklearn 1.8 (`_get_tags` removed) |
| numpy | `==1.26.4` | |
| pandas | `==2.2.3` | |

All pins live in [`requirements.txt`](requirements.txt) (serving) and [`requirements-train.txt`](requirements-train.txt) (training extras: Optuna, SHAP, imbalanced-learn). A rebuild 6 months from now pulls the exact same wheels. Dependabot is configured to PR updates; no pin changes without a full re-train + smoke-test cycle.

---

## License

MIT
