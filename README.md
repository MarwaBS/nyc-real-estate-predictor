# NYC Real Estate Price Prediction

[![CI](https://github.com/MarwaBS/nyc-real-estate-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/MarwaBS/nyc-real-estate-predictor/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-on%20Hugging%20Face-yellow)](https://huggingface.co/spaces/MarwaBS/nyc-real-estate-predictor)

**Classify NYC properties into price zones and predict actual values using gradient boosting ensembles and multi-task deep learning on 4,500+ listings with geospatial features.**

> Every model is trained without data leakage. Previous R2=0.997 results were caused by PRICE_PER_SQFT (derived from target) — this has been removed and [documented as ADR-001](docs/decisions/001-remove-price-per-sqft.md).

> **Built on a published library we own.** The leakage-firewall logic that catches PRICE_PER_SQFT — and the broader bug classes documented in JAMA, *Nature Communications*, and the Kaggle Santander 2019 reveal — is extracted as a standalone package: [**`schema-firewall`** v0.1.0 on PyPI](https://pypi.org/project/schema-firewall/0.1.0/) ([source](https://github.com/MarwaBS/schema-firewall)). This repo pins `schema-firewall==0.1.0` in [`requirements.txt`](requirements.txt) and re-validates the integration in its `External Benchmark` CI job on every push. `pip install schema-firewall` works globally.

This repository contains **two separate evaluation surfaces** that should not be conflated:

| Surface | Data | Purpose | Primary result | Evidence artefact |
|---|---|---|---|---|
| Trained-model evaluation | Kaggle 2023 listings (4,504 rows; BEDS, BATH, LAT/LON, SUBLOCALITY) | Model quality on matched distribution | R² = 0.815 on 20% holdout | [`reports/training_metrics.json`](reports/training_metrics.json) (committed; models + data are local-only, so this artefact is the auditable record) |
| External benchmark | NYC.gov 2024 Rolling Sales (~80k rows; no BEDS / BATH / LAT/LON) | Out-of-distribution scoring of a lean shared-feature model under a sealed schema contract | **R²(log) = 0.375 on 18,314 real 2024 sales** | [`benchmarks/results.json`](benchmarks/results.json) (committed; **fully reproducible by anyone** — the benchmark model ships in the repo and the data is a public download) |

The external benchmark is the independently verifiable surface: `python -m benchmarks.run_benchmark` on any clone downloads the public data, verifies the schema lock, and recomputes the number. The flagship evaluation is honest but local — its data and models have no public remote (DVC without a remote), so its numbers are backed by the committed metrics artefact, not by stranger-reproducibility. See [§External Benchmark](#external-benchmark--nycgov-2024) for the full information-boundary statement.

---

## Results

Every number in this section is read from the committed evidence artefact
[`reports/training_metrics.json`](reports/training_metrics.json), written by
`run_training.py` on each training run with full provenance (commit SHA,
scikit-learn version, seed, split sizes). If a number here is not in that
file, it does not belong here.

| Task | Model | Metric | Score |
|---|---|---|---|
| Price Zone (4-class) | **XGBoost + threshold tuning** | Macro F1 | **0.724** |
| Price Zone (4-class) | XGBoost (argmax) | Macro F1 | 0.711 |
| Price Zone (4-class) | LightGBM | Macro F1 | 0.692 |
| Price Regression | **XGBoost** | R2 (honest, no leakage) | **0.815** |
| Price Regression | Random Forest | R2 (honest, no leakage) | 0.804 |
| Price Regression | LightGBM | R2 (honest, no leakage) | 0.796 |

All scores on held-out 20% stratified test set (3,603 train / 901 test). No data leakage. Artifacts produced under the pinned environment (Python 3.12, numpy 1.26.4, scikit-learn 1.8.0 — recorded in the artifact's provenance block together with `working_tree_clean` and the producing commit).

Threshold tuning optimized per-class probability thresholds (Low=0.361, Medium=0.9, High=0.492, Very High=0.5), improving macro F1 from 0.711 to 0.724 (+0.014).

> The multi-task PyTorch path (`src/dl/`) is implemented and runs as an
> optional training stage when `requirements-train.txt` extras (torch) are
> installed; its results are not part of the shipped artefact and are
> therefore not quoted here.

### SHAP feature importance (top 10)

From the artefact's `classification.shap_top10` (mean |SHAP| over the test
sample, averaged across the four classes; prefixes are the
ColumnTransformer's):

| Rank | Feature | Mean abs SHAP |
|---|---|---|
| 1 | DIST_MANHATTAN_CENTER | 1.149 |
| 2 | PROPERTYSQFT | 0.858 |
| 3 | BATH | 0.786 |
| 4 | DIST_CENTRAL_PARK | 0.404 |
| 5 | SUBLOCALITY (target-encoded) | 0.391 |
| 6 | TOTAL_ROOMS | 0.347 |
| 7 | ROOMS_PER_SQFT | 0.299 |
| 8 | ZIPCODE (target-encoded) | 0.203 |
| 9 | TYPE_condo (one-hot) | 0.200 |
| 10 | BED_BATH_RATIO | 0.137 |

### Fairness by borough

From the artefact's `classification.fairness_by_borough` (a small group of
rows with missing borough is also recorded in the artifact as `"nan"`):

| Borough | Macro F1 |
|---|---|
| Staten Island | 0.778 |
| Bronx | 0.681 |
| Brooklyn | 0.677 |
| Manhattan | 0.627 |
| Queens | 0.613 |

---

## External Benchmark — NYC.gov 2024

**What this is.** A schema-constrained, leakage-proof, **fully reproducible**
out-of-distribution benchmark: a lean regressor trained only on the three
features the Kaggle training data and NYC.gov Rolling Sales genuinely share
(borough, property square footage, ZIP) is scored against real 2024 sale
transactions, under a SHA-sealed transformation contract that the
orchestrator verifies before anything runs.
**What this is not.** A production housing-price predictor. Not the flagship
model (which needs BEDS/BATH/coordinates that NYC.gov does not publish).
Not an accuracy claim for the flagship under shift.
**Why a lean model.** NYC.gov Rolling Sales is transaction data, not
listings — it publishes none of the flagship's listing features. v1 of this
benchmark proved exactly that (0 rows scoreable; the full story is in
[`benchmarks/POSTMORTEMS.md`](benchmarks/POSTMORTEMS.md)). v2+ answers the
follow-up question honestly: *how much does the shared-feature subset alone
predict, out of distribution?*

### Current sealed results (SCHEMA_MAP v3 — [`benchmarks/results.json`](benchmarks/results.json))

| | Value |
|---|---|
| Raw rows downloaded (5 boroughs) | 79,929 |
| Scored | **18,314** |
| Dropped (per-reason accounting below) | 61,615 |
| **R² in log-price space** | **0.375** |
| Leakage — name-based / semantic (Pearson + Spearman + MI) | not triggered |
| Schema SHA vs registry | verified BEFORE the run (hard gate) |
| Prediction health (NaN / Inf / collapse) | passed |
| Leakage tripwire (R² > 0.95 ⇒ investigate) | not triggered |

Drop reasons (sum reconciles to `n_dropped` — enforced at run time):

| Reason | Count |
|---|---:|
| `missing_gross_sqft` | 28,572 |
| `sale_price_non_positive` | 27,316 |
| `not_family_dwelling` (out of v2+ scope) | 3,688 |
| `sale_price_out_of_range` | 2,016 |
| `missing_year_built` | 22 |
| `missing_zip` | 1 |

**Read the number honestly:** 0.375 R²(log) is what borough + sqft + ZIP
explain about 2024 family-dwelling sale prices, full stop. It is *supposed*
to be far below the flagship's in-distribution 0.815 — the point of the
benchmark is that this gap is measured and sealed, not hidden.

**Reproduce it yourself** (no private data needed — the lean model is
committed at `models/benchmark_regressor.joblib`, 0.6 MB):

```bash
pip install -r requirements.txt && pip install openpyxl
python -m benchmarks.run_benchmark
```

The `External Benchmark` CI workflow runs exactly this on every relevant
push and weekly (NYC.gov drift watch), so the committed number is
continuously re-derived by an environment that is not the author's laptop.

### Layer separation (the whole point)

The benchmark is composed of three independent layers. Each has its own success condition. Conflating them is the most common misread.

**A. Data validity layer** — [SCHEMA_MAP.md](benchmarks/SCHEMA_MAP.md) (sealed version pinned in [`SCHEMA_MAP_VERSIONS.json`](benchmarks/SCHEMA_MAP_VERSIONS.json)) + [`benchmarks/mapping.py`](benchmarks/mapping.py)
Row-wise, stateless column mapping with an exhaustive, priority-ordered drop-rule table (§4 of the contract). Success = deterministic, auditable, target-independent; verified by the adversarial suite in [`tests/benchmarks/test_schema_firewall.py`](tests/benchmarks/test_schema_firewall.py).

**B. Evaluation layer** — [`benchmarks/invariants.py`](benchmarks/invariants.py)
The firewall checks: schema-SHA lock vs the registry (LF-normalised hashing, so the lock is platform-deterministic), name-based leakage (`FORBIDDEN_COLUMNS`), semantic leakage (Pearson + Spearman + normalised MI), drop-log reconciliation, finite-target guard, prediction health (NaN / Inf / collapse). **Every one of these is enforced inside the orchestrator at run time** — the lock check runs before the download, and a violation aborts the run rather than being recorded as an FYI.

**C. Benchmark layer** — [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py) + [`benchmarks/results.json`](benchmarks/results.json) + [`benchmarks/POSTMORTEMS.md`](benchmarks/POSTMORTEMS.md)
One-shot orchestration: verify lock → download → map → enforce invariants → score → write results. No tuning, no retry, no schema edits after seeing results. Success = reproducible, version-stamped output; the first-run number is the shipped number.

### Information-boundary statement

**Flagship performance is bounded by the observable features in NYC.gov
2024, not by model quality.** The flagship regressor requires BEDS, BATH,
coordinate-derived distances, and SUBLOCALITY; NYC.gov Rolling Sales
publishes none of these, and no amount of cleaning or retraining can
recover information the data source does not contain. That is why the
benchmark scores the lean shared-feature model — and why its 0.375 is a
statement about the shared features, never about the flagship.

Scope is restricted to 1–3 family dwellings: for condos/coops NYC.gov
reports the *building's* gross square footage, not the unit's, which is not
comparable to the per-unit Kaggle `PROPERTYSQFT`. Known structural caveats
(segment-conditioned leakage limits, non-random dropped rows) are sealed
into the contract — see SCHEMA_MAP.md §6 and §8.

### What will NOT change to "improve" this number

- The sealed `SCHEMA_MAP.md` (locked by SHA in `SCHEMA_MAP_VERSIONS.json`,
  enforced by CI **and** by the orchestrator before every run).
- Drop rules within a sealed version (changing them = bump the version,
  reseal, rerun, and treat all prior results as invalid — SCHEMA_MAP §9).
- Model features (the benchmark model is the benchmark model; retraining is
  new-version work).
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
src/data/features.py        Geospatial (haversine distances), numeric, target encoding
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
| **Geospatial** | `src/utils/geo.py` | Haversine distances to fixed landmarks (vectorised numpy). H3/KMeans/subway lookups were EDA-only and were removed from the production path — they never fed a model |
| **ML models** | `src/models/` | sklearn Pipelines, Optuna tuning, stacking ensemble, SMOTE-ENN |
| **Deep learning** | `src/dl/` | Multi-task dense net (PyTorch): entity embeddings + shared MLP trunk, classification + regression heads, Focal Loss |
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

**Data prerequisite (read first):** the training data and flagship model
artefacts are managed with DVC **without a public remote** — a fresh clone
contains neither. What works without them:

- the full unit + firewall test suite (`pytest tests/`) — hermetic;
- the **external benchmark**, end to end (`python -m benchmarks.run_benchmark`)
  — the lean model is committed and the data is a public download;
- `/health` and `/docs` on the API (model-dependent routes return 503 until
  artefacts exist).

Training the flagship and serving real predictions require the local
dataset (`output/cleaned_house_dataset.csv` via DVC):

```bash
pip install -r requirements.txt

# Train models (writes models/*.joblib + reports/training_metrics.json)
python run_training.py

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

### Geospatial

| Feature | Method | Notes |
|---|---|---|
| DIST_MANHATTAN_CENTER | Haversine to (40.758, -73.985) | hand-rolled, vectorised numpy |
| DIST_CENTRAL_PARK | Haversine to (40.783, -73.965) | hand-rolled, vectorised numpy |
| DIST_NEAREST_SUBWAY | = DIST_MANHATTAN_CENTER | **documented proxy by design**: station-level data is not bundled, and training + serving must share identical semantics (the API computes the same value; see MODEL_CARD.md) |

H3 hex indexing and KMeans neighborhood clustering were explored during
EDA but never entered any model's feature list, so they are not part of
the production pipeline (and their code/dependencies were removed —
compute that exists only to be listed in a README is a claim, not a
feature).

### Categorical encoding

| Feature | Method | Why |
|---|---|---|
| BOROUGH, TYPE | OneHotEncoder | Low cardinality (5-8 values) |
| ZIPCODE, SUBLOCALITY | TargetEncoder (smoothing=10, fit per CV fold) | High cardinality (~150 ZIPs) — OneHot would create 150 sparse columns |

---

## Models

**What produced the shipped artefacts (and every number in §Results):**
`run_training.py` — XGBoost vs LightGBM for classification, XGBoost vs
LightGBM vs Random Forest for regression, fixed hyperparameters, best
model selected on the held-out split, plus per-class threshold tuning.
That is the path whose outputs are recorded in
`reports/training_metrics.json`.

**Extended training path (optional, `requirements-train.txt`):**
`src/models/train_classification.py` / `train_dl.py` additionally
implement Optuna Bayesian tuning, CatBoost, a stacking ensemble,
SMOTE-ENN imbalance handling, and the multi-task PyTorch net. These are
real, runnable code — but their outputs are not the shipped artefacts,
so this README does not quote numbers from them.

### Classification: Price Zone (Low / Medium / High / Very High)

| Model (shipped path) | Tuning | Class Imbalance |
|---|---|---|
| XGBoost | fixed hyperparameters | none (threshold tuning post-hoc) |
| LightGBM | fixed hyperparameters | class_weight="balanced" |

### Regression: Actual Price

XGBoost / LightGBM / Random Forest predicting LOG_PRICE (log-transform stabilizes variance). Predictions converted back via `expm1()`.

### Deep Learning: Multi-Task Dense Net

(Entity embeddings + a shared MLP trunk with classification and regression heads —
a plain dense network, **not** the TabNet architecture. No sequential attention,
sparsemax feature selection, or per-sample attention masks.)

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

---

## Testing

```bash
# Full test suite with coverage (CI gate: 88%, src/ + benchmarks/ + api/)
pytest tests/ -v --tb=short --cov=src --cov=benchmarks --cov=api --cov-report=term-missing --cov-fail-under=88

# Run only leakage prevention tests
pytest tests/test_no_leakage.py -v

# Run only API tests
pytest tests/test_api.py -v
```

The suite covers:
- Data cleaning pipeline correctness
- Feature engineering (derived features, target creation, cardinality capping)
- **Data leakage prevention** — PRICE_PER_SQFT blocked in config AND validated at runtime
- Geospatial utilities (haversine; plus a guard that dead geo features stay removed)
- FastAPI endpoints (health, predict, validation errors)
- Model-loading version guard (cross-version sklearn artefacts are refused)
- The full adversarial benchmark firewall (`tests/benchmarks/`): schema lock,
  CRLF-invariance of the lock, drop-log reconciliation, NaN-target rules,
  statelessness, target-independence, collapse detectors

CI runs 4 jobs: `lint` (ruff check + ruff format + mypy + bandit, covering `src/ api/ tests/ benchmarks/`), `test` (pytest + 88% coverage gate over `src/ + benchmarks/ + api/`), `security` (pip-audit + CycloneDX SBOM emission), `docker-build` (multi-stage build + Trivy HIGH/CRITICAL scan + `/health` smoke-run). The `External Benchmark` workflow additionally re-runs the firewall suite and the full benchmark (with the committed model) on benchmark-relevant pushes and weekly.

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
│       ├── geo.py                Haversine distances (vectorised numpy)
│       └── validation.py         Schema checks, assert_no_leakage()
│
├── api/                          FastAPI prediction service
│   ├── main.py                   POST /predict, GET /health
│   └── schemas.py                Pydantic v2 request/response models
│
├── streamlit_app/
│   └── app.py                    Interactive NYC map + prediction form
│
├── tests/                        Unit + adversarial firewall suite, 88% coverage gate
│   ├── test_data_cleaner.py
│   ├── test_features.py
│   ├── test_no_leakage.py        DATA LEAKAGE PREVENTION (critical)
│   ├── test_geo.py
│   ├── test_api.py
│   └── benchmarks/               Schema-lock + drop-engine adversarial suite
│
├── docs/decisions/               Architecture Decision Records
│   ├── 001-remove-price-per-sqft.md
│   ├── 002-xgboost-primary-model.md
│   └── 003-multi-task-deep-learning.md
│
├── notebooks/                    EDA + analysis (import from src/)
├── models/                       Flagship artifacts gitignored (DVC-local);
│                                 benchmark_regressor.joblib IS committed
├── reports/training_metrics.json Committed evidence artefact for §Results
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
| DL | PyTorch 2.x (multi-task dense net, Focal Loss, entity embeddings) |
| Tuning | Optuna (Bayesian optimization) |
| Explainability | SHAP |
| Geospatial | hand-rolled haversine (vectorised numpy) |
| Encoding | category-encoders (TargetEncoder), sklearn OneHotEncoder |
| Imbalanced learning | imbalanced-learn (SMOTE-ENN) |
| API | FastAPI, Pydantic v2, Uvicorn |
| UI | Streamlit, Plotly |
| Testing | pytest (88% coverage gate over src/ + benchmarks/ + api/) |
| Linting | ruff (check + format), mypy, bandit |
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
| scikit-learn | `==1.8.0` | MUST match; cross-version loads are now REFUSED at runtime (`src/models/predict.py::ModelVersionError` promotes sklearn's `InconsistentVersionWarning` to a hard error) |
| xgboost | `==2.1.4` | `.ubj` format is stable across 2.1.x patch releases |
| lightgbm | `==4.6.0` | bumped from 4.3.0 (PYSEC-2024-231) |
| category-encoders | `==2.8.1` | 2.6.x is incompatible with sklearn 1.8 (`_get_tags` removed) |
| numpy | `==1.26.4` | |
| pandas | `==2.2.3` | |

All pins live in [`requirements.txt`](requirements.txt) (serving) and [`requirements-train.txt`](requirements-train.txt) (training extras: Optuna, SHAP, imbalanced-learn). A rebuild 6 months from now pulls the exact same wheels. Dependabot is configured to PR updates; no pin changes without a full re-train + smoke-test cycle.

---

## License

MIT
