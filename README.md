# NYC Real Estate Price Prediction

[![CI](https://github.com/MarwaBS/nyc-real-estate-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/MarwaBS/nyc-real-estate-predictor/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-on%20Hugging%20Face-yellow)](https://huggingface.co/spaces/MarwaBS/nyc-real-estate-predictor)

**Predict NYC residential prices and derive their price zones with one XGBoost regressor over 4,500+ listings with geospatial features.**

**Shipped model** — `XGBoost` — the trained artefact in [`reports/training_metrics.json`](reports/training_metrics.json) is the single source of truth; a CI gate fails if this field disagrees with it.

> Every model is trained without data leakage. Previous R2=0.997 results were caused by PRICE_PER_SQFT (derived from target) — this has been removed and [documented as ADR-001](docs/decisions/001-remove-price-per-sqft.md).

> **Built on a published library we own.** The leakage-firewall logic that catches PRICE_PER_SQFT — and the broader bug classes documented in JAMA, *Nature Communications*, and the Kaggle Santander 2019 reveal — is extracted as a standalone package: [**`schema-firewall`** on PyPI](https://pypi.org/project/schema-firewall/) ([source](https://github.com/MarwaBS/schema-firewall)). This repo pins `schema-firewall==0.1.3` in [`requirements.txt`](requirements.txt) and re-validates the integration in its `External Benchmark` CI job on every push. `pip install schema-firewall` works globally.

This repository contains **two separate evaluation surfaces** that should not be conflated:

| Surface | Data | Purpose | Primary result | Evidence artefact |
|---|---|---|---|---|
| Trained-model evaluation | Kaggle 2023 listings (4,526 rows; BEDS, BATH, LAT/LON, SUBLOCALITY) | Model quality on matched distribution | R² = 0.835 on the 20% test split (naive borough-median baseline: 0.177) | [`reports/training_metrics.json`](reports/training_metrics.json) (committed; the raw CSV is committed too, so `python run_training.py` reproduces this from a fresh clone) |
| External benchmark | NYC.gov 2024 Rolling Sales (~80k rows; no BEDS / BATH / LAT/LON) | Out-of-distribution scoring of a lean shared-feature model under a sealed schema contract | **R²(log) = 0.250 on 18,321 real 2024 sales** | [`benchmarks/results.json`](benchmarks/results.json) (committed; **fully reproducible by anyone** — the benchmark model ships in the repo and the data is a public download) |

Both surfaces are reproducible from a fresh clone: `python -m benchmarks.run_benchmark` downloads the public NYC.gov data, verifies the schema lock, and recomputes the benchmark number, while `python run_training.py` cleans the committed raw Kaggle CSV and regenerates every flagship artefact and the metrics file behind the R² above. See [§External Benchmark](#external-benchmark--nycgov-2024) for the full information-boundary statement.

---

## Results

Every number in this section is read from the committed evidence artefact
[`reports/training_metrics.json`](reports/training_metrics.json), written by
`run_training.py` on each training run with full provenance (commit SHA,
scikit-learn version, seed, split sizes). If a number here is not in that
file, it does not belong here.

Headline scores are on the **test** split, produced by the model that the
**val** split selected. Candidates are compared on val only; test is scored
once, after selection is fixed.

| Task | Model | Metric | Val (selection) | **Test (reported)** |
|---|---|---|---|---|
| Price Regression | **XGBoost** (selected on val) | R2 | 0.774 | **0.835** |
| Price Zone (bucketed from the above) | — | Macro F1 | — | **0.712** |

Across **20 seeds** of the full protocol (split, train-only fitting and
candidate selection re-run each time — `scripts/measure_seed_variance.py`,
recorded in [`reports/seed_variance.json`](reports/seed_variance.json)):
test R² **0.814 ± 0.028**, zones macro F1 **0.717 ± 0.020**, against a
per-borough-median baseline of 0.170 ± 0.017 R² and 0.242 ± 0.058 F1.
XGBoost wins selection in 16/20 runs (candidates Random Forest 3, LightGBM 1) — at 4,526
rows the candidate ranking is seed-sensitive, which is exactly why the
spread is published next to the point estimates.

Split: 2,896 train / 724 val / 906 test, stratified on pooled price quartiles (a balancing key only — served zone labels come from train-derived cut-points). Losing
candidates show "not scored" because they never touch test — scoring them
there and then quoting the winner's number is how a selected maximum gets
published as a hold-out estimate. Artifacts produced under the pinned
environment (Python 3.12.13, numpy 1.26.4, scikit-learn 1.8.0 — recorded in
the artifact's provenance block together with `working_tree_clean` and the
producing commit).

**Per-class threshold tuning was removed; serving has no thresholds to tune.**
An earlier revision published 0.724 macro F1 from per-class thresholds fitted
against the *test* labels and then scored on those same labels, so the
advertised "+0.014 gain" was the tuner reading its own answer sheet. Measured
honestly — thresholds fitted on half the test set and scored on the other half,
over 20 stratified splits — tuning was worth +0.0006 mean (std 0.0106), helping
12 splits and hurting 8. It is noise, so it was removed rather than moved to
val.

**These numbers are not comparable to any published before 2026-07-19.** The
dataset itself changed: `BOROUGH` and `ZIPCODE` are now derived by committed
code from the raw export rather than inherited from a pre-cleaned CSV that
nothing in this repo could regenerate, and a 2³¹−1 overflow sentinel is now
dropped instead of being capped into a plausible-looking listing. Zones are
scored on test only: they are bucketed from the regressor's predictions, so
there is no separate zone model to select on val and no val macro F1 to
compare against.

### SHAP feature importance (top 10)

From the artefact's `classification.shap_top10` (mean |SHAP| over the
regressor's test predictions; prefixes are the ColumnTransformer's):

| Rank | Feature | Mean abs SHAP |
|---|---|---|
| 1 | BATH | 0.369 |
| 2 | DIST_MANHATTAN_CENTER | 0.362 |
| 3 | PROPERTYSQFT | 0.144 |
| 4 | DIST_CENTRAL_PARK | 0.116 |
| 5 | TOTAL_ROOMS | 0.108 |
| 6 | ZIPCODE (target-encoded) | 0.097 |
| 7 | TYPE_co-op (one-hot) | 0.079 |
| 8 | SUBLOCALITY (target-encoded) | 0.071 |
| 9 | ROOMS_PER_SQFT | 0.034 |
| 10 | BOROUGH_the bronx (one-hot) | 0.027 |

### Fairness by borough

From the artefact's `classification.fairness_by_borough`. There is no `"nan"`
group any more: rows whose borough cannot be derived are dropped during
cleaning rather than carried into training as an unnamed category.

| Borough | Macro F1 |
|---|---|
| Manhattan | 0.696 |
| Brooklyn | 0.691 |
| The Bronx | 0.688 |
| Queens | 0.679 |
| Staten Island | 0.529 |

The spread (0.696 Manhattan vs 0.529 Staten Island) is wide enough to matter
and is reported unexplained: no experiment here establishes a cause, so it is
recorded as an observation. Every borough clears its own majority-class
baseline by at least 0.39 macro F1 — the floor `check_borough_floor` enforces
at train time, which fails the run rather than publishing a breach.

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
| Raw rows downloaded (5 boroughs) | 80,476 |
| Scored | **18,321** |
| Dropped (per-reason accounting below) | 62,155 |
| **R² in log-price space** | **0.250** |
| Naive baseline (borough-median train price, same rows) | **-0.016** |
| Leakage firewall (`schema-firewall`) | **passed** — no forbidden column; no statistical mirroring among the columns the MI gate covers (Pearson + Spearman + MI). Scope note below |
| Schema SHA vs registry | verified BEFORE the run (hard gate) |
| Prediction health (NaN / Inf / collapse) | passed |
| Leakage tripwire (R² > 0.95 ⇒ investigate) | not triggered |

**Scope of the statistical gate.** `check_leakage` reads numeric columns, so
[`benchmarks/invariants.py`](benchmarks/invariants.py) factorises low-cardinality string columns and
**drops** the rest before the call rather than letting them pass through unassessed. Of the three
shipped features, `zip_code` is a high-cardinality string and is therefore covered by the name-based
forbidden-column check only, not by Pearson/Spearman/MI. That is deliberate — factorising 1,000 ZIP
codes into arbitrary integers would feed the detector a meaningless ordinal — but it means "no
statistical mirroring" is a claim about `borough` and `property_sqft`, not about every column.

Drop reasons (sum reconciles to `n_dropped` — enforced at run time):

| Reason | Count |
|---|---:|
| `missing_gross_sqft` | 28,705 |
| `sale_price_non_positive` | 27,683 |
| `not_family_dwelling` (out of v2+ scope) | 3,744 |
| `sale_price_out_of_range` | 2,001 |
| `missing_year_built` | 21 |
| `missing_zip` | 1 |
| **Total** | **62,155** |

The externally-benchmarked score is **0.250 R²(log)**, below a previously
published **0.375** (on 18,314 rows). That earlier figure came from a benchmark
model trained on `output/cleaned_house_dataset.csv` when that file had **no
producer in this repo**: it could not be regenerated from committed code and
carried a 2,147,483,647 overflow sentinel plus `BOROUGH`/`ZIPCODE` columns the
cleaner never created. Retrained on the reproducible dataset, the score is
0.250. The 0.375 was never reproducible, so it is not a baseline this figure
regressed against.

**Read the number honestly:** 0.250 R²(log) is what borough + sqft + ZIP
explain about 2024 family-dwelling sale prices, full stop. It is *supposed*
to be far below the flagship's in-distribution 0.835 — the point of the
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
Row-wise, stateless column mapping with an exhaustive, priority-ordered drop-rule table (§4 of the contract). Success = deterministic, auditable, target-independent; verified by the hostile-input suite in [`tests/benchmarks/test_schema_firewall.py`](tests/benchmarks/test_schema_firewall.py).

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
benchmark scores the lean shared-feature model — and why its 0.250 is a
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
Raw CSV (4,801 rows)
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
    +---> run_training.py                       Candidate training + selection on val
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
| **ML models** | `src/models/` | sklearn Pipelines; one XGBoost regressor |
| **Explainability** | `src/models/explain.py` | SHAP TreeExplainer, per-prediction explanations, fairness by borough |
| **API** | `api/` | FastAPI with Pydantic v2 schemas, health checks |
| **UI** | `streamlit_app/` | Interactive NYC map, prediction form, calibrated price range |
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

**Data prerequisite (read first):** the raw Kaggle snapshot
(`Resources/NY-House-Dataset.csv`, 1.3 MB) is committed, so a fresh clone can
retrain everything:

```bash
python run_training.py    # cleans the raw CSV, trains, writes models/ + reports/
```

That regenerates `output/cleaned_house_dataset.csv` and every model artefact.
The trained `.joblib` files are committed and pinned byte-for-byte by
[`models/MANIFEST.sha256`](models/MANIFEST.sha256), so the artefacts a reviewer
downloads are provably the ones the quoted metrics describe; the cleaned CSV is
gitignored because `run_training.py` derives it on every run.

Also available without training:

- the full unit + firewall test suite (`pytest tests/`) — hermetic;
- the **external benchmark**, end to end (`python -m benchmarks.run_benchmark`)
  — the lean model is committed and the data is a public download;
- `/health` and `/docs` on the API (model-dependent routes return 503 until
  artefacts exist).

Serving real predictions requires the local
dataset, which `python run_training.py` generates from the committed raw CSV:

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
| ROOMS_PER_SQFT | TOTAL_ROOMS / SQFT | Density metric |

### Geospatial

| Feature | Method | Notes |
|---|---|---|
| DIST_MANHATTAN_CENTER | Haversine to (40.758, -73.985) | hand-rolled, vectorised numpy |
| DIST_CENTRAL_PARK | Haversine to (40.783, -73.965) | hand-rolled, vectorised numpy |

H3 hex indexing and KMeans neighborhood clustering were explored during
EDA but never entered any model's feature list, so they are not part of
the production pipeline (and their code/dependencies were removed —
compute that exists only to be listed in a README is a claim, not a
feature).

### Categorical encoding

| Feature | Method | Why |
|---|---|---|
| BOROUGH, TYPE | OneHotEncoder | Low cardinality (5-8 values) |
| ZIPCODE, SUBLOCALITY | TargetEncoder (smoothing=10, fit inside the Pipeline on train only) | High cardinality (~150 ZIPs) — OneHot would create 150 sparse columns |

---

## Models

**What produced the shipped artefacts (and every number in §Results):**
`run_training.py` — XGBoost vs LightGBM vs Random Forest for regression,
fixed hyperparameters, best model selected on the **val** split and then
scored once on **test**; zones are that model's predictions bucketed through
`PRICE_ZONE_BINS`. Outputs are recorded in `reports/training_metrics.json`.

The reasoning behind each choice — why one regressor over a classifier, what
was tried and removed, and how every constant was measured — is in
[`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md).

### Regression: Actual Price

XGBoost / LightGBM / Random Forest predicting LOG_PRICE (log-transform stabilizes variance), compared on val; the winner is scored once on test. Predictions converted back via `expm1()`.

## Explainability

- **SHAP summary plot**: Global feature importance (mean |SHAP|) — replaces `.feature_importances_`
- **SHAP waterfall**: Per-prediction explanation (which features drove this specific prediction)
- **SHAP dependence**: DIST_MANHATTAN_CENTER vs PRICE_ZONE (geographic price gradient)
- **Fairness analysis**: Macro F1 computed per borough to detect geographic bias

---

## Testing

```bash
# Full test suite with coverage (CI gate: 85%)
pytest tests/ -v --tb=short --cov=src --cov=benchmarks --cov=api --cov=run_training --cov=streamlit_app --cov-report=term-missing --cov-fail-under=85

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
- The full hostile-input benchmark firewall (`tests/benchmarks/`): schema lock,
  CRLF-invariance of the lock, drop-log reconciliation, NaN-target rules,
  statelessness, target-independence, collapse detectors

CI runs 4 jobs: `lint` (ruff check + ruff format across the tree; mypy + bandit over every tracked Python file outside `tests/`), `test` (pytest + 85% coverage gate over `src/ + api/ + benchmarks/ + streamlit_app/ + run_training.py`), `security` (pip-audit + CycloneDX SBOM emission), `docker-build` (multi-stage build + Trivy HIGH/CRITICAL scan + `/health` smoke-run). The `External Benchmark` workflow additionally re-runs the firewall suite and the full benchmark (with the committed model) on benchmark-relevant pushes and weekly.

---

## Data leakage: why R2=0.997 was wrong

The original regression model used `PRICE_PER_SQFT` (= PRICE / PROPERTYSQFT) as a feature. Since the target is PRICE, this gives the model a near-perfect answer:

```
PRICE = PRICE_PER_SQFT * PROPERTYSQFT   # trivial algebra
```

R2=0.997 was not a real prediction — it was circular computation. After removing this feature:
- Honest R2 = **0.835** (XGBoost, selected on val) — a real result, not inflated
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
│   │   ├── evaluate.py           Metrics, confusion matrix, fairness
│   │   ├── explain.py            SHAP values + global importance
│   │   └── predict.py            Load model + inference
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
├── tests/                        Unit + hostile-input firewall suite, 85% coverage gate
│   ├── test_data_cleaner.py
│   ├── test_features.py
│   ├── test_no_leakage.py        DATA LEAKAGE PREVENTION (critical)
│   ├── test_geo.py
│   ├── test_api.py
│   └── benchmarks/               Schema-lock + drop-engine hostile-input suite
│
├── docs/decisions/               Architecture Decision Records
│   ├── 001-remove-price-per-sqft.md
│   ├── 002-xgboost-primary-model.md
│   ├── 003-multi-task-deep-learning.md
│
├── notebooks/                    EDA + analysis (import from src/)
├── models/                       Flagship artifacts committed + MANIFEST.sha256-pinned;
│                                 regenerate with run_training.py
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
| ML | scikit-learn; XGBoost shipped (selected on val), Random Forest / LightGBM candidates |
| Tuning | none — fixed hyperparameters, candidates compared on val |
| Explainability | SHAP |
| Geospatial | hand-rolled haversine (vectorised numpy) |
| Encoding | category-encoders (TargetEncoder), sklearn OneHotEncoder |
| API | FastAPI, Pydantic v2, Uvicorn |
| UI | Streamlit, Plotly |
| Testing | pytest (85% coverage gate over src/, api/, benchmarks/, streamlit_app/ and run_training.py) |
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

All pins live in [`requirements.txt`](requirements.txt) (serving) and [`requirements-train.txt`](requirements-train.txt) (training extras: SHAP, MLflow). A rebuild 6 months from now pulls the exact same wheels. Dependabot is configured to PR updates; no pin changes without a full re-train + smoke-test cycle.

---

## License

MIT
