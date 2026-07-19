# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses SemVer for tagged releases.

## [Unreleased]

### Fixed — the cleaned dataset had no producer (data provenance)

- **`output/cleaned_house_dataset.csv` was not the output of
  `src/data/cleaner.py`, and no code in the repo produced it.** It held a
  `PRICE` of 2,147,483,647, which this pipeline's IQR cap makes impossible,
  and `BOROUGH`/`ZIPCODE` columns `clean_pipeline` never created. Root cause:
  `normalize_borough`/`normalize_zipcode` were guarded by
  `if col in df.columns` against a raw export containing neither, so both were
  unconditional no-ops — and the same guard silently skipped the
  borough-aware imputation this project documents. `python run_training.py`
  therefore did not reproduce the shipped models.
- **Added the missing derivation.** `derive_borough` resolves through a
  measured fallback chain (`SUBLOCALITY` 78.5%, `ADMINISTRATIVE_AREA_LEVEL_2`
  0.8%, `LOCALITY` 47.0%; 99.2% chained); `derive_zipcode` extracts 5 digits
  from the misnamed `STATE` field (100%). The 36 rows nothing resolves have
  shifted geocode columns and are dropped rather than guessed at.
- **The raw Kaggle CSV is committed and the DVC layer is deleted.**
  `.dvc/config` had no remote, so every pointer was unresolvable for any
  clone. A fresh clone can now retrain.
- **The external benchmark FELL from R²(log) 0.375 to 0.250** (−33% relative).
  The 0.375 came from a benchmark model trained on the unreproducible dataset;
  it is not a baseline this regressed against, it is a number that should not
  have been published. Postmortem in `benchmarks/POSTMORTEMS.md`.
- **`PROPERTY_CATEGORY` removed** — training, the API and the dashboard all
  hardcoded it to `"residential"`, so the one-hot encoder saw a single level.

### Fixed — provenance fields that could not report the truth

- **`working_tree_clean` could never be `true`.** It was sampled inside the
  metrics writer, after the run had written `models/`, the cleaned dataset and
  `price_interval.json`, so a run from a pristine checkout still recorded
  `false`. Now sampled before any write.
- **The emitted provenance note asserted the numbers were "not independently
  reproducible by a stranger"** because data and models were "local-only (DVC,
  no public remote)". It regenerated that falsehood on every run after the raw
  CSV and models were committed.
- **`calibrate_on="test"` is refused.** Labelling was already honest, but the
  call site was one word from fitting the served interval on the split its
  coverage is advertised against, and the full suite stayed green — the
  artefact-reading gate only inspects the committed file and CI never
  retrains.

### Changed — undefended constants replaced with measured ones (A5 defense drill)

- **The served price interval is calibrated, not asserted.** `±15%` was
  hardcoded in three files and derived from nothing; measured against the test
  split it contained the true price **32%** of the time while being presented
  to users as a price range. It is replaced by the 10th-90th percentile of
  `actual / predicted` measured on the **validation** split (0.671x / 1.390x),
  which covers **76.3%** of test against an 80% target. The multipliers ship as
  `models/price_interval.json`, pinned by the manifest, so a retrain that moves
  the residuals rewrites them — constants in source would have gone stale
  silently. The dashboard now labels the range with its coverage.
- **`/health` returns 503 when the serving stack is not loaded.** It returned a
  hardcoded `"ok"`, and every consumer (Dockerfile HEALTHCHECK, docker-compose
  `service_healthy`, the HF start script, the CI smoke test) checks the HTTP
  status while none reads the body — so a container with zero models passed all
  four. The two model probes also swallowed exceptions with no logging.
- **The benchmark leakage firewall fails the run.** A confirmed leak recorded
  `"triggered": true` and exited 0, making the benchmark's headline control the
  one gate that could never turn CI red. It now exits non-zero *after* writing
  `results.json`, so the evidence survives the failure.
- **`reproducibility.tolerance` claim removed.** It recorded `"±1e-6 on
  metrics"` as a string nothing in the repo compared anything against. Replaced
  with what is actually enforced: pinned deps, fixed seed, schema SHA gate.
- **`daily_rate_limit` renamed to `predict_rate_limit`** (env var
  `PREDICT_RATE_LIMIT`). The name promised a daily quota while the value was
  per-minute, so `60/minute` advertised a cap of 60 while permitting 86,400/day.
  The `/predict` docstring no longer hardcodes "the 61st request" into the
  public `/docs` page, which was false for any deployment overriding the default.
- **Vacuous assertions replaced.** `r2 > -10` (a constant predictor scores 0)
  became `r2 > 0`; `0 <= macro_f1 <= 1.0` (a metric's own codomain) removed as
  the majority-baseline check above it is the real gate; `predicted_price > 0`
  became a $10k floor — the historical bug that served a Manhattan condo at
  single-digit dollars would have passed the old assertion.
- **CI actions pinned.** `trivy-action@master` and `free-disk-space@main` were
  the only floating refs, and trivy is the action enforcing the HIGH/CRITICAL
  gate — an upstream change could silently redefine it.

### Changed — test-set contamination removed (headline numbers move)

- **Three-way train/val/test split.** Model selection (both classifiers and
  all three regressors) previously compared candidates on the *test* split
  and then published the winner's score from that same split — a selected
  maximum reported as a hold-out estimate. Candidates are now compared on a
  new val split (2,896 / 724 / 906) and test is scored exactly once, by the
  already-selected model.
- **Deep-learning early stopping no longer watches test.** `patience=15` was
  evaluated against the test labels, which selects the stopping epoch on test.
  It now reads the val split.
- **Per-class threshold tuning removed entirely.** Thresholds were fitted
  against the test labels and the resulting macro F1 (0.724) published as a
  hold-out result, so the advertised +0.014 gain over argmax was in-sample.
  Fitted on one half of the test set and scored on the other across 20
  stratified splits, the effect is +0.0006 (std 0.0106), helping 12 and
  hurting 8 — noise. Removed rather than moved to val. `src/models/threshold.py`
  is replaced by `src/models/decode.py` (argmax only, one shared decode for
  API + dashboard); `models/optimal_thresholds.joblib` is deleted.
- **Reported metrics (test split, current artefacts):** classification macro
  F1 **0.727** (XGBoost; 0.721 on val), regression R² **0.835** (XGBoost;
  0.826 on val). These are not comparable to any figure published before
  2026-07-19 — see the data-provenance entry below, which changed the dataset
  itself.
- **`assert_no_leakage` rejects any price-derived name.** It enumerated five
  spellings, so `["BEDS", "PRICE"]` — the raw target — passed the leakage
  guard. It now rejects any feature name containing "price"; no legitimate
  feature in `src/config.py` does.

### Added
- `LICENSE` (MIT) matching the README stack-table claim.
- `SECURITY.md` with disclosure policy + in-scope / out-of-scope boundaries.
- `CHANGELOG.md` (this file).
- `MODEL_CARD.md` documenting model intended use, training data, evaluation
  methodology, fairness analysis, and known limitations (Google "Model Cards
  for Model Reporting" format).
- `.github/dependabot.yml` — weekly grouped updates for pip + docker +
  github-actions ecosystems.
- `.trivyignore` — starter file (empty ignore list).
- Multi-stage `Dockerfile` with `python:3.12-slim-bookworm` base + `apt-get
  upgrade -y` in runtime for OS-level CVE patches.
- CI: `docker-build` job with Trivy HIGH/CRITICAL scan and `/health` smoke-
  run.
- CI: CycloneDX SBOM emission via `pip-audit --format=cyclonedx-json` in the
  security job (90-day artifact retention).
- CI: `bandit` step in the lint job for static security analysis.
- API-key authentication on `POST /predict` via `X-API-Key` header with
  timing-safe comparison.
- CORS fail-fast in prod — when `ENV=prod` and `ALLOWED_ORIGINS` is `*` or
  empty, startup raises rather than silently accepting wildcard credentials.

### Changed
- Python runtime bumped from 3.11 → 3.12 across Dockerfile, `pyproject.toml`
  (ruff + mypy targets), and CI setup-python calls.
- Coverage gate raised from 70% → 88% and broadened to measure `api/` (the
  serving + auth/rate-limit/predict layer, previously unmeasured) alongside
  `src/` + `benchmarks/`, in `ci.yml` and the `Makefile`. Actual coverage 93%.
- `requirements.txt` pinned to exact versions (not `>=`) for reproducibility.
  Dependabot manages upgrades.
- `/predict` 500-error responses no longer include the raw exception message
  (information-disclosure fix); failures are logged internally with a generic
  client-facing body.
- `slowapi` is now a hard dependency — the `try: import ... except
  ImportError: _HAS_SLOWAPI = False` fallback was removed. A deploy without
  `slowapi` installed now fails fast.

### Fixed
- **Served price-zone labels were wrong for 3 of the 4 classes.** The
  classifier's class indices follow the label encoder's ALPHABETICAL order
  (`High, Low, Medium, Very High`), but every serving path (API, predict
  module, Streamlit) decoded them with the semantic config order
  (`Low, Medium, High, Very High`) — a $1.4M Manhattan condo was served
  zone "Low". All serving paths now decode through the shipped
  `models/label_encoder.joblib` (`src.models.predict.get_zone_classes`),
  the single source of truth for class order. Regression tests pin the
  decode order to `label_encoder.classes_` (mock encoders deliberately use
  an order that differs from the config list) and assert zone/price
  consistency against the shipped artefacts.
- **Threshold names and per-class report rows in
  `reports/training_metrics.json` were misattributed** by the same
  root cause (`run_training.py` keyed encoder-ordered columns with
  config-ordered names). Training now keys thresholds and classification
  reports with `label_encoder.classes_`, records the order in
  `classification.metrics.labels`, and the artefact was regenerated by a
  full rerun under the pinned environment: every aggregate and per-class
  value is bit-identical to the previous artefact modulo the re-keying
  (they are permutation-invariant), except one final-ulp float in the
  non-selected RandomForest regression candidate (thread-order summation
  jitter; displayed precision unaffected). README and MODEL_CARD
  threshold attributions corrected accordingly (values unchanged, names
  re-keyed).
- README stack table: test-count claim corrected from "6 test files" to the
  actual count (14 test files).
- `api/main.py` `_get_classifier` and `_get_regressor` now have explicit
  `-> Any` return types; the `# type: ignore[no-untyped-def]` suppressions
  are removed.

### Security
- CORS wildcard in production is now rejected at startup via a Pydantic
  `model_validator(mode="after")` on the settings, matching ResumeForge's M3
  pattern. The guard is now covered by `tests/test_settings.py` (prod +
  wildcard/empty/wildcard-in-list refuse startup; prod + explicit origins
  and dev + wildcard start clean).
- Trivy container scan now runs on every CI build and fails on HIGH/CRITICAL
  CVEs with a known fix (`ignore-unfixed: true`).
- `starlette>=1.3.1` pinned in `requirements.txt` to exclude CVE-2026-54283
  (HIGH; the `request.form()` DoS in starlette 1.3.0). fastapi's transitive
  constraint is `starlette>=0.46.0` with no upper cap, so this explicit floor
  governs: a clean `pip install -r requirements.txt` (and therefore the Docker
  image, which installs the same file) resolves starlette 1.3.1, never 1.3.0.

## [1.0.0] — 2026-04-xx

Initial production-grade ML pipeline.

### Added
- 4-model classification comparison (XGBoost / LightGBM / Random Forest /
  Multi-Task DL) with threshold tuning (macro-F1 0.711 → 0.724).
- Price regression (XGBoost R² = 0.815 honest, no leakage).
- SHAP global + per-prediction explainability.
- Fairness analysis by borough (best Staten Island 0.778 → worst Queens 0.613 macro-F1; `fairness_by_borough` in `reports/training_metrics.json` is authoritative).
- Data-leakage guard: `PRICE_PER_SQFT` blocked in config + runtime
  validated by `test_no_leakage.py` in CI.
- FastAPI `/predict` + `/health` endpoints; Streamlit dashboard.
- MLflow experiment tracking + DVC data versioning.
- 3 ADRs: `001-remove-price-per-sqft.md`, `002-xgboost-primary-model.md`,
  `003-multi-task-deep-learning.md`.
