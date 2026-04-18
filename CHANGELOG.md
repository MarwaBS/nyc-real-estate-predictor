# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses SemVer for tagged releases.

## [Unreleased]

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
- Coverage gate raised from 70% → 80% in `ci.yml`.
- `requirements.txt` pinned to exact versions (not `>=`) for reproducibility.
  Dependabot manages upgrades.
- `/predict` 500-error responses no longer include the raw exception message
  (information-disclosure fix); failures are logged internally with a generic
  client-facing body.
- `slowapi` is now a hard dependency — the `try: import ... except
  ImportError: _HAS_SLOWAPI = False` fallback was removed. A deploy without
  `slowapi` installed now fails fast.

### Fixed
- README stack table: test-count claim corrected from "6 test files" to the
  actual count (14 test files).
- `api/main.py` `_get_classifier` and `_get_regressor` now have explicit
  `-> Any` return types; the `# type: ignore[no-untyped-def]` suppressions
  are removed.

### Security
- CORS wildcard in production is now rejected at startup via a Pydantic
  `model_validator(mode="after")` on the settings, matching ResumeForge's M3
  pattern.
- Trivy container scan now runs on every CI build and fails on HIGH/CRITICAL
  CVEs with a known fix (`ignore-unfixed: true`).

## [1.0.0] — 2026-04-xx

Initial production-grade ML pipeline.

### Added
- 4-model classification comparison (XGBoost / LightGBM / Random Forest /
  Multi-Task DL) with threshold tuning (F1 0.704 → 0.724).
- Price regression (XGBoost R² = 0.815 honest, no leakage).
- SHAP global + per-prediction explainability.
- Fairness analysis by borough (Staten Island 0.795 → Manhattan 0.619).
- Data-leakage guard: `PRICE_PER_SQFT` blocked in config + runtime
  validated by `test_no_leakage.py` in CI.
- FastAPI `/predict` + `/health` endpoints; Streamlit dashboard.
- MLflow experiment tracking + DVC data versioning.
- 3 ADRs: `001-remove-price-per-sqft.md`, `002-xgboost-primary-model.md`,
  `003-multi-task-deep-learning.md`.
