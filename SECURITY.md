# Security Policy

## Supported versions

Only the `main` branch is actively maintained. There are no tagged releases;
patches land on `main`.

## Reporting a vulnerability

**Do not open a public GitHub issue for security bugs.**

Email: `marwabensalem30@gmail.com` with subject prefix `[SECURITY]` and the
following details:

- A description of the vulnerability
- Steps to reproduce (PoC if available)
- Affected component (API endpoint, data pipeline, deployment surface, etc.)
- Your assessment of severity and potential impact

You can expect an initial acknowledgement within 72 hours.

## Scope

**In scope:**
- `POST /predict` endpoint input validation + auth
- `GET /health` endpoint information disclosure
- Model-inversion / membership-inference attacks on the trained regressor
- Leakage of secrets via error messages or logs
- Container-image CVEs scanned by Trivy in CI (see `.trivyignore` for managed
  risks)
- Supply-chain findings from `pip-audit` + CycloneDX SBOM artefacts

**Out of scope:**
- Issues requiring physical access to a user's machine
- Social engineering / phishing reports
- Denial-of-service against the public demo surfaces. The FastAPI
  `/predict` endpoint is rate-limited per client IP via `slowapi`
  (`api/main.py::PREDICT_RATE_LIMIT`, HTTP 429 beyond the limit); the
  Streamlit demo runs behind its hosting platform's own ingress controls
  (`slowapi` does not apply to Streamlit).


## Dependabot alerts on training-only dependencies

GitHub's Dependabot scans the full dependency tree of every `requirements*.txt`
file in the repo. This project deliberately splits dependencies into:

- **`requirements.txt`** — runtime: what ships in the production Docker image
  (pandas, numpy, scikit-learn, xgboost, lightgbm, fastapi, slowapi, streamlit, etc.).
- **`requirements-train.txt`** — training-only: `shap` (the training-time SHAP
  pass) and `mlflow` (experiment tracking). These are required to RE-TRAIN the
  model but are NEVER copied into the production image (see `Dockerfile` and
  `deploy/huggingface/Dockerfile`). torch, catboost, optuna and imbalanced-learn
  were removed with the multi-task net and the extended training path.

Most Dependabot alerts on this repo's default branch originate from
`requirements-train.txt` — mlflow carries the bulk of the pending CVEs.
**None of these reach the production serving path.** They are training-time
tools run by hand in a developer's environment. No workflow installs
`requirements-train.txt`, and neither Dockerfile copies it, so they never reach
the public-facing inference container.

Triage policy:

1. Alerts originating SOLELY from `requirements-train.txt` are acknowledged
   and will be addressed when the upstream package ships a non-breaking fix.
2. Alerts originating from `requirements.txt` (runtime) are HIGH priority.
   Dependabot opens the pull request, except for the packages whose minor
   updates are frozen while numpy stays at 1.x (the list is in
   `.github/dependabot.yml`, recomputed by `tests/test_dependency_freeze.py`).
   Those still receive patch updates; a fix that needs a minor bump is taken by
   hand. `streamlit` is in that set, and its last CVE fix was a minor bump.
3. Alerts where both files share an affected package are treated as runtime
   alerts (HIGH priority).

This file documents the policy so reviewers understand why the alert count
is non-zero despite the runtime surface carrying no unignored pip-audit
finding, `PYSEC-2024-277` being ignored with its rationale in `ci.yml` (verified by the
CI `security` job running `pip-audit -r requirements.txt`).
