# Security Policy

## Supported versions

Only the `main` branch is actively maintained. Security patches are applied to
the latest tagged release only.

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
- Model-inversion / membership-inference attacks on the trained classifier or
  regressor
- Leakage of secrets via error messages or logs
- Container-image CVEs scanned by Trivy in CI (see `.trivyignore` for managed
  risks)
- Supply-chain findings from `pip-audit` + CycloneDX SBOM artifacts

**Out of scope:**
- Issues requiring physical access to a user's machine
- Social engineering / phishing reports
- Denial-of-service against the Streamlit demo (public, intentionally rate-
  limited via `slowapi`)

## Handling of known managed risks

The `.trivyignore` file (if present) lists CVE IDs this project has triaged and
accepted as managed risk. Each entry has inline rationale. These are not
vulnerabilities we are hiding — they are vulnerabilities in dependencies whose
fix requires a breaking migration that is scheduled but not yet executed.

## Dependabot alerts on training-only dependencies

GitHub's Dependabot scans the full dependency tree of every `requirements*.txt`
file in the repo. This project deliberately splits dependencies into:

- **`requirements.txt`** — runtime: what ships in the production Docker image
  (pandas, numpy, scikit-learn, xgboost, lightgbm, fastapi, slowapi, streamlit, etc.).
- **`requirements-train.txt`** — training-only: torch, pytorch-tabnet, catboost,
  optuna, shap, mlflow, imbalanced-learn. These are required to RE-TRAIN the
  model from scratch but are NEVER copied into the production image (see
  `Dockerfile` and `deploy/huggingface/Dockerfile`).

Most Dependabot alerts on this repo's default branch originate from
`requirements-train.txt` packages (mlflow has 19 pending CVEs, torch has
2-5, catboost transitives). **None of these reach the production serving
path.** They are training-time tools used in a developer's local environment
or in CI's scheduled retraining workflow, not in the public-facing inference
container.

Triage policy:

1. Alerts originating SOLELY from `requirements-train.txt` are acknowledged
   and will be addressed when the upstream package ships a non-breaking fix.
2. Alerts originating from `requirements.txt` (runtime) are HIGH priority and
   will be addressed within the patch window via Dependabot pull request.
3. Alerts where both files share an affected package are treated as runtime
   alerts (HIGH priority).

This file documents the policy so reviewers understand why the alert count
is non-zero despite the runtime surface being CVE-clean (verified by the
CI `security` job running `pip-audit -r requirements.txt`).
