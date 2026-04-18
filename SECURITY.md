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
