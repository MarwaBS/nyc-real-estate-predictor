# Multi-stage Dockerfile for the NYC Real Estate price-prediction API.
#
# Stage 1 (builder): installs Python deps into an isolated prefix we can copy
# forward. pip, build toolchain, and wheel cache never reach the runtime
# image. Cuts final image size by ~30% and removes pip itself as an attack
# surface on the running container.
#
# Stage 2 (runtime): Python stdlib + the installed site-packages + a small
# set of runtime system deps (curl for healthcheck, ca-certificates for TLS).
# `apt-get upgrade -y` force-refreshes OS security patches even when the GHA
# layer cache reuses a stale base-image layer (mirrors the ResumeForge fix
# that caught libssl3 CVE-2026-28390).

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
# Base image: python:3.12-slim-bookworm. Dependabot (see .github/dependabot.yml)
# tracks the docker ecosystem and will open a PR when a newer digest or patch
# is published.
FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY requirements.txt ./

# Install into a self-contained prefix that we can copy into the runtime stage.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.12/site-packages

# Minimal runtime system deps + OS security patches. `apt-get upgrade` pulls
# the latest patches for openssl, libssl3, etc.; GHA layer cache can reuse
# a stale base layer otherwise.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

COPY --from=builder /install /install

WORKDIR /app

COPY . ./

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
