"""
API settings — Pydantic BaseSettings with environment-aware validation.

Three environment variables drive the API's security posture:

    ENV=dev|staging|prod    — default "dev"
    ALLOWED_ORIGINS         — comma-separated list; default "*"
    API_KEY                 — if set, /predict requires X-API-Key header
                              matching; if unset, /predict is open (dev mode)
    DAILY_RATE_LIMIT        — slowapi-style rate limit; default "100/minute"

The `validate_cors_not_wildcard_in_prod` model-validator is the critical
piece: when `ENV=="prod"`, an empty or wildcard `ALLOWED_ORIGINS` raises
at startup rather than silently accepting any Origin header. Mirrors the
ResumeForge M3 fix.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """Runtime settings for the prediction API."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = Field(default="dev", description="Deployment environment")
    allowed_origins: str = Field(
        default="*",
        description="Comma-separated CORS origins; wildcard rejected in prod.",
    )
    api_key: str = Field(
        default="",
        description=(
            "Optional shared-secret API key. If set, /predict requires the "
            "X-API-Key request header matching this value (timing-safe "
            "comparison). If empty, /predict is open (dev/portfolio mode)."
        ),
    )
    daily_rate_limit: str = Field(
        default="100/minute",
        description="slowapi-format rate limit for /predict.",
    )

    @property
    def origins_list(self) -> list[str]:
        """Parse the comma-separated origins into a clean list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def validate_cors_not_wildcard_in_prod(self) -> APISettings:
        """Refuse to start in prod when CORS is not locked down.

        Raises ValueError when `env == "prod"` AND `allowed_origins` is empty,
        is `"*"`, or contains `"*"` as any comma-separated item. Dev and
        staging keep the permissive default for frictionless local work.
        """
        if self.env != "prod":
            return self
        origins = self.origins_list
        if not origins or any(o == "*" for o in origins):
            raise ValueError(
                "ALLOWED_ORIGINS must be an explicit comma-separated list in "
                "prod (no wildcard). Current value: "
                f"{self.allowed_origins!r}. Set ALLOWED_ORIGINS to e.g. "
                '"https://app.example.com,https://admin.example.com".'
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    """Cached settings accessor. Construct once per process."""
    return APISettings()
