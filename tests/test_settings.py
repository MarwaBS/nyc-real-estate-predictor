"""Tests for api.settings — the prod CORS startup guard.

The guard (``validate_cors_not_wildcard_in_prod``) is the control that makes
a prod deploy with wildcard CORS REFUSE to start. It previously had zero
tests, so a regression would have shipped silently. Settings are constructed
with explicit kwargs — in pydantic-settings, init kwargs take precedence over
environment variables and ``.env``, so these tests are hermetic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.settings import APISettings


def test_prod_wildcard_origins_refuses_startup() -> None:
    with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
        APISettings(env="prod", allowed_origins="*")


def test_prod_empty_origins_refuses_startup() -> None:
    with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
        APISettings(env="prod", allowed_origins="  , ,")


def test_prod_wildcard_hidden_in_list_refuses_startup() -> None:
    with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
        APISettings(env="prod", allowed_origins="https://app.example.com,*")


def test_prod_explicit_origins_start_clean() -> None:
    settings = APISettings(
        env="prod",
        allowed_origins="https://app.example.com,https://admin.example.com",
    )
    assert settings.origins_list == [
        "https://app.example.com",
        "https://admin.example.com",
    ]


def test_dev_wildcard_stays_permissive() -> None:
    """Dev keeps the frictionless default — the guard is prod-only."""
    settings = APISettings(env="dev", allowed_origins="*")
    assert settings.origins_list == ["*"]
