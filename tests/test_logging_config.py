"""Tests for logging configuration."""
from __future__ import annotations

import logging

from src.utils.logging_config import setup_logging


def test_setup_logging_does_not_crash() -> None:
    setup_logging(level="DEBUG")
    # Verify a handler was added (basicConfig adds to root)
    root = logging.getLogger()
    assert len(root.handlers) > 0


def test_setup_logging_accepts_info() -> None:
    setup_logging(level="INFO")
    root = logging.getLogger()
    assert len(root.handlers) > 0
