"""setup_logging must reconfigure a root logger that is already configured.

basicConfig without force= silently no-ops when any root handler exists, so
each test seeds a sentinel handler and a wrong level BEFORE the call — an
assertion that starts from a bare root passes without force= and gates
nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from src.utils.logging_config import setup_logging


@pytest.fixture
def preconfigured_root() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.addHandler(logging.NullHandler())
    root.setLevel(logging.CRITICAL)
    yield root
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_setup_logging_wins_over_an_existing_configuration(
    preconfigured_root: logging.Logger,
) -> None:
    setup_logging(level="DEBUG")
    assert preconfigured_root.level == logging.DEBUG
    assert len(preconfigured_root.handlers) == 1
    assert isinstance(preconfigured_root.handlers[0], logging.StreamHandler)


def test_setup_logging_falls_back_to_info_on_an_unknown_level(
    preconfigured_root: logging.Logger,
) -> None:
    setup_logging(level="NOT_A_LEVEL")
    assert preconfigured_root.level == logging.INFO
