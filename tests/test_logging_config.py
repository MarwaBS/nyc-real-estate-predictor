"""setup_logging must actually configure the root logger.

Asserting only "some handler exists" passes under any prior logging setup;
these assert the requested level and handler land even when a handler is
already installed (pytest's capture handler guarantees one is).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from src.utils.logging_config import setup_logging


@pytest.fixture
def bare_root() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    for handler in saved_handlers:
        root.removeHandler(handler)
    yield root
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_setup_logging_sets_the_requested_level_and_a_stream_handler(
    bare_root: logging.Logger,
) -> None:
    setup_logging(level="DEBUG")
    assert bare_root.level == logging.DEBUG
    assert len(bare_root.handlers) == 1
    assert isinstance(bare_root.handlers[0], logging.StreamHandler)


def test_setup_logging_falls_back_to_info_on_an_unknown_level(
    bare_root: logging.Logger,
) -> None:
    setup_logging(level="NOT_A_LEVEL")
    assert bare_root.level == logging.INFO
