"""The provenance block must be able to tell the truth.

Two of its fields were previously incapable of it: ``working_tree_clean`` was
sampled after the run had written models/ and the cleaned dataset, so it read
``false`` even from a pristine checkout, and a hardcoded ``note`` asserted the
numbers were not reproducible by a stranger long after the raw CSV and the
models were committed. A provenance field that cannot report the good outcome
is not evidence, it is decoration.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import tempfile
import textwrap
from pathlib import Path

import pytest

import run_training

ARTEFACT = Path(__file__).resolve().parents[1] / "reports" / "training_metrics.json"


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


def _first_call_line(source: str, name: str) -> int | None:
    """Line of the first call to ``name`` in ``source``, or None if never called."""
    tree = ast.parse(textwrap.dedent(source))
    return min(
        (
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ),
        default=None,
    )


def test_working_tree_clean_is_sampled_before_main_writes_anything() -> None:
    """``prepare_data`` writes the cleaned dataset, so a sample taken after it
    can only ever record False.

    Ordering is read from the parsed call graph rather than the source text:
    a substring search for the call name is satisfied by a comment naming it,
    so deleting the call and leaving the comment behind kept this green.
    """
    source = inspect.getsource(run_training.main)

    sample_at = _first_call_line(source, "_git_working_tree_clean")
    first_write_at = _first_call_line(source, "prepare_data")

    assert sample_at is not None, "main no longer samples the working tree at all"
    assert first_write_at is not None, "main no longer calls prepare_data"
    assert sample_at < first_write_at, (
        "working_tree_clean is sampled after prepare_data() has already "
        "written the cleaned dataset, so it can only ever record False"
    )


def test_write_training_metrics_records_the_flag_it_is_given() -> None:
    """The threaded value must survive into the artefact unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            for flag in (True, False):
                run_training._write_training_metrics(
                    flag,
                    {"selected_model": "stub", "metrics": {}},
                    {"selected_model": "stub", "metrics": {}},
                    n_train=1,
                    n_val=1,
                    n_test=1,
                    features=["f"],
                    zone_bins=[0.0, 1.0, 2.0, 3.0, float("inf")],
                    baseline={"predictor": "stub"},
                )
                written = json.loads(
                    Path("reports/training_metrics.json").read_text(encoding="utf-8")
                )
                assert written["working_tree_clean"] is flag
        finally:
            os.chdir(cwd)


def test_provenance_note_does_not_deny_reproducibility(metrics: dict) -> None:
    """The shipped note must not contradict the committed repo.

    The raw CSV and every model artefact are tracked, so the run IS
    reproducible by a stranger; the old note said the opposite and
    regenerated itself on every run.
    """
    note = metrics["note"].lower()
    assert "not independently reproducible" not in note
    assert "no public remote" not in note


def test_reported_split_is_test_and_selection_split_is_val(metrics: dict) -> None:
    """Guards the claim the headline numbers rest on."""
    assert metrics["provenance"]["selection_split"] == "val"
    assert metrics["provenance"]["reported_split"] == "test"
