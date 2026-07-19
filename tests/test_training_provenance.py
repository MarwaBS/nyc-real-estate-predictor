"""The provenance block must be able to tell the truth.

Two of its fields were previously incapable of it: ``working_tree_clean`` was
sampled after the run had written models/ and the cleaned dataset, so it read
``false`` even from a pristine checkout, and a hardcoded ``note`` asserted the
numbers were not reproducible by a stranger long after the raw CSV and the
models were committed. A provenance field that cannot report the good outcome
is not evidence, it is decoration.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path

import pytest

import run_training

ARTEFACT = Path(__file__).resolve().parents[1] / "reports" / "training_metrics.json"


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


def test_working_tree_clean_is_sampled_before_main_writes_anything() -> None:
    """The sample must happen before the first write in ``main``, not after.

    Threading a True into ``_write_training_metrics`` and asserting True comes
    out cannot detect this bug: it never exercises the ordering the field
    depends on. Moving the sample back to where it was -- after prepare_data
    writes the cleaned dataset -- left that version of this test green and the
    original defect fully reintroduced.

    So assert the ordering directly, on the source of ``main``: the
    ``_git_working_tree_clean()`` call must appear before the first call that
    writes to disk. ``prepare_data`` writes CLEANED_DATASET, which is the
    earliest write in the run.
    """
    source = inspect.getsource(run_training.main)

    sample_at = source.find("_git_working_tree_clean()")
    first_write_at = source.find("prepare_data()")

    assert sample_at != -1, "main no longer samples the working tree at all"
    assert first_write_at != -1, "main no longer calls prepare_data"
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
