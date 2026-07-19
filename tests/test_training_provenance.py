"""The provenance block must be able to tell the truth.

Two of its fields were previously incapable of it: ``working_tree_clean`` was
sampled after the run had written models/ and the cleaned dataset, so it read
``false`` even from a pristine checkout, and a hardcoded ``note`` asserted the
numbers were not reproducible by a stranger long after the raw CSV and the
models were committed. A provenance field that cannot report the good outcome
is not evidence, it is decoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_training

ARTEFACT = Path(__file__).resolve().parents[1] / "reports" / "training_metrics.json"


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


def test_working_tree_clean_is_sampled_before_the_run_writes_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag must reflect the tree at start, not the tree the run dirtied.

    Fails against the previous code, which called _git_working_tree_clean()
    inside the writer: by then prepare_data had written the cleaned CSV and
    the models, so a run starting from a pristine checkout still recorded
    false. Here a True is threaded in and must survive to the artefact.
    """
    monkeypatch.chdir(tmp_path)

    run_training._write_training_metrics(
        True,
        {"selected_model": "stub", "metrics": {}},
        {"selected_model": "stub", "metrics": {}},
        n_train=1,
        n_val=1,
        n_test=1,
        features=["f"],
    )

    written = json.loads((tmp_path / "reports" / "training_metrics.json").read_text())
    assert written["working_tree_clean"] is True


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
