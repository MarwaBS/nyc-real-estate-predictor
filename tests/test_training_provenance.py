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
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

import run_training

ARTEFACT = Path(__file__).resolve().parents[1] / "reports" / "training_metrics.json"


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


STUB_PROTOCOL = {
    "reg_record": {"selected_model": "stub", "metrics": {}},
    "clf_record": {
        "selected_model": "stub",
        "metrics": {"accuracy": 0.0, "macro_f1": 0.0},
    },
    "best_pipeline": None,
    "splits": {name: (None, None) for name in ("train", "val", "test")},
    "n_train": 1,
    "n_val": 1,
    "n_test": 1,
    "features": ["f"],
    "zone_bins": [0.0, 1.0, 2.0, 3.0, float("inf")],
    "baseline": {"predictor": "stub"},
}


def test_the_sampled_tree_state_is_the_one_that_reaches_the_artefact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs ``main`` with the pipeline stubbed and asserts two things at run
    time: the sample happens before ``prepare_data`` writes the cleaned dataset,
    and the value it returned is the value written.

    Reading the source instead passes on two ordinary edits. Move the call into
    a helper and its definition still comes first lexically. Keep the call, drop
    its result and hardcode the flag, and every name is still where it was.
    """
    manager = mock.MagicMock()
    manager.sample.return_value = False
    manager.run_protocol.return_value = STUB_PROTOCOL
    manager.calibrate.return_value = {}
    manager.shap.return_value = []
    (tmp_path / "models").mkdir()

    with mock.patch.multiple(
        run_training,
        _git_working_tree_clean=manager.sample,
        prepare_data=manager.prepare_data,
        run_protocol=manager.run_protocol,
        calibrate_price_interval=manager.calibrate,
        shap_top10=manager.shap,
        save_drift_baseline=manager.save_drift,
        write_manifest=manager.manifest,
        MODELS_DIR=tmp_path / "models",
    ):
        monkeypatch.chdir(tmp_path)
        run_training.main()

    order = [call[0] for call in manager.mock_calls]
    assert "sample" in order, "main no longer samples the working tree"
    assert order.index("sample") < order.index("prepare_data"), (
        "the working tree is sampled after prepare_data has written the cleaned "
        "dataset, so the flag can only ever record False"
    )

    written = json.loads(
        (tmp_path / "reports" / "training_metrics.json").read_text(encoding="utf-8")
    )
    assert written["working_tree_clean"] is False, (
        "the artefact does not carry the value the sample returned"
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
