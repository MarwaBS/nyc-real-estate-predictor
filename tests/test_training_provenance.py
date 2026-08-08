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
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

import run_training

ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = ROOT / "reports" / "training_metrics.json"


def test_every_estimator_is_built_single_threaded() -> None:
    """With n_jobs=-1 the thread count decides the order the float sums
    accumulate. Two Linux runs of one commit scored val R2 0.7740 and 0.7719,
    and the lower one shipped a different candidate."""
    tree = ast.parse((ROOT / "run_training.py").read_text(encoding="utf-8"))
    threads = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "n_jobs"
    ]
    assert threads, "no n_jobs argument left; the estimators moved"
    parallel = [
        ast.unparse(value)
        for value in threads
        if not (isinstance(value, ast.Constant) and value.value == 1)
    ]
    assert not parallel, f"estimators built with a thread count of {parallel}"


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
    # A distinct answer per call. A constant return_value cannot tell the first
    # sample from a second one handed to the writer after the models are on disk,
    # which is the same defect wearing a different shape.
    manager.sample.side_effect = [False, True, True, True]
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
        write_manifest=manager.manifest,
        MODELS_DIR=tmp_path / "models",
    ):
        monkeypatch.chdir(tmp_path)
        run_training.main()

    order = [call[0] for call in manager.mock_calls]
    assert order.count("sample") == 1, (
        f"the working tree is sampled {order.count('sample')} times; a second "
        "sample reads a tree this run has already written to"
    )
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


def test_the_working_tree_sampler_reports_both_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordering test above replaces this function with a mock, so nothing
    else executes it. Hardcoding its body to ``return True`` left the whole
    suite green while the flag could only report the good outcome."""

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "initial")

    monkeypatch.chdir(tmp_path)
    assert run_training._git_working_tree_clean() is True, (
        "a committed tree with no edits must read clean"
    )
    tracked.write_text("two\n", encoding="utf-8")
    assert run_training._git_working_tree_clean() is False, (
        "an edited tracked file must read dirty"
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
