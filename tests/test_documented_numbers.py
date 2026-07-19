"""Published headline numbers must match the artefacts they claim to quote.

Two consecutive audits found the same class of defect: README, MODEL_CARD,
CHANGELOG, the ADRs and the public Space page all carried figures from an
earlier training run, including a regressor named as LightGBM when the shipped
model was XGBoost. Every one of those was corrected by hand, which is exactly
why they rotted -- nothing recomputed them.

This reads the numbers back out of the prose and compares them to
``reports/training_metrics.json`` and ``benchmarks/results.json``. It is
deliberately narrow: it checks the headline claims a reader would act on, not
every digit in the repo, because a gate that tries to parse all prose becomes
noise and gets disabled.

What it deliberately does NOT do is blocklist superseded values. A bare search
for "0.800" matches the SHAP importance of PROPERTYSQFT as readily as the old
regression R2, so such a check fails on correct documents and has to be edited
after every retrain. Claims are pinned at their specific site instead -- the
regressor name is matched on the line that states it, not anywhere in the file.
A stale figure sitting somewhere this file does not name can still survive; the
coverage is the headline set, not the whole document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "reports" / "training_metrics.json").read_text("utf-8"))
BENCH = json.loads((ROOT / "benchmarks" / "results.json").read_text("utf-8"))


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


DOCS_QUOTING_TEST_R2 = ["README.md", "MODEL_CARD.md", "deploy/huggingface/README.md"]


@pytest.mark.parametrize("doc", DOCS_QUOTING_TEST_R2)
def test_documents_quote_the_artefacts_test_r2(doc: str) -> None:
    r2 = METRICS["regression"]["metrics"]["r2"]
    assert f"{r2:.3f}" in _read(doc), (
        f"{doc} does not quote the shipped test R2 {r2:.3f}"
    )


def test_documents_name_the_regressor_that_actually_shipped() -> None:
    """MODEL_CARD named LightGBM while the artefact was XGBoost."""
    shipped = METRICS["regression"]["selected_model"]
    card = _read("MODEL_CARD.md")
    claimed = re.search(r"\*\*Regressor\*\* — `(\w+)`", card)
    assert claimed is not None, "MODEL_CARD no longer states a regressor"
    assert claimed.group(1).lower() == shipped.lower(), (
        f"MODEL_CARD names {claimed.group(1)}; the artefact is {shipped}"
    )


def test_documents_quote_the_shipped_interval_coverage() -> None:
    coverage = METRICS["regression"]["price_interval"]["coverage_test"]
    as_pct = f"{coverage * 100:.1f}%"
    for doc in ("MODEL_CARD.md", "deploy/huggingface/README.md"):
        assert as_pct in _read(doc), f"{doc} does not quote coverage {as_pct}"


def test_readme_quotes_the_benchmark_score_and_row_count() -> None:
    readme = _read("README.md")
    r2 = BENCH["performance"]["r2_log_space"]
    assert f"{r2:.3f}" in readme
    assert f"{BENCH['n_scored']:,}" in readme


def test_readme_drop_table_reconciles_to_the_artefact() -> None:
    """The table advertises 'sum reconciles to n_dropped -- enforced at run
    time' and once summed 540 short of its own stated total."""
    readme = _read("README.md")
    for reason, count in BENCH["drop_reasons"].items():
        assert f"{count:,}" in readme, f"README omits {reason} = {count:,}"
    assert f"{BENCH['n_dropped']:,}" in readme
    assert sum(BENCH["drop_reasons"].values()) == BENCH["n_dropped"]


def test_split_sizes_are_quoted_consistently() -> None:
    prov = METRICS["provenance"]
    split = (
        f"{prov['n_train']:,} train / {prov['n_val']:,} val / {prov['n_test']:,} test"
    )
    assert split in _read("README.md"), f"README does not state the split as {split}"
