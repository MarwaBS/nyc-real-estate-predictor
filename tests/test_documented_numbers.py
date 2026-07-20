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

from src.config import ONEHOT_FEATURES, TARGET_ENCODED_FEATURES

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
    claimed = re.search(r"\*\*Regressor\*\*\s*[-—]\s*`([^`]+)`", card)
    assert claimed is not None, "MODEL_CARD no longer states a regressor"

    # Normalised: the artefact says "random_forest", prose says "Random Forest".
    def normalise(s: str) -> str:
        return s.lower().replace(" ", "_")

    assert normalise(claimed.group(1)) == normalise(shipped), (
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


# Docs describing what ships today. CHANGELOG and the ADRs are excluded from
# the scans below: their older entries are dated records of what was true then.
LIVE_DOCS = [
    "README.md",
    "MODEL_CARD.md",
    "deploy/huggingface/README.md",
    "deploy/huggingface/DEPLOY.md",
]

# A line stating that something is gone must be allowed to name it. Matched
# against the text PRECEDING the claim, not the whole line: "macro F1 = 0.699
# ... There is no classifier" is a live claim followed by a historical remark,
# and a whole-line filter exempted it — the mutation to 0.727 passed.
_HISTORICAL = re.compile(
    r"earlier|previous|no longer|there is no|was removed|is gone|used to"
    r"|deleted|void|before 2026|superseded"
    # A measurement of a rejected variant is a different quantity, not a
    # contradicted claim — the cap-factor derivation reports one for each
    # candidate factor.
    r"|uncapped|variant",
    re.IGNORECASE,
)

# Components deleted from the codebase. A live doc naming one is describing
# software that does not exist.
DELETED_COMPONENTS = [
    "price_zone_best",
    "label_encoder",
    "DIST_NEAREST_SUBWAY",
    "SQFT_CATEGORY",
    "train_classification",
    "optimal_thresholds",
]

# Symbols a notebook must not import or reference — deleted functions and the
# never-shipped feature ideas explored in early EDA. A notebook that names one
# either fails to import or demonstrates an architecture the repo abandoned.
DELETED_NOTEBOOK_SYMBOLS = [
    "add_neighborhood_clusters",
    "add_h3_index",
    "nearest_station_distance",
    "get_label_encoder",
    "build_classification_pipeline",
    "train_classification",
    "price_zone_best",
    "label_encoder",
    "DIST_NEAREST_SUBWAY",
    "SQFT_CATEGORY",
    "SQFT_BINS",
    "SQFT_LABELS",
    "_HAS_TARGET_ENCODER",
    "NEIGHBORHOOD_CLUSTER",
    "LOG_SQFT",
    "dl_multitask",
    "MultiTask",
]

NOTEBOOKS = sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "notebooks").glob("*.ipynb")
)


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_notebooks_reference_no_deleted_symbol(nb: str) -> None:
    """Notebooks are committed portfolio artefacts a reviewer will open, but CI
    does not execute them — so a stale import of a deleted symbol ships unseen.
    ``02_eda_features`` did exactly that: it imported ``add_neighborhood_clusters``
    and recreated the ``DIST_NEAREST_SUBWAY`` proxy the feature module documents
    as removed. Markdown cells are scanned too — a deleted feature advertised in
    prose is the same stale claim as one imported in code."""
    cells = json.loads((ROOT / nb).read_text(encoding="utf-8"))["cells"]
    text = "\n".join(
        "".join(c["source"]) for c in cells if c["cell_type"] in ("code", "markdown")
    )
    named = sorted({s for s in DELETED_NOTEBOOK_SYMBOLS if s in text})
    assert not named, f"{nb} references deleted symbols: {named}"


@pytest.mark.parametrize("doc", LIVE_DOCS)
def test_no_live_doc_describes_a_component_that_was_deleted(doc: str) -> None:
    """Presence checks cannot catch this: the Space page described an XGBoost
    classifier and a label encoder while every number in it was correct."""
    offenders = [
        f"{doc}:{i}: {name}"
        for i, line in enumerate(_read(doc).splitlines(), 1)
        if not _HISTORICAL.search(line)
        for name in DELETED_COMPONENTS
        if name in line
    ]
    assert not offenders, "live docs name deleted components:\n" + "\n".join(offenders)


def _claimed_values(text: str, keyword: str) -> list[tuple[int, float]]:
    """Every number stated within 60 characters after `keyword`, with its line."""
    found = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in re.finditer(keyword, line, re.IGNORECASE):
            if _HISTORICAL.search(line[: m.start()]):
                continue
            window = line[m.end() : m.end() + 60]
            found.extend(
                (i, float(v)) for v in re.findall(r"\*{0,2}(0\.\d{3})\*{0,2}", window)
            )
    return found


@pytest.mark.parametrize("doc", LIVE_DOCS)
def test_every_live_macro_f1_claim_matches_the_artefact(doc: str) -> None:
    """Asserting the true value appears somewhere passes while a contradictory
    figure sits three lines away — which is how 0.727 survived beside 0.699."""
    shipped = round(METRICS["classification"]["metrics"]["macro_f1"], 3)
    wrong = [
        f"{doc}:{line}: macro F1 {value} (artefact: {shipped})"
        for line, value in _claimed_values(_read(doc), r"macro[- ]?F1")
        if value != shipped
    ]
    assert not wrong, "contradicted macro F1 claims:\n" + "\n".join(wrong)


@pytest.mark.parametrize("doc", LIVE_DOCS)
def test_every_live_test_r2_claim_matches_the_artefact(doc: str) -> None:
    """The external benchmark's R2(log) is a different quantity, so only the
    in-distribution values are compared."""
    shipped = round(METRICS["regression"]["metrics"]["r2"], 3)
    val = round(METRICS["regression"]["selection_metrics_val"]["r2"], 3)
    bench = round(BENCH["performance"]["r2_log_space"], 3)
    # 0.997 is the leaked PRICE_PER_SQFT figure ADR-001 exists to document. It
    # is named deliberately and repeatedly, so it is allowed by value rather
    # than by loosening the historical-phrasing filter every line must pass.
    allowed = {shipped, val, bench, 0.997}
    wrong = [
        f"{doc}:{line}: R2 {value} (artefact test {shipped} / val {val})"
        for line, value in _claimed_values(_read(doc), r"R[²2]")
        if value not in allowed
    ]
    assert not wrong, "contradicted R2 claims:\n" + "\n".join(wrong)


@pytest.mark.parametrize("doc", ["README.md", "MODEL_CARD.md"])
def test_per_borough_f1_is_quoted_from_the_artefact(doc: str) -> None:
    """The ungated table that carried "0.887 Staten Island vs 0.601 Queens"
    three lines under its own correct figures."""
    text = _read(doc)
    for borough, macro_f1 in METRICS["classification"]["fairness_by_borough"].items():
        # Table cell or bullet only. A looser gap matched the SHAP entry
        # "DIST_MANHATTAN_CENTER (0.343)" as a Manhattan borough score, so the
        # name must not be part of a longer identifier.
        stated = re.findall(
            rf"\b{re.escape(borough)}\b(?![_\w])[\s|]*(0\.\d{{3}})",
            text,
            re.IGNORECASE,
        )
        assert stated, f"{doc} does not report a macro F1 for {borough}"
        expected = f"{macro_f1:.3f}"
        assert set(stated) == {expected}, (
            f"{doc} states {borough} macro F1 {sorted(set(stated))}; "
            f"the artefact says {expected}"
        )


@pytest.mark.parametrize(
    "doc", ["MODEL_CARD.md", "docs/decisions/002-xgboost-primary-model.md"]
)
def test_docs_do_not_name_an_unselected_model_as_shipped(doc: str) -> None:
    """ADR-002 announced "both shipped models are now XGBoost" while the
    artefact held a Random Forest and there was only one model."""
    shipped = METRICS["regression"]["selected_model"].replace("_", " ")
    others = {"xgboost", "lightgbm", "random forest"} - {shipped}
    wrong = [
        f"{doc}:{i}: names {name} as selected/shipped"
        for i, line in enumerate(_read(doc).splitlines(), 1)
        if re.search(r"selected|shipped", line, re.IGNORECASE)
        and not re.search(r"not shipped|never|compared on val", line, re.IGNORECASE)
        and not _HISTORICAL.search(line)
        for name in others
        if re.search(name, line, re.IGNORECASE)
    ]
    assert not wrong, f"the artefact ships {shipped}:\n" + "\n".join(wrong)


# Every file that presents the system to a reader, not only the markdown docs:
# the Space page, the dashboard footer, the Dockerfile comments and the pin
# comments all carried "gradient boosting"/"XGBoost" while the artefact shipped
# a Random Forest — and the model-name gate covered none of them.
MODEL_CLAIM_FILES = [
    *LIVE_DOCS,
    "Dockerfile",
    "streamlit_app/app.py",
    "requirements.txt",
]

# A non-shipped-family mention is legitimate only as a candidate/comparison,
# a version pin, a wheel/runtime note, or the ADR filename.
_CANDIDATE_CONTEXT = re.compile(
    r"compar|candidat|not shipped| vs |==|wheels|002-xgboost", re.IGNORECASE
)

# Model families the protocol compares; the artefact names the shipped one.
_FAMILY_PATTERNS = {
    "random_forest": r"random.?forest",
    "xgboost": r"xgboost",
    "lightgbm": r"lightgbm",
}


@pytest.mark.parametrize("doc", MODEL_CLAIM_FILES)
def test_no_surface_presents_an_unselected_family_as_shipped(doc: str) -> None:
    """The Space page, Dockerfile, footer and pin comments all once carried
    "XGBoost" while the artefact shipped a Random Forest — outside the reach
    of the MODEL_CARD-only gate."""
    shipped = METRICS["regression"]["selected_model"]
    assert shipped in _FAMILY_PATTERNS, f"unknown family {shipped!r} — extend the map"
    others = "|".join(p for f, p in _FAMILY_PATTERNS.items() if f != shipped)
    if shipped == "random_forest":
        # "gradient boosting" as a shipped-model description misleads exactly
        # when the shipped model is not a boosted tree.
        others += r"|gradient.?boost"
    wrong = [
        f"{doc}:{i}: {line.strip()[:70]}"
        for i, line in enumerate(_read(doc).splitlines(), 1)
        if re.search(others, line, re.IGNORECASE)
        and not _CANDIDATE_CONTEXT.search(line)
        and not _HISTORICAL.search(line)
    ]
    assert not wrong, (
        f"the shipped model is {shipped}; these lines present another family "
        "as shipped:\n" + "\n".join(wrong)
    )


def test_no_tracked_file_references_private_projects() -> None:
    """Comments cross-referencing the author's private repositories expose
    their names to every reader and explain nothing a local comment couldn't."""
    private_names = re.compile(r"resume.?forge", re.IGNORECASE)
    this_file = Path(__file__).resolve()
    offenders = []
    for pattern in ("**/*.py", "**/*.md", "**/*.yml", "**/*.toml", "**/*.cfg"):
        for path in ROOT.glob(pattern):
            if any(
                part in {".venv312", ".venv", ".git", "node_modules"}
                for part in path.parts
            ):
                continue
            if path.resolve() == this_file:  # the regex above matches itself
                continue
            if private_names.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(str(path.relative_to(ROOT)))
    for name in (".gitignore", ".trivyignore", "Dockerfile", "Makefile"):
        if (ROOT / name).exists() and private_names.search(
            (ROOT / name).read_text(encoding="utf-8", errors="ignore")
        ):
            offenders.append(name)
    assert not offenders, f"private project referenced in: {sorted(offenders)}"


def test_the_stated_coverage_gate_matches_ci() -> None:
    """README stated three different gate values (65, 88, 69) at once, all
    wrong. The CI workflow is the authority."""
    ci = _read(".github/workflows/ci.yml")
    gate = re.search(r"--cov-fail-under=(\d+)", ci)
    assert gate is not None, "ci.yml no longer runs a coverage gate"

    readme = _read("README.md")
    stated_gates = set(re.findall(r"(\d+)% coverage gate", readme))
    stated_cmds = set(re.findall(r"--cov-fail-under=(\d+)", readme))
    assert stated_gates | stated_cmds <= {gate.group(1)}, (
        f"CI gates at {gate.group(1)}%; README states {stated_gates | stated_cmds}"
    )


def test_the_stated_feature_count_matches_the_fitted_model() -> None:
    """MODEL_CARD claimed 14 features (10 numeric) against an artefact holding
    12 (8 numeric) — the count was not updated when two were removed."""
    features = METRICS["provenance"]["features"]
    card = _read("MODEL_CARD.md")
    stated = re.search(
        r"\*\*Feature set:\*\*\s*(\d+)\s*total\s*—\s*(\d+)\s*numeric", card
    )
    assert stated is not None, "MODEL_CARD no longer states a feature count"

    n_categorical = len(ONEHOT_FEATURES) + len(TARGET_ENCODED_FEATURES)
    assert int(stated.group(1)) == len(features), (
        f"MODEL_CARD says {stated.group(1)} features; the artefact lists {len(features)}"
    )
    assert int(stated.group(2)) == len(features) - n_categorical, (
        f"MODEL_CARD says {stated.group(2)} numeric; the artefact implies "
        f"{len(features) - n_categorical}"
    )


def test_split_sizes_are_quoted_consistently() -> None:
    prov = METRICS["provenance"]
    split = (
        f"{prov['n_train']:,} train / {prov['n_val']:,} val / {prov['n_test']:,} test"
    )
    assert split in _read("README.md"), f"README does not state the split as {split}"
