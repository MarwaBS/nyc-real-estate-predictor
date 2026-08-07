"""Every claim a reader can check, compared to the thing it describes.

Published figures against the artefacts. Tool config and the import graph are
in ``test_gate_scope.py``.

Hand-maintained figures rot: README, MODEL_CARD, CHANGELOG, the ADRs and the
public Space page have all carried figures from an earlier training run,
including a regressor named as LightGBM when the shipped model was XGBoost.
Nothing recomputed them, so nothing failed when they went stale.

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

from src.config import (
    CENTRAL_PARK,
    MANHATTAN_CENTER,
    ONEHOT_FEATURES,
    TARGET_ENCODED_FEATURES,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "reports" / "training_metrics.json").read_text("utf-8"))
BENCH = json.loads((ROOT / "benchmarks" / "results.json").read_text("utf-8"))
CAP_STUDY = json.loads((ROOT / "reports" / "cap_factor_study.json").read_text("utf-8"))


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
# pyproject.toml is prose-bearing metadata (description), so it is scanned too.
LIVE_DOCS = [
    "README.md",
    "MODEL_CARD.md",
    "DESIGN_DECISIONS.md",
    "deploy/huggingface/README.md",
    "deploy/huggingface/DEPLOY.md",
    "pyproject.toml",
]

# A line stating that something is gone must be allowed to name it.
_HISTORICAL = re.compile(
    r"earlier|previous|no longer|there is no|was removed|is gone|used to"
    r"|deleted|void|before 2026|superseded",
    re.IGNORECASE,
)

# Sentence end or table cell wall.
_CLAUSE_BREAK = re.compile(r"(?<=[.;:])\s|\|")


def _is_historical(line: str, start: int) -> bool:
    """Whether the claim at ``start`` sits in a clause marked historical. That
    clause alone: a whole-line filter exempted every other claim beside it."""
    edges = [0] + [m.end() for m in _CLAUSE_BREAK.finditer(line)] + [len(line)]
    for begin, end in zip(edges, edges[1:], strict=False):
        if begin <= start < end:
            return bool(_HISTORICAL.search(line[begin:end]))
    return False


# Components deleted from the codebase. A live doc naming one is describing
# software that does not exist.
DELETED_COMPONENTS = [
    "price_zone_best",
    "label_encoder",
    "DIST_NEAREST_SUBWAY",
    "SQFT_CATEGORY",
    "train_classification",
    "optimal_thresholds",
    "argmax",
]

# Symbols a notebook must not import or reference, deleted functions and the
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
    does not execute them, so a stale import of a deleted symbol ships unseen.
    ``02_eda_features`` did exactly that: it imported ``add_neighborhood_clusters``
    and recreated the ``DIST_NEAREST_SUBWAY`` proxy the feature module documents
    as removed. Markdown cells are scanned too, a deleted feature advertised in
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
        for name in DELETED_COMPONENTS
        if name in line and not _is_historical(line, line.index(name))
    ]
    assert not offenders, "live docs name deleted components:\n" + "\n".join(offenders)


def _study_values() -> set[float]:
    """Seed-study and baseline quantities quoted beside the point estimates.
    They are different quantities with their own gates, so the contradiction
    scans must not flag them, the claim window can straddle a compound
    sentence like "test R2 0.814 ± 0.028, zones macro F1 0.717 ± 0.020"."""
    values = {
        round(SEED_VARIANCE[k][s], 3)
        for k in (
            "test_r2",
            "zones_macro_f1",
            "baseline_test_r2",
            "baseline_zones_macro_f1",
        )
        for s in ("mean", "std")
    }
    values |= {
        round(METRICS["baseline"]["test_r2"], 3),
        round(METRICS["baseline"]["test_zones_macro_f1"], 3),
        round(abs(BENCH["performance"]["baseline_r2_log_space"]), 3),
    }
    return values


#: Any decimal of two places or more. Matching only `0.` left the served
#: multiplier 1.457 unscanned. The trailing guard drops versions (`1.26.4`).
_DECIMAL = re.compile(r"(?<![\d.])(\d+\.\d{2,})(?!\.?\d)")


def _artefact_floats() -> set[float]:
    """Every fractional quantity the artefacts hold, unrounded."""
    values = {
        METRICS["regression"]["metrics"]["r2"],
        METRICS["regression"]["selection_metrics_val"]["r2"],
        METRICS["classification"]["metrics"]["macro_f1"],
        METRICS["classification"]["metrics"]["accuracy"],
        METRICS["regression"]["price_interval"]["coverage_test"],
        abs(BENCH["performance"]["r2_log_space"]),
        # The leaked PRICE_PER_SQFT figure ADR-001 exists to document. Named
        # deliberately and repeatedly, so it is allowed by value.
        0.997,
    }
    values |= _study_values()
    values |= set(METRICS["classification"]["fairness_by_borough"].values())
    values |= {
        figure
        for entry in METRICS["classification"]["borough_floor"].values()
        for key, figure in entry.items()
        if key != "n"
    }
    values |= {row["mean_abs_shap"] for row in METRICS["classification"]["shap_top10"]}
    values |= {row["val_mae_common"] for row in CAP_STUDY["rows"]}
    values |= {row["val_r2_common"] for row in CAP_STUDY["rows"]}
    values |= {row["pct_at_cap"] for row in CAP_STUDY["rows"]}
    values.add(BENCH["leakage_tripwire"]["threshold"])
    # The Haversine anchors, from the constants the features are built from.
    values |= {abs(c) for anchor in (MANHATTAN_CENTER, CENTRAL_PARK) for c in anchor}
    values |= {
        figure
        for key, figure in CAP_STUDY.items()
        if isinstance(figure, float) and key != "shipped_factor"
    }
    values |= {
        figure
        for key, figure in METRICS["regression"]["price_interval"].items()
        if isinstance(figure, (int, float))
    }
    # The cap paragraph's gain is the difference between two figures above.
    mae = {row["factor"]: row["val_mae_common"] for row in CAP_STUDY["rows"]}
    values.add(round(mae[CAP_STUDY["shipped_factor"]] - mae[1.5], 4))
    return values | set(_UNGATED_FIGURES)


#: Figures the docs state that no artefact holds. The set is closed: each is a
#: one-off measurement or a superseded value the prose names as such, and adding
#: one means editing this list, which is a visible act in review.
_UNGATED_FIGURES = {
    0.014,  # the retracted threshold-tuning gain, named as in-sample
    0.0006,  # its honest out-of-sample re-measurement, run once, no artefact
    0.0106,  # the standard deviation of that same study
    0.375,  # the superseded benchmark score, named as never reproducible
    3.12,  # the Python version, stated in the environment tables
    1.33,  # the binomial standard error MODEL_CARD derives in prose
}


@pytest.mark.parametrize("doc", LIVE_DOCS)
def test_every_two_place_figure_in_a_live_doc_is_in_the_artefacts(doc: str) -> None:
    """Every decimal of two places or more, not only those beside a metric name.

    Two bounds: a figure written to one place, and a real figure quoted against
    the wrong metric, because membership does not know which quantity owns
    which number."""
    allowed = _artefact_floats()
    wrong = []
    for i, line in enumerate(_read(doc).splitlines(), 1):
        for match in _DECIMAL.finditer(line):
            if _is_historical(line, match.start()):
                continue
            value = match.group(1)
            places = len(value.split(".")[1])
            # Compared at the precision the document chose, so a figure written
            # to more places than the artefact carries is still checked.
            if float(value) not in {round(v, places) for v in allowed}:
                wrong.append(f"{doc}:{i}: {value} is in no artefact")
    assert not wrong, "figures no artefact holds:\n" + "\n".join(wrong)


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


@pytest.mark.parametrize("doc", ["README.md", "deploy/huggingface/README.md"])
def test_reader_facing_docs_declare_the_shipped_model(doc: str) -> None:
    """Prose in these files names model families in candidate, comparison and
    historical contexts, so scanning it for the shipped-model claim never
    converged. Each reader-facing surface instead carries one machine-readable
    field, compared to the artefact by string equality."""
    shipped = METRICS["regression"]["selected_model"]
    claimed = re.search(r"\*\*Shipped model\*\*\s*[-—]\s*`([^`]+)`", _read(doc))
    assert claimed is not None, f"{doc} no longer declares a **Shipped model** field"

    def normalise(text: str) -> str:
        return text.lower().replace(" ", "_")

    assert normalise(claimed.group(1)) == normalise(shipped), (
        f"{doc} declares {claimed.group(1)}; the artefact ships {shipped}"
    )


def test_pyproject_description_names_the_shipped_model() -> None:
    """Package metadata sat outside every scan above: the description still
    said Random Forest after the shipped regressor changed to XGBoost."""
    match = re.search(r'^description = "([^"]*)"', _read("pyproject.toml"), re.M)
    assert match is not None, "pyproject.toml no longer carries a description"
    described = match.group(1).lower().replace(" ", "_")

    shipped = METRICS["regression"]["selected_model"]
    assert shipped in described, (
        f"pyproject description does not name the shipped model {shipped}"
    )
    stale = sorted(
        name
        for name in METRICS["regression"]["candidates_val"]
        if name != shipped and name in described
    )
    assert not stale, f"pyproject description names non-shipped model(s): {stale}"


def test_every_relative_link_in_the_docs_resolves() -> None:
    """A renamed or deleted file leaves the prose pointing at nothing, and the
    reader finds out instead of CI. External URLs are out of scope: they fail
    for reasons that have nothing to do with this commit."""
    broken = []
    for doc in sorted(ROOT.rglob("*.md")):
        if any(part.startswith(".venv") or part == ".git" for part in doc.parts):
            continue
        text = doc.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]]*\]\(([^)#\s]+)\)", text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (doc.parent / target).exists():
                line = text[: match.start()].count("\n") + 1
                broken.append(f"{doc.relative_to(ROOT).as_posix()}:{line} -> {target}")
    assert not broken, "markdown links pointing at nothing:\n" + "\n".join(broken)


def test_the_cap_factor_paragraph_quotes_the_study() -> None:
    """MODEL_CARD's cap paragraph is the longest derivation in the doc set and
    had no producer: its dollar figures were the pooled fit, left behind when
    cap bounds moved to a train-only fit, and its MAE figures were from an
    earlier run of the study."""
    card = _read("MODEL_CARD.md")
    low, high = CAP_STUDY["train_fit_price_bounds"]
    quoted = [
        f"${high:,.0f}",
        f"−${abs(low):,.0f}",
        f"${CAP_STUDY['common_support_ceiling']:,.0f}",
    ]
    quoted += [f"{row['val_mae_common']:.4f}" for row in CAP_STUDY["rows"]]
    # Only the two factors the paragraph weighs against each other: the
    # percentages are what it uses to justify keeping 3.0 over 1.5.
    compared = {CAP_STUDY["shipped_factor"], 1.5}
    quoted += [
        f"{row['pct_at_cap']}%"
        for row in CAP_STUDY["rows"]
        if row["factor"] in compared
    ]
    missing = [value for value in quoted if value not in card]
    assert not missing, f"MODEL_CARD does not quote the cap study: {missing}"


def test_the_benchmark_artefact_names_a_registered_contract_version() -> None:
    """The contract may move ahead of the artefact, but the version the artefact
    records must still be one the registry seals, with the hash it was sealed
    under. Otherwise the recorded provenance points at nothing."""
    registry = json.loads(_read("benchmarks/SCHEMA_MAP_VERSIONS.json"))["versions"]
    version = BENCH["schema_map_version"]
    assert version in registry, (
        f"results.json records {version}, absent from the registry"
    )
    assert registry[version] == BENCH["schema_map_sha256"], (
        f"results.json records a {version} hash the registry does not seal"
    )


def test_readme_states_the_contract_version_the_results_were_produced_under() -> None:
    """The contract can be bumped without re-running the benchmark, so the two
    versions legitimately differ. What must not happen is the README calling the
    older artefact current: it said "SCHEMA_MAP v3" while the live contract was
    v4, which is the sealed document contradicting its own seal."""
    produced_under = BENCH["schema_map_version"]
    stated = re.search(r"produced under SCHEMA_MAP (v\d+)", _read("README.md"))
    assert stated is not None, (
        "README no longer says which contract version the results came from"
    )
    assert stated.group(1) == produced_under, (
        f"README says the results were produced under {stated.group(1)}; "
        f"benchmarks/results.json records {produced_under}"
    )


def test_the_capped_target_caveat_is_stated_and_quoted() -> None:
    """The headline R2 is scored against a clipped target. The study is tied to
    the training artefact first: on its own it can drift with the prose and stay
    green, because its own figures feed the doc scan."""
    assert CAP_STUDY["shipped_model_test_r2_capped_target"] == round(
        METRICS["regression"]["metrics"]["r2"], 4
    ), "the study scored a different model than the one that shipped"
    assert CAP_STUDY["n_test"] == METRICS["provenance"]["n_test"], (
        "the study used a different test split than the shipped run"
    )

    listed = CAP_STUDY["shipped_model_test_r2_listed_target"]
    censored = CAP_STUDY["test_rows_censored_by_the_cap"]
    for doc in ("README.md", "MODEL_CARD.md"):
        text = _read(doc)
        assert f"{listed:.4f}" in text, f"{doc} does not quote the listed-target R2"
        assert f"{censored} of {CAP_STUDY['n_test']}" in text, (
            f"{doc} does not state how many test rows the cap censors"
        )


#: Counts and money figures a live doc states that no artefact holds. Closed
#: set: an approximation, an illustrative quantity, and a data constant.
_UNGATED_INTEGERS = {
    4_500,  # "4,500+ listings", rounded down from the cleaned count
    1_000,  # illustrative ZIP cardinality in the leakage-scope note
    2_147_483_647,  # the int32 overflow sentinel the cleaner drops
    18_314,  # rows behind the superseded 0.375 benchmark score
    1_130,  # "~1,130 listings each", the quartile size rounded
}

_SEPARATED_INTEGER = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d])")


def _artefact_integers() -> set[int]:
    provenance = METRICS["provenance"]
    counts = {provenance[k] for k in ("n_train", "n_val", "n_test")}
    values = set(counts) | {sum(counts)}
    values |= {
        BENCH["n_scored"],
        BENCH["n_dropped"],
        BENCH["n_scored"] + BENCH["n_dropped"],
    }
    values |= set(BENCH["drop_reasons"].values())
    values |= {CAP_STUDY["n_train"], CAP_STUDY["n_test"]}
    values |= {row["train_rows_at_cap"] for row in CAP_STUDY["rows"]}
    values |= {int(abs(b)) for b in CAP_STUDY["train_fit_price_bounds"]}
    values.add(int(CAP_STUDY["common_support_ceiling"]))
    # Zone cut-points appear both in full and in thousands.
    for cut in provenance["price_zone_bins"]:
        values |= {int(cut), int(cut) // 1_000}
    raw = (ROOT / "Resources" / "NY-House-Dataset.csv").read_text(
        encoding="utf-8", errors="ignore"
    )
    # Raw rows: the committed CSV's data lines, header excluded.
    values.add(len([line for line in raw.splitlines() if line.strip()]) - 1)
    return values | _UNGATED_INTEGERS


@pytest.mark.parametrize("doc", LIVE_DOCS)
def test_every_separated_count_in_a_live_doc_is_in_the_artefacts(doc: str) -> None:
    """Row counts and money figures rot the same way fractions do, and the
    fractional scan does not see them: 4,526 could be rewritten to 9,999 with
    the suite green.

    Comma-separated only: a bare run of four digits here is usually a year, a
    port or a colour code, and exempting years would exempt a count written as
    2024."""
    allowed = _artefact_integers()
    wrong = []
    for i, line in enumerate(_read(doc).splitlines(), 1):
        for match in _SEPARATED_INTEGER.finditer(line):
            if _is_historical(line, match.start()):
                continue
            if int(match.group(0).replace(",", "")) not in allowed:
                wrong.append(f"{doc}:{i}: {match.group(0)} is in no artefact")
    assert not wrong, "counts no artefact holds:\n" + "\n".join(wrong)


def test_readme_shap_table_matches_the_artefact() -> None:
    """README once carried the previous model's SHAP values (BATH 0.446 …)
    under a false 'read from the artefact' heading. The doc gates covered R²
    and the model name but not this table."""
    readme = _read("README.md")
    shap = METRICS["classification"]["shap_top10"]
    for row in shap[:4]:
        # ColumnTransformer prefixes (num__, cat_target__) are stripped in the
        # prose, so match the bare feature name and its 3-dp value.
        feature = row["feature"].split("__", 1)[-1].split("_")[0]
        value = f"{row['mean_abs_shap']:.3f}"
        pattern = rf"{re.escape(feature)}[^|]*\|\s*{re.escape(value)}"
        assert re.search(pattern, readme, re.IGNORECASE), (
            f"README SHAP table does not carry {feature} = {value} "
            f"(artefact rank {shap.index(row) + 1})"
        )
    # The stale table's top value must be gone specifically.
    assert "0.446" not in readme, "README still carries the prior model's SHAP 0.446"


def test_the_stated_feature_count_matches_the_fitted_model() -> None:
    """MODEL_CARD claimed 14 features (10 numeric) against an artefact holding
    12 (8 numeric), the count was not updated when two were removed."""
    features = METRICS["provenance"]["features"]
    card = _read("MODEL_CARD.md")
    stated = re.search(
        r"\*\*Feature set:\*\*\s*(\d+)\s*total\s*[-—]\s*(\d+)\s*numeric", card
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


SEED_VARIANCE = json.loads((ROOT / "reports" / "seed_variance.json").read_text("utf-8"))


def test_seed_variance_claims_match_the_recorded_study() -> None:
    """The mean ± std quoted in README/MODEL_CARD must be the recorded ones."""
    r2 = SEED_VARIANCE["test_r2"]
    f1 = SEED_VARIANCE["zones_macro_f1"]
    r2_claim = f"{r2['mean']:.3f} ± {r2['std']:.3f}"
    f1_claim = f"{f1['mean']:.3f} ± {f1['std']:.3f}"
    for doc in ("README.md", "MODEL_CARD.md"):
        text = _read(doc)
        assert r2_claim in text, f"{doc} does not quote test R² {r2_claim}"
        assert f1_claim in text, f"{doc} does not quote zones F1 {f1_claim}"

    # _study_values() exempts these from the contradiction scans.
    readme = _read("README.md")
    for key in ("baseline_test_r2", "baseline_zones_macro_f1"):
        spread = SEED_VARIANCE[key]
        claim = f"{spread['mean']:.3f} ± {spread['std']:.3f}"
        assert claim in readme, f"README does not quote the {key} spread {claim}"
    # The selection-count claim must match too.
    counts = SEED_VARIANCE["selected_model_counts"]
    winner = max(counts, key=lambda k: counts[k])
    n = SEED_VARIANCE["n_seeds"]
    assert f"{counts[winner]}/{n}" in _read("README.md")


def test_the_shipped_seed_metrics_sit_inside_the_recorded_spread() -> None:
    """The headline artefact must be a draw from the distribution the study
    describes, a stale study file would let the two drift apart."""
    r2 = METRICS["regression"]["metrics"]["r2"]
    spread = SEED_VARIANCE["test_r2"]
    assert spread["min"] - 1e-9 <= r2 <= spread["max"] + 1e-9


def test_benchmark_baseline_is_recorded_and_quoted() -> None:
    """README's benchmark table must quote the baseline scored on the same
    rows, from the artefact."""
    baseline = BENCH["performance"]["baseline_r2_log_space"]
    assert baseline is not None
    assert f"{baseline:.3f}" in _read("README.md"), (
        f"README does not quote the benchmark baseline {baseline:.3f}"
    )


def test_split_sizes_are_quoted_consistently() -> None:
    prov = METRICS["provenance"]
    split = (
        f"{prov['n_train']:,} train / {prov['n_val']:,} val / {prov['n_test']:,} test"
    )
    assert split in _read("README.md"), f"README does not state the split as {split}"
