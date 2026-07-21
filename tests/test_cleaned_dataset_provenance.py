"""What the cleaner must produce from the real raw export.

The defect this guards: `output/cleaned_house_dataset.csv` was once a file with
no producer anywhere in the repo. It held a PRICE of 2,147,483,647 that this
pipeline's IQR cap makes impossible, and `BOROUGH`/`ZIPCODE` columns
`clean_pipeline` never created, so `python run_training.py` did not reproduce
the shipped models -- and nothing failed.

The artefact is no longer committed; it is a build output regenerated on every
training run. So there is no committed file to compare against, and a test that
recomputed it and diffed it against a freshly built copy would compare
`clean_pipeline` to itself. What is worth asserting is that the cleaner, run on
the committed raw export, actually yields the inputs training requires.

These run everywhere: `Resources/NY-House-Dataset.csv` is committed, so there is
no skipif and no environment in which this silently reports "skipped".
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.cleaner import INT32_MAX, clean_pipeline
from src.data.loader import load_raw


@pytest.fixture(scope="module")
def cleaned() -> pd.DataFrame:
    return clean_pipeline(load_raw())


def test_cleaner_yields_the_model_inputs_training_requires(
    cleaned: pd.DataFrame,
) -> None:
    """BOROUGH is one-hot encoded and ZIPCODE target-encoded downstream.

    Before the derivation existed, the raw export produced neither column and
    the failure surfaced only at predict time.
    """
    assert "BOROUGH" in cleaned.columns
    assert "ZIPCODE" in cleaned.columns
    assert cleaned["BOROUGH"].notna().all()
    assert cleaned["ZIPCODE"].notna().all()


def test_the_derivation_drops_exactly_the_rows_it_cannot_resolve(
    cleaned: pd.DataFrame,
) -> None:
    """Retention, not null-freeness, is what detects a weakened derivation.

    Rows whose borough cannot be derived are DROPPED, so every surviving row
    looks perfect no matter how weak the source chain is -- asserting "no
    nulls" passes even with the highest-coverage source removed. What degrades
    is how many rows survive.

    The count is exact rather than a ratio floor. 37 of the 4,563 deduplicated
    rows go: 36 have shifted geocode columns (LOCALITY reads "United States"
    and ADMINISTRATIVE_AREA_LEVEL_2 holds a ZIP) and 1 is the 2**31-1 overflow
    sentinel. A ratio threshold like ">= 0.99" cannot answer "why not 0.98?";
    this count can name every row it expects to lose.
    """
    from src.data.cleaner import deduplicate, normalize_text_columns

    deduped = len(normalize_text_columns(deduplicate(load_raw())))

    assert deduped == 4563, f"deduplication changed: {deduped} rows"
    assert deduped - len(cleaned) == 37, (
        f"expected to drop 36 underivable + 1 overflow sentinel = 37, "
        f"dropped {deduped - len(cleaned)}"
    )


def test_every_borough_resolves_to_a_canonical_nyc_name(cleaned: pd.DataFrame) -> None:
    """A neighbourhood name leaking through would become its own one-hot level."""
    assert set(cleaned["BOROUGH"].unique()) == {
        "manhattan",
        "brooklyn",
        "queens",
        "the bronx",
        "staten island",
    }


def test_the_overflow_sentinel_row_is_dropped_not_capped() -> None:
    """The sentinel row must LEAVE the dataset, not survive as a listing.

    Row count distinguishes drop from keep: cleaning the frame with and without
    the sentinel gives the same count only if the sentinel is dropped. A
    magnitude assertion would not — the downstream train-fitted cap would clip
    the value either way.
    """
    raw = load_raw()
    n_sentinel = int((raw["PRICE"] >= INT32_MAX).sum())
    assert n_sentinel == 1, "raw snapshot should hold exactly one 2**31-1 PRICE"

    with_sentinel = len(clean_pipeline(raw))
    without_sentinel = len(clean_pipeline(raw[raw["PRICE"] < INT32_MAX]))

    assert with_sentinel == without_sentinel, (
        "cleaning the frame with and without the sentinel gives different row "
        "counts, so the sentinel is being kept rather than dropped"
    )


def test_zipcodes_are_five_digit_strings(cleaned: pd.DataFrame) -> None:
    """ZIPCODE is target-encoded, so a malformed value becomes a real category."""
    zips = cleaned["ZIPCODE"].astype(str)
    assert zips.str.fullmatch(r"\d{5}").all()
