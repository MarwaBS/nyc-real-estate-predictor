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


def test_the_borough_chain_resolves_almost_every_deduplicated_row(
    cleaned: pd.DataFrame,
) -> None:
    """Retention, not null-freeness, is what detects a weakened derivation.

    Rows whose borough cannot be derived are DROPPED, so every surviving row
    looks perfect no matter how weak the source chain is -- asserting only
    "no nulls" passes even with the highest-coverage source removed. What
    actually degrades is how many rows survive.

    Measured: the SUBLOCALITY -> ADMINISTRATIVE_AREA_LEVEL_2 -> LOCALITY chain
    resolves 99.2% of the 4,563 deduplicated rows (36 unresolvable). Dropping
    SUBLOCALITY alone, the highest-coverage source at 78.5%, collapses this.
    """
    from src.data.cleaner import deduplicate, normalize_text_columns

    deduped = len(normalize_text_columns(deduplicate(load_raw())))
    retained = len(cleaned) / deduped

    assert retained >= 0.99, (
        f"borough/ZIP derivation retained {retained:.1%} of {deduped} "
        f"deduplicated rows; the measured chain retains 99.2%"
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
    """The sentinel row must LEAVE the dataset, not survive at the cap.

    Asserting a post-cap magnitude cannot detect this: cap_outliers clips the
    sentinel to the IQR bound ($4,483,000) unconditionally, so `max < 10M`
    holds whether or not the drop runs. Deleting the guarded line left that
    version of this test green — it asserted a property cap_outliers
    guarantees on its own.

    Row count is what distinguishes the two: dropping removes the row,
    capping keeps it.
    """
    raw = load_raw()
    n_sentinel = int((raw["PRICE"] >= INT32_MAX).sum())
    assert n_sentinel == 1, "raw snapshot should hold exactly one 2**31-1 PRICE"

    with_sentinel = len(clean_pipeline(raw))
    without_sentinel = len(clean_pipeline(raw[raw["PRICE"] < INT32_MAX]))

    assert with_sentinel == without_sentinel, (
        "cleaning the frame with and without the sentinel gives different row "
        "counts, so the sentinel is being kept (capped) rather than dropped"
    )


def test_zipcodes_are_five_digit_strings(cleaned: pd.DataFrame) -> None:
    """ZIPCODE is target-encoded, so a malformed value becomes a real category."""
    zips = cleaned["ZIPCODE"].astype(str)
    assert zips.str.fullmatch(r"\d{5}").all()
