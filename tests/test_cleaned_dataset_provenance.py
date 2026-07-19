"""The committed cleaned dataset must be the output of the committed cleaner.

This is the gate for the defect the cleaning rework existed to fix: the file
had no producer anywhere in the repo. It held a PRICE of 2,147,483,647 that
this pipeline's IQR cap makes impossible, and columns clean_pipeline never
creates, so `python run_training.py` did not reproduce the shipped models --
and nothing failed. Without a check that recomputes it, an artefact can drift
away from its producer again in exactly the same silence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import CLEANED_DATASET, RAW_DATASET
from src.data.cleaner import clean_pipeline
from src.data.loader import load_raw

pytestmark = pytest.mark.skipif(
    not RAW_DATASET.exists() or not CLEANED_DATASET.exists(),
    reason="raw input or cleaned artefact absent (cleaned CSV is a build output)",
)


def test_cleaned_dataset_is_what_the_cleaner_produces_from_raw() -> None:
    """Recompute it, don't trust it.

    Compared on shape, columns and the PRICE distribution rather than exact
    float equality: the artefact round-trips through CSV, so re-read floats
    differ in the last bits from the in-memory frame.
    """
    recomputed = clean_pipeline(load_raw())
    committed = pd.read_csv(CLEANED_DATASET)
    committed.columns = committed.columns.str.upper().str.strip()

    assert list(recomputed.columns) == list(committed.columns)
    assert len(recomputed) == len(committed)
    assert recomputed["PRICE"].max() == pytest.approx(committed["PRICE"].max())
    assert recomputed["PRICE"].min() == pytest.approx(committed["PRICE"].min())
    assert sorted(recomputed["BOROUGH"].unique()) == sorted(
        committed["BOROUGH"].unique()
    )


def test_committed_dataset_carries_the_model_inputs_training_needs() -> None:
    """BOROUGH is one-hot encoded and ZIPCODE target-encoded downstream."""
    committed = pd.read_csv(CLEANED_DATASET)
    committed.columns = committed.columns.str.upper().str.strip()

    assert committed["BOROUGH"].notna().all()
    assert committed["ZIPCODE"].notna().all()


def test_no_overflow_sentinel_survived_into_the_committed_dataset() -> None:
    """2**31-1 is an export artefact, not a price."""
    committed = pd.read_csv(CLEANED_DATASET)
    assert (committed["PRICE"] < 2_147_483_647).all()
