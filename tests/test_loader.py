"""Tests for data loader module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.loader import load_cleaned, load_raw


def test_load_cleaned_reads_csv(tmp_path: Path) -> None:
    csv = tmp_path / "test.csv"
    csv.write_text("PRICE,BEDS,BATH,ZIPCODE\n500000,2,1.0,10022\n")
    df = load_cleaned(csv)
    assert len(df) == 1
    assert "PRICE" in df.columns
    assert df["ZIPCODE"].iloc[0] == "10022"


def test_load_cleaned_normalizes_zipcode(tmp_path: Path) -> None:
    csv = tmp_path / "test.csv"
    csv.write_text("PRICE,ZIPCODE\n100,10022.0\n200,073\n")
    df = load_cleaned(csv)
    assert df["ZIPCODE"].iloc[0] == "10022"


def test_load_raw_uppercases_columns(tmp_path: Path) -> None:
    csv = tmp_path / "test.csv"
    csv.write_text("price,beds,bath\n500000,2,1.0\n")
    df = load_raw(csv)
    assert "PRICE" in df.columns
    assert "BEDS" in df.columns


def test_load_cleaned_missing_file_raises() -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        load_cleaned(Path("/nonexistent/path.csv"))
