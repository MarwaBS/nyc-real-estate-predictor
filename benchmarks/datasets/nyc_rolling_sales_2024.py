"""Download and parse NYC.gov Rolling Sales 2024 into a single DataFrame.

Data source: https://www.nyc.gov/site/finance/property/property-rolling-sales-data.page

The City of New York publishes one Excel file per borough per rolling
12-month window. This module downloads all five, concatenates them into
a single raw frame, and hands it to :func:`apply_schema_map` unchanged.
Schema transformation lives in ``benchmarks/mapping.py`` by contract.

Contract obligations:

- Network IO is the only thing this module is allowed to do.
- No cleaning, no filtering, no column renaming. The raw NYC.gov
  schema lands in the caller's hands verbatim (modulo pandas' Excel
  parsing conventions: header row detection, dtype inference).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests


NYC_ROLLING_SALES_URLS: dict[str, str] = {
    "manhattan": "https://www.nyc.gov/assets/finance/downloads/pdf/rolling_sales/rollingsales_manhattan.xlsx",
    "bronx": "https://www.nyc.gov/assets/finance/downloads/pdf/rolling_sales/rollingsales_bronx.xlsx",
    "brooklyn": "https://www.nyc.gov/assets/finance/downloads/pdf/rolling_sales/rollingsales_brooklyn.xlsx",
    "queens": "https://www.nyc.gov/assets/finance/downloads/pdf/rolling_sales/rollingsales_queens.xlsx",
    "statenisland": "https://www.nyc.gov/assets/finance/downloads/pdf/rolling_sales/rollingsales_statenisland.xlsx",
}

HEADER_ROW_CANDIDATES = (4, 3, 5)  # NYC publishes with a 4-line metadata header


@dataclass(frozen=True)
class DownloadManifest:
    """Per-borough download record for reproducibility logging."""

    borough: str
    url: str
    bytes_downloaded: int
    sha256: str


def _locate_header_row(content: bytes) -> int:
    """Find the row that contains the canonical NYC.gov column names.

    The published file layout has ~4 lines of agency metadata before
    the header. We probe a small set of candidates looking for 'SALE
    PRICE' to identify the real header; if none match, the caller
    gets a clear error rather than silently misaligned columns.
    """
    for header_row in HEADER_ROW_CANDIDATES:
        try:
            probe = pd.read_excel(
                io.BytesIO(content),
                engine="openpyxl",
                header=header_row,
                nrows=0,
            )
        except Exception:
            continue
        columns = {str(c).strip().upper() for c in probe.columns}
        if "SALE PRICE" in columns:
            return header_row
    raise RuntimeError(
        "could not locate 'SALE PRICE' header in NYC.gov xlsx "
        f"after probing rows {HEADER_ROW_CANDIDATES}"
    )


def _fetch(url: str, *, timeout: int = 60) -> bytes:
    headers = {"User-Agent": "nyc-real-estate-benchmark/0.1 (MarwaBS)"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content


def download_nyc_rolling_sales() -> tuple[pd.DataFrame, list[DownloadManifest]]:
    """Download all five borough xlsx files and return a concatenated frame.

    Returns:
        A tuple ``(raw, manifests)``:

        - ``raw``: concatenated DataFrame of all five boroughs with
          NYC.gov's published column names, whitespace-stripped.
        - ``manifests``: one :class:`DownloadManifest` per borough,
          stamped with URL, byte count, and content SHA256 for the
          benchmark reproducibility log.
    """
    import hashlib

    frames: list[pd.DataFrame] = []
    manifests: list[DownloadManifest] = []

    for borough, url in NYC_ROLLING_SALES_URLS.items():
        content = _fetch(url)
        header_row = _locate_header_row(content)
        frame = pd.read_excel(
            io.BytesIO(content),
            engine="openpyxl",
            header=header_row,
        )
        frame.columns = [str(c).strip() for c in frame.columns]
        frames.append(frame)
        manifests.append(
            DownloadManifest(
                borough=borough,
                url=url,
                bytes_downloaded=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )

    raw = pd.concat(frames, ignore_index=True)
    return raw, manifests


__all__ = [
    "NYC_ROLLING_SALES_URLS",
    "DownloadManifest",
    "download_nyc_rolling_sales",
]
