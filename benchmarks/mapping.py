"""Schema mapping from NYC.gov Rolling Sales 2024 to the model feature space.

Contract: ``benchmarks/SCHEMA_MAP.md`` (version pinned in
``SCHEMA_MAP_VERSIONS.json``).

**Step 3 status:** stub only. ``apply_schema_map`` raises
``NotImplementedError`` by design. The adversarial test suite in
``tests/benchmarks/test_schema_firewall.py`` is written against this
stub to confirm the firewall tests fail deterministically (TDD red
phase) before the real mapping is implemented in Step 4.

The ``MappingReport`` dataclass is frozen in Step 3 because the
adversarial test suite references its field names. Changing the shape
of ``MappingReport`` in Step 4 would require touching tests; keeping
it stable here lets Step 4 land as a pure implementation change.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MappingReport:
    """Per-run summary of a mapping application.

    Attributes:
        n_raw: number of rows in the input frame before filtering.
        n_dropped: number of rows removed by structural filters.
        n_scored: number of rows retained for downstream inference.
        drop_reasons: count per named drop reason (see SCHEMA_MAP.md §4).
            Must satisfy ``sum(drop_reasons.values()) == n_dropped``.
        kept_mask: boolean mask over the raw frame's index, True where
            the row survived all filters. ``kept_mask.sum() == n_scored``.
    """

    n_raw: int
    n_dropped: int
    n_scored: int
    drop_reasons: dict[str, int]
    kept_mask: pd.Series


def apply_schema_map(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, MappingReport]:
    """Map NYC.gov Rolling Sales columns to the trained model's feature space.

    See ``benchmarks/SCHEMA_MAP.md`` for the transformation contract.

    Step 3 stub — raises ``NotImplementedError`` by design. The real
    implementation lands in Step 4 under the same signature.

    Returns:
        A tuple ``(X, target, report)``:

        - ``X``: feature frame, containing only columns listed in the
          mapping contract and **none** from ``FORBIDDEN_COLUMNS``.
        - ``target``: ``log1p(SALE PRICE)`` as a pandas Series aligned
          to ``X``'s index.
        - ``report``: :class:`MappingReport` with raw/dropped/scored
          counts and per-reason breakdown.
    """
    raise NotImplementedError(
        "apply_schema_map is not yet implemented. This is the Step 3 "
        "TDD red phase. The real mapping lands in Step 4; see "
        "benchmarks/SCHEMA_MAP.md for the contract and "
        "INTERNAL/EXECUTION_CONTRACT.md for the rollout plan."
    )


__all__ = ["MappingReport", "apply_schema_map"]
