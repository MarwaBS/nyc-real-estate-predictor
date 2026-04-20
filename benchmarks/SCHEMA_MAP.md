# SCHEMA_MAP.md — NYC.gov Rolling Sales 2024 → Model Feature Contract

**Version:** v1
**Status:** Active
**Scope:** External benchmark only. This file governs the transformation
between NYC.gov Rolling Sales 2024 and the feature space used by the
trained NYC Real Estate Predictor. It does **not** govern training data.

---

## 0. Purpose

This file defines a deterministic, non-target-aware transformation contract
between the NYC.gov Rolling Sales 2024 dataset and the model input space.

Any deviation from this mapping invalidates benchmark results for the
version in which the deviation occurred.

This file is treated as:

- CI-verified input contract (`tests/benchmarks/test_schema_firewall.py`)
- audit surface for leakage detection (ADR-002)
- immutable once a benchmark run starts for the corresponding version

---

## 1. Core Principle (anti-leakage invariant)

No transformation in this mapping may:

- reference the target variable (`SALE PRICE`)
- reference any statistic derived from the target (mean, median,
  percentile, z-score, rank, frequency, etc.)
- reference future-derived metadata (post-sale signals)

All transformations must be:

- stateless (row-independent)
- deterministic (same input → same output across runs)
- column-name-indexed (not positional)

Violation of any of the above fails CI.

---

## 2. Column Mapping Contract

| Raw column (NYC.gov)        | Model feature       | Transformation rule                               |
|-----------------------------|---------------------|---------------------------------------------------|
| `SALE PRICE`                | `TARGET` (log_price)| `log1p(SALE PRICE)`; **excluded from X**          |
| `GROSS SQUARE FEET`         | `property_sqft`     | strip commas → int; `0` → drop row (see §4)       |
| `LAND SQUARE FEET`          | `land_sqft`         | strip commas → int                                |
| `BOROUGH`                   | `borough_name`      | integer code (1..5) → canonical name via lookup   |
| `BUILDING CLASS CATEGORY`   | `property_type`     | first 2 chars; only retain rows where prefix `R*` |
| `YEAR BUILT`                | `year_built`        | identity; `0` → drop row                          |
| `ZIP CODE`                  | `zip_code`          | identity (integer)                                |

Borough lookup (fixed):
```
1 → Manhattan
2 → Bronx
3 → Brooklyn
4 → Queens
5 → Staten Island
```

---

## 3. Explicit Feature Exclusion Set (leakage firewall)

The following columns must never enter the model input `X`:

- `SALE PRICE` (target)
- `SALE DATE` (temporal leakage if joined with macro indices)
- `PRICE_PER_SQFT` (ADR-001 banned feature)
- any column algebraically or statistically derived from `SALE PRICE`
- any aggregate statistic involving `SALE PRICE` (mean, median, percentile,
  z-score, rank, frequency encoding)

Enforcement: `benchmarks/invariants.py::check_no_forbidden_columns` plus
`check_target_independence` (Pearson + Spearman + normalised MI).

---

## 4. Row Filtering Rules

Rows are dropped **only** for structural validity. No filter may depend on
the target distribution.

| Condition                                            | Reason                         |
|------------------------------------------------------|--------------------------------|
| `SALE PRICE` ≤ 0                                     | invalid / non-arms-length      |
| `SALE PRICE` < 10,000 or > 100,000,000               | outlier / commercial artifact  |
| `GROSS SQUARE FEET` = 0                              | missing structural feature     |
| `YEAR BUILT` = 0                                     | missing structural feature     |
| `BUILDING CLASS CATEGORY` does not start with `R`    | out of scope (residential only)|

All dropped rows must be counted per reason and logged in
`benchmarks/results.json → drop_reasons`.

No filtering may depend on feature–target correlation or on a target
percentile.

---

## 5. Transformation Guarantees

Every transformation must satisfy:

### 5.1 Determinism
Identical input frame → identical output frame across local, CI, and HF
runners. Enforced by `test_mapping_is_deterministic`.

### 5.2 Statelessness
No transformation may depend on dataset-wide statistics, batch-level
normalisation, or neighbouring rows. Enforced by
`test_mapping_is_stateless_across_subsets`.

### 5.3 Target Independence
No feature construction may touch `SALE PRICE` or any field derived from
it. Enforced by `test_mapping_ignores_target_identity`.

### 5.4 Column-Name Indexing
Transformations reference columns by name, never by position. Enforced by
`test_mapping_uses_column_names_not_positions`.

---

## 6. Anti-Leakage Proof Obligations

A mapping is considered valid only if the full firewall suite
(`tests/benchmarks/test_schema_firewall.py`) is green:

- `test_forbidden_column_rejected` (name-based)
- `test_target_independence_3method` (Pearson + Spearman + normalised MI)
- `test_mapping_is_deterministic`
- `test_mapping_is_stateless_across_subsets`
- `test_mapping_ignores_target_identity`
- `test_mapping_uses_column_names_not_positions`
- `test_drop_reasons_equal_dropped_rows`
- `test_filter_independent_of_target_distribution`
- `test_schema_map_sha_matches_version`
- `test_predictions_healthy` (collapse / NaN / constant)

Known limitation (documented, not addressed in this version):
**segment-conditioned leakage.** The suite catches global and row-level
leakage. Per-stratum MI detection would require more samples per stratum
than this dataset supports (~180 rows/borough). Not implemented; flagged
in the public README alongside benchmark results.

---

## 7. Logging Contract

Every benchmark run must write to `benchmarks/results.json`:

- `run_date` (ISO 8601 UTC)
- `commit_sha`
- `schema_map_version`
- `schema_map_sha256`
- `data_source` (URL)
- `data_sha256`
- `n_raw`, `n_dropped`, `n_scored`
- `drop_reasons` (dict; keys = reasons above; values = counts)
- `feature_columns` (list of columns in X)
- `metrics` (R², macro F1, per-borough F1)
- `health_checks` (pass/fail per check)
- `leakage_tripwire` (threshold + triggered bool)

---

## 8. Known Constraints (explicit honesty layer)

This mapping assumes:

- human-defined feature selection may contain bias not visible to the
  firewall (segment-conditioned leakage; see §6)
- schema drift may occur in future NYC.gov dataset updates
- dropped rows are **not** missing at random; the model is out-of-scope
  for co-op sales and mixed-use buildings
- metrics under distribution shift are expected to degrade; the firewall
  validates pipeline integrity, not model robustness

These are known constraints, not silent failure modes.

---

## 9. Versioning Rule

Any change to this file requires:

1. increment `SCHEMA_MAP_VERSION` in `benchmarks/invariants.py`
2. append the new `{version: sha256}` pair to
   `benchmarks/SCHEMA_MAP_VERSIONS.json`
3. re-run the full benchmark
4. treat all prior `results.json` artefacts as invalid for the new version

Enforced by `test_schema_map_sha_matches_version`. Authority separation
between author and reviewer is enforced by `.github/CODEOWNERS` +
branch protection on `main` (required review, required status checks).

No exceptions.

---

## 10. CI Enforcement Hooks

CI must fail if any of the following is true:

- a column from the exclusion set (§3) appears in `X`
- the target-independence check rejects any feature
- the mapping produces different output on re-run with identical input
- the mapping produces different output on a single-row subset vs the
  full frame for that row
- `drop_reasons` counts do not sum to `n_dropped`
- filtering behaviour changes when the target column is perturbed without
  changing structural fields
- `SCHEMA_MAP.md` SHA256 no longer matches the value pinned for
  `SCHEMA_MAP_VERSION` in `SCHEMA_MAP_VERSIONS.json`
- prediction array is fully NaN, fully Inf, or ≥ 95% identical values
