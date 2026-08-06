# SCHEMA_MAP.md — NYC.gov Rolling Sales 2024 → Model Feature Contract

**Version:** v4
**Status:** Active
**Scope:** External benchmark only. This file governs the transformation
between NYC.gov Rolling Sales 2024 and the feature space used by the
**lean benchmark regressor** (`benchmarks/train_benchmark_model.py`). It does
**not** govern training data or the flagship model.

> **v3 rationale.** Three changes over v2, all defensive: (1) a blank
> `SALE PRICE` (NaN) is now dropped explicitly and FIRST — NaN compares
> False against every numeric threshold, so v2 let blank-price rows
> survive the price filters and produce a `log1p(NaN)` target; (2)
> `YEAR BUILT` missing (NaN) is dropped alongside `= 0`, with its role
> made explicit (record-quality filter, not a feature); (3) the §4 table
> now lists EVERY rule the drop engine enforces, in priority order — v2's
> table omitted the `invalid_borough` and `missing_zip` rules that the
> reference implementation applied. Feature mapping is unchanged.
>
> **v2 rationale.** The flagship model uses listing features (`BEDS`, `BATH`,
> coordinate-derived distances, `SUBLOCALITY`) that NYC.gov transaction records
> simply do not contain, so it cannot be validated externally. v2 introduces a
> lean model trained only on the three features both datasets genuinely share —
> **borough, property square footage, ZIP** — so the benchmark scores real,
> unseen NYC.gov 2024 sales. Scope is narrowed to **1–3 family dwellings**: for
> condos/coops NYC.gov reports the *building's* gross square footage, not the
> unit's, which is not comparable to the per-unit Kaggle `PROPERTYSQFT`.
> (v1 filtered on a `BUILDING CLASS CATEGORY` `R`-prefix that matched nothing —
> NYC categories start with a digit — so every row was dropped and the model's
> features never matched the data. v2 fixes both.)

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

The model input `X` contains exactly three features (matching
`benchmarks.train_benchmark_model.BENCHMARK_FEATURES`):

| Raw column (NYC.gov)        | Model feature       | Transformation rule                               |
|-----------------------------|---------------------|---------------------------------------------------|
| `SALE PRICE`                | `TARGET` (log_price)| `log1p(SALE PRICE)`; **excluded from X**          |
| `BOROUGH`                   | `borough`           | integer code (1..5) → canonical name via lookup   |
| `GROSS SQUARE FEET`         | `property_sqft`     | strip commas → float; `0` → drop row (see §4)     |
| `ZIP CODE`                  | `zip_code`          | integer → string category                         |

`YEAR BUILT` and `BUILDING CLASS CATEGORY` are read for **row filtering only**
(§4) and never enter `X`. `LAND SQUARE FEET` is unused.

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
the target distribution. The table below is EXHAUSTIVE and in priority
order — each dropped row is assigned the first matching reason, and the
reference implementation (`benchmarks/mapping.py::_run_drop_engine`)
enforces exactly these rules in exactly this order:

| # | Condition                                              | Drop reason (results.json key) |
|---|--------------------------------------------------------|--------------------------------|
| 1 | `SALE PRICE` is blank/NaN                              | `missing_sale_price` — a blank price would otherwise survive every numeric threshold (NaN compares False) and poison the log-target |
| 2 | `SALE PRICE` ≤ 0                                       | `sale_price_non_positive` — invalid / non-arms-length |
| 3 | `SALE PRICE` < 10,000 or > 100,000,000                 | `sale_price_out_of_range` — outlier / commercial artefact |
| 4 | `GROSS SQUARE FEET` is NaN or ≤ 0                      | `missing_gross_sqft` — missing structural feature |
| 5 | `YEAR BUILT` is NaN or = 0                             | `missing_year_built` — record-quality filter: not a model feature, but a sale row with no build year is a low-confidence record |
| 6 | `BOROUGH` not in {1..5}                                | `invalid_borough` — unmappable to a borough name |
| 7 | `ZIP CODE` is NaN or ≤ 0                               | `missing_zip` — missing structural feature |
| 8 | `BUILDING CLASS CATEGORY` is not a 1–3 family dwelling | `not_family_dwelling` — out of scope (condos/coops/commercial — see v2 rationale) |

All dropped rows must be counted per reason and logged in
`benchmarks/results.json → drop_reasons`; the orchestrator hard-fails if
`sum(drop_reasons) != n_dropped` or if any retained row carries a
non-finite log-price target.

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
(`tests/benchmarks/test_schema_firewall.py`) is green (exact test names):

- `test_forbidden_column_rejected` (name-based, parametrised over the set)
- `test_target_independence_catches_renamed_target` +
  `test_target_independence_catches_nonlinear_target_encoding`
  (Pearson + Spearman + normalised MI)
- `test_mapping_is_deterministic`
- `test_mapping_is_stateless_across_subsets`
- `test_mapping_ignores_target_identity`
- `test_mapping_uses_column_names_not_positions`
- `test_drop_reasons_equal_dropped_rows`
- `test_nan_sale_price_is_dropped_not_poisoning_target` +
  `test_nan_year_built_is_dropped` (v3 NaN rules)
- `test_filter_independent_of_target_distribution`
- `test_schema_map_sha_matches_registered_version` +
  `test_schema_map_sha_is_crlf_invariant` +
  `test_verify_schema_map_lock_passes_for_sealed_contract` +
  `test_verify_schema_map_lock_rejects_unsealed_edit`
- `test_collapse_detector_*` (collapse / NaN / constant predictions)

Known limitation (documented, not addressed in this version):
**segment-conditioned leakage.** The suite catches global and row-level
leakage. Per-stratum MI detection is not implemented; the
threshold below which it stops being informative has not been measured, so
no row count is claimed here. Not implemented; flagged
in the public README alongside benchmark results.

---

## 7. Logging Contract

Every benchmark run must write to `benchmarks/results.json` (field names
match `benchmarks/run_benchmark.py` exactly):

- `run_date`, `run_ended` (ISO 8601 UTC)
- `commit_sha`
- `schema_map_version`
- `schema_map_sha256` (LF-normalised; verified against the registry BEFORE
  the run starts — a mismatch aborts the run, it is never recorded as data)
- `data_source` (URL) + `data_manifest` (per-file source URL + SHA-256)
- `n_raw`, `n_dropped`, `n_scored`
- `drop_reasons` (dict; keys = §4 reasons; values = counts; must sum to
  `n_dropped`)
- `feature_columns` (list of columns in X)
- `leakage` (per-detector triggered/message)
- `inference` (status + feature reconciliation)
- `health_checks` (pass/fail per check)
- `performance` (`r2_log_space`, `n_scored` — the regression-only v2+
  contract; there is no classifier in the lean benchmark, hence no F1)
- `leakage_tripwire` (threshold + triggered bool)
- `reproducibility` (what enforces it; no cross-architecture claim)

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

Enforced twice: by `test_schema_map_sha_matches_registered_version` in CI, and at
RUN TIME by `benchmarks.invariants.verify_schema_map_lock`, which the
orchestrator calls before downloading anything — a benchmark cannot
execute, locally or in CI, against a contract whose hash is not sealed
in the registry. Edits are caught mechanically:
`test_schema_map_sha_matches_registered_version` fails on any change that
does not bump the version, and `verify_schema_map_lock` aborts a run before
the download. This is a single-maintainer repository; there is no independent
human reviewer, and `.github/CODEOWNERS` routes review requests rather than
gating merges.

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
