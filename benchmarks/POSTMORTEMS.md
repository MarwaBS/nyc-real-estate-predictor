# POSTMORTEMS.md

Dated log of observed behaviour during external benchmark runs. One
entry per run. No tuning commentary. No marketing tone. Each entry
records observed metrics, the expected/actual gap, which invariants
held, which were stressed, and what the run taught about system limits.

---

## 2026-04-20 — Run 1 (SCHEMA_MAP v1, commit 9c7de2b)

### Observed metrics (raw)

| Metric | Value |
|---|---|
| Raw rows downloaded   | 81,567 |
| Rows dropped          | 81,567 (100.0%) |
| Rows scored           | 0 |
| Name-based leakage    | not triggered |
| Semantic leakage      | not triggered |
| Leakage tripwire      | not triggered |
| Model inference       | failed — schema mismatch |
| R² / F1               | unobservable |

Drop-reason breakdown:

| Reason | Count | Share |
|---|---:|---:|
| `non_residential_class`   | 50,799 | 62.3% |
| `sale_price_non_positive` | 27,704 | 34.0% |
| `sale_price_out_of_range` |  2,037 |  2.5% |
| `missing_gross_sqft`      |  1,027 |  1.3% |

Data manifest (SHA-256 truncated):

| Borough | Bytes | SHA-256 |
|---|---:|---|
| Manhattan      | 1,794,149 | `1d680a17b5fb66a2…` |
| Bronx          |   647,165 | `d64bf1bab7cb52fd…` |
| Brooklyn       | 2,278,991 | `9bd378b86b403884…` |
| Queens         | 2,615,326 | `4004d6e198c66623…` |
| Staten Island  |   714,393 | `89bf52091f905870…` |

### Expected vs actual

Expected before the run: some kept rows, model inference producing a
lower R² than the Kaggle holdout (training = 0.815), with the leakage
gate still passing.

Actual: zero kept rows; no predictions produced; leakage gate still
passed. Wider gap than anticipated, but not a system break — see
invariants below.

### Invariants that held

- **Schema lock.** `SCHEMA_MAP.md` SHA-256 matched the `v1` entry in
  `SCHEMA_MAP_VERSIONS.json`. No silent edits.
- **Drop-log consistency.** `n_dropped == sum(drop_reasons.values())`;
  `n_raw == n_scored + n_dropped`.
- **Determinism.** Pipeline executed end-to-end on the published
  five-borough dataset without order-dependent branching.
- **Target independence.** Neither name-based nor three-method
  (Pearson + Spearman + normalised MI) checks fired. Semantic leakage
  check on an empty feature frame trivially passes, so the signal here
  is weak — not a validation of the MI detector under real load.
- **Pipeline reliability.** Five HTTP fetches, five Excel parses, one
  concatenation, one mapping application, one inference attempt —
  completed without unhandled exceptions (the inference failure was
  captured as structured output, not a crash).

### Invariants that were stressed

- **Prediction health checks** (`check_predictions_healthy`) were
  skipped because no predictions were produced. The detector has not
  been exercised against real model output yet.
- **`test_filter_independent_of_target_distribution`** passed on the
  synthetic fixture; real-data behaviour was not tested here because
  no rows survived to be compared.

### What the run taught about system limits

Two structural findings emerged. Both are real, both are outside the
scope of v1, and **neither will be addressed by editing SCHEMA_MAP.md
post-hoc.** Any response lands in a future version with its own
benchmark run.

1. **SCHEMA_MAP v1 residential scope is narrower than NYC.gov's 2024
   residential coverage.** The `BUILDING CLASS CATEGORY` prefix-`R*`
   rule retains condominiums (R4/R5/R6/R7) but excludes one-/two-/
   three-family dwellings (prefixes `01`/`02`/`03`) and coops
   (`09`/`10`). Those prefixes account for 62.3% of the 2024 dataset.
   This is a deliberate v1 scope choice (set before the run), not a
   bug. If a future iteration wants broader coverage, the change
   triggers `SCHEMA_MAP_VERSION` bump + full re-run + invalidated
   prior `results.json` per the versioning rule.

2. **SCHEMA_MAP v1 feature space does not intersect the trained
   model's required feature space sufficiently for inference.** The
   model was trained on the Kaggle 2023 listings schema and requires
   15 features including `BEDS`, `BATH`, `DIST_MANHATTAN_CENTER`,
   `DIST_CENTRAL_PARK`, `DIST_NEAREST_SUBWAY`, and `SUBLOCALITY`.
   NYC.gov Rolling Sales 2024 publishes transaction data only — it
   does not include room counts, coordinates, or neighbourhood
   sub-divisions. This is a **schema-level distribution shift**, not
   a value-level one. Cleaning cannot recover it. A future benchmark
   run that wants to produce R² would need either (a) a different
   model trained on the NYC.gov schema, or (b) an auxiliary join
   against a public geocoded / enriched dataset, both of which are
   new-version work.

### Known gaps this run does NOT address

- Segment-conditioned leakage (see `SCHEMA_MAP.md §6`). Per-stratum
  MI detection still requires stratum sample sizes we do not have.
- Value-level NYC.gov schema drift (e.g., next year the city renames
  a column). Catch path: header-row detection in the downloader, plus
  the schema SHA lock would surface indirectly via pipeline failure.
- pandas `FutureWarning` on the `test_filter_independent_of_target_distribution`
  rescaling helper (int column assigned floats). Captured; not
  addressed per Rule A (no test edits after green).

### What will NOT happen in response to this run

- No edit to `SCHEMA_MAP.md` to broaden residential scope.
- No edit to the drop engine to keep more rows.
- No change to the trained model, the training pipeline, or the
  feature list.
- No retroactive lowering of prediction-health thresholds.

These would invalidate the benchmark by turning it into outcome
shaping. The v1 contract produced the result; the v1 contract stays
locked.

### Decision

Firewall validation: **success**. The benchmark's stated claim —
*"a leakage-guarded, schema-locked pipeline produces reproducible,
auditable behaviour under real-world distribution shift"* — holds on
the real 2024 NYC.gov data. The 🟡 structural weaknesses surfaced by
this run are the benchmark doing its job, not the benchmark failing.
