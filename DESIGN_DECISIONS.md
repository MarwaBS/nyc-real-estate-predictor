# Design Decisions

Why the system is built the way it is — the reasoning, the dead ends, and the
measurement behind every non-obvious constant. Each entry points to the code,
artefact, or ADR that holds the evidence, so nothing here is an unbacked claim.

The numbers below are the shipped `RANDOM_SEED=42` artefact; the 20-seed spread
that frames them lives in [`reports/seed_variance.json`](reports/seed_variance.json).

## Architecture: one regressor, zones derived

The system predicts price with a single regressor over `LOG_PRICE`, and the
4-class price **zone is that prediction bucketed** through `PRICE_ZONE_BINS`
(`src/models/decode.py`) — not a second model.

- **Why not a dedicated classifier?** When one was measured it beat the bucketed
  zones by ~1.3 SE on a 906-row split — not an established difference (figures in
  the CHANGELOG). It was cut because two models can disagree on one response (a
  "High" zone beside a price that buckets to "Medium") with nothing to catch it.
  Bucketing makes the published macro-F1 describe exactly what the service
  returns. See `run_training.run_protocol`.
- **Why beat a naive baseline, not just report R²?** Every headline is reported
  next to a per-borough-median predictor scored on the same rows: test R² 0.835
  vs 0.177 in-distribution, and 0.250 vs −0.016 on the external benchmark. A
  number without its floor is not evidence of skill (`_borough_median_baseline`
  in `run_training.py`; `benchmarks/results.json`).

## Dead ends (what was tried and removed)

- **Multi-task deep-learning trunk** — built, then reversed: it lost to the tree
  on the metric, at 139 statements of 0%-covered complexity. Figures and the
  reversal in [ADR-003](docs/decisions/003-multi-task-deep-learning.md).
- **Per-class threshold tuning** — previously published a macro-F1 that was
  fitted on the test labels and scored on the same labels. Measured
  out-of-sample (fit on half the test set, scored on the other, 20 splits) the
  gain was noise (figures in the CHANGELOG). Deleted.
- **Pooled fitting of cross-row statistics** — cap bounds, zone cut-points and
  the category vocabulary were once fit on the full dataset before the split, a
  train/test leak. Now every one fits on the train split only
  (`run_training.build_splits`), enforced by `tests/test_train_only_fitting.py`.
- **A hardcoded ±15% price band** — derived from nothing; when measured it
  contained the true price about a third of the time. Replaced by the conformal
  interval below.
- **Features that carried no signal, all since deleted:**
  - `LOG_SQFT` — deleted; trees are invariant to monotone transforms (val R²
    identical to four decimals with and without).
  - `DIST_NEAREST_SUBWAY` — deleted; it was a duplicate of
    `DIST_MANHATTAN_CENTER`, no station data bundled.
  - `PROPERTY_CATEGORY` — deleted; hardcoded to one level everywhere, so the
    encoder saw a single category.
- **An unreproducible cleaned CSV** — the committed dataset had no producer and
  held an impossible 2³¹−1 price. Root-caused and rebuilt so `run_training.py`
  regenerates it. See [`benchmarks/POSTMORTEMS.md`](benchmarks/POSTMORTEMS.md)
  and the CHANGELOG.

## Hyperparameters — derived, not tuned by feel

- **IQR cap factor = 3.0.** A measured trade-off, re-derivable with
  [`scripts/measure_cap_factor.py`](scripts/measure_cap_factor.py): on held-out
  val over a common support, factor 1.5 scores marginally better on MAE but
  collapses ~11% of train listings onto one price against ~7% at 3.0 — a model
  that cannot distinguish anything above ~$3M fails the requirement for one
  listing in nine. Bounds fit on the train split only.
- **Zone cut-points.** Equal-frequency quartiles of the **train** capped prices
  ($499k / $825k / $1,496k), so a quarter of listings falls in each zone; the
  earlier round-number bins split the data 1610/1183/929/805. Serving reads the
  same values from config, gated by `tests/test_config_artefact_agreement.py`.
- **Conformal interval target = 0.80** with the finite-sample correction
  (`ceil((n+1)(1−α))/n`-th order statistic, not the plain quantile). Calibrated
  on val, coverage reported once on test (77.9% — within 1.6 SE of target). The
  uncorrected quantile under-covers by construction (`calibrate_price_interval`).
- **`min_samples_leaf=10`** on the Random Forest candidate — a leaf averages ≥10
  comparable sales, and it also bounds the artefact under GitHub's 100 MB limit
  (unbounded, the 500 trees produce a 129 MB file).
- **Model selection is seed-sensitive** and reported as such: across 20 seeds the
  three candidates win 16 / 3 / 1 (XGBoost / Random Forest / LightGBM) — they are
  statistically indistinguishable at this data size, so the shipped selection is
  a per-seed val comparison, not a claim of a better family
  ([`reports/seed_variance.json`](reports/seed_variance.json)).

## Where the rest of the reasoning lives

| Topic | Source |
|---|---|
| Architecture decisions (leakage, model, DL) | [`docs/decisions/`](docs/decisions/) (ADR-001…003) |
| Every measured change, in order | [`CHANGELOG.md`](CHANGELOG.md) |
| External-benchmark run failures + limits | [`benchmarks/POSTMORTEMS.md`](benchmarks/POSTMORTEMS.md) |
| Model card (intended use, fairness, caveats) | [`MODEL_CARD.md`](MODEL_CARD.md) |
| The gates that keep these claims true | [`tests/`](tests/), `scripts/verify_gates.py` |
