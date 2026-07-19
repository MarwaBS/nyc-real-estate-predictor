# Model Card — NYC Real Estate Price Prediction

Format loosely follows *"Model Cards for Model Reporting"* (Mitchell et al., 2019). Fields chosen for practical use by a hiring reviewer or downstream consumer, not for academic completeness.

---

## Model details

- **Persons or organisations developing the model:** Marwa Ben Salem (solo).
- **Model date:** 2026-07-18 (last trained — `run_date` in `reports/training_metrics.json` is authoritative).
- **Model version:** v1.0.0.
- **Model types:** two artifacts trained jointly on the same feature set:
  - **Classifier** — `XGBoost`, decoded by argmax. 4-class price zone (Low / Medium / High / Very High).
  - **Regressor** — `LightGBM` on `LOG_PRICE` target. Point-estimate. Predictions converted back via `expm1()`.
- **Additional models compared (not shipped as primary):** LightGBM, Random Forest (regression). The extended training path (`src/models/train_classification.py`, `src/dl/`) also implements Optuna search, CatBoost, a stacking ensemble, SMOTE-ENN, and a multi-task PyTorch dense net (entity embeddings + shared MLP trunk with classification + regression heads — not the TabNet architecture despite the legacy name) — runnable with `requirements-train.txt`, but not the source of the shipped artifacts.
- **Training / tuning (shipped artifacts):** `run_training.py` — fixed hyperparameters, best-of-candidates selection on the **val** split, with **test** scored exactly once by the already-selected model. Full record with provenance (commit SHA, sklearn version, seed, splits) in `reports/training_metrics.json`.
  - **Provenance-SHA caveat:** the `commit_sha` recorded in `reports/training_metrics.json` (and in `benchmarks/results.json`) is the PR-branch commit that produced the artifact. This repo squash-merges, which orphans branch commits, so those SHAs are **not ancestors of `main`** — the artifact's chain of custody is instead enforced continuously: the External Benchmark workflow re-derives the sealed metrics weekly from the committed model on live NYC.gov data, so an artifact/code mismatch surfaces as a red scheduled run rather than relying on SHA ancestry.
- **Paper or resource:** architecture, feature engineering, and decisions documented in `README.md` + `docs/decisions/*.md` (ADRs 001–003).
- **Licence:** MIT.
- **Citation / contact:** `marwabensalem30@gmail.com`; include `[MODEL_CARD]` in subject.

## Intended use

- **Primary intended uses:**
  - Portfolio demonstration of end-to-end ML engineering (data cleaning → feature engineering → modelling → tuning → explainability → API → UI → deployment).
  - Educational: illustrate a data-leakage guard (`assert_no_leakage`) and honest R² (0.835) vs. the near-perfect R² a leaked `PRICE_PER_SQFT` produces (0.997, measured on the earlier dataset revision in ADR-001 and not re-measured since).
- **Primary intended users:** hiring managers / reviewers evaluating the author's ML engineering signal; engineers looking at how to structure a tabular-ML project with DL as an ablation.
- **Out-of-scope uses:** the model is **NOT suitable for real pricing decisions**. It is trained on a ~4,500-row public Kaggle snapshot of NYC listings; it will drift against current-market reality, has no per-user auth / SLA, and is not fairness-audited beyond borough-level F1 disparity.

## Factors

- **Relevant groupings:** NYC borough (Staten Island, Bronx, Brooklyn, Queens, Manhattan) — F1 varies materially (0.777 → 0.591).
- **Evaluation factors:** price zone (4 classes, stratified), sublocality (target-encoded with smoothing), property type (one-hot).
- **Factors NOT evaluated:** seller/buyer demographics (not in the dataset); temporal drift across listing date (dataset is a single snapshot); accessibility amenities (not in features).

## Metrics

- **Model performance measures:**
  - Classification: macro F1 = **0.727** (XGBoost) on the test split; 0.721 on val, which is where it was selected.
  - Regression: R² = **0.835** (XGBoost) on test; 0.826 on val. Honest, no leakage (see ADR-001).
  - Split: 2,882 train / 721 val / 901 test, stratified on price zone.
- **Decision rule:** argmax over the classifier's per-class probabilities, decoded through the label encoder's class order (recorded in the artefact's `classification.metrics.labels`). Earlier versions shipped per-class tuned thresholds and advertised macro F1 0.724; those thresholds were fitted on the test labels and scored on the same labels, so the number was in-sample. Fitted on one half of the test set and scored on the other over 20 stratified splits, tuning is worth +0.0006 (std 0.0106) — noise — and has been removed.
- **Variation approaches:** none repeated across random seeds in the reported numbers. A single seed (`RANDOM_SEED=42`) is used. **Honest limitation:** a Staff-level submission would report mean ± std over N seeds; this project does not.

## Evaluation data

- **Datasets:** `Resources/NY-House-Dataset.csv` (Kaggle public snapshot, 4,801 rows cleaned to 4,526). The raw CSV is committed, so `python run_training.py` regenerates the cleaned dataset and every artefact below from a fresh clone.
- **Motivation:** illustrative; chosen for small-enough-to-experiment-with size while having enough geospatial and categorical signal to make feature engineering non-trivial.
- **Preprocessing:** `src/data/cleaner.py` handles dedupe, borough/ZIP derivation, borough-aware imputation, overflow-sentinel removal, outlier capping, and normalisation. Target `PRICE_ZONE` is derived via fixed cut-points (documented in `src/config.py`).

  `BOROUGH` and `ZIPCODE` do not exist in the raw export and are derived: borough from `SUBLOCALITY` → `ADMINISTRATIVE_AREA_LEVEL_2` → `LOCALITY` through `BOROUGH_MAP` (78.5% / 0.8% / 47.0% individually, 99.2% chained), ZIP by extracting 5 digits from the misnamed `STATE` field (100%). The 36 rows no source resolves have shifted geocode columns (`LOCALITY` = "United States", `ADMINISTRATIVE_AREA_LEVEL_2` holding a ZIP) and are dropped rather than guessed at.

## Training data

- **Same as evaluation:** stratified 80/20 split from the same cleaned dataset. No separate external corpus.
- **Split strategy:** stratified on `PRICE_ZONE` to preserve class balance across train/test.
- **Feature set:** 10 numeric + 5 categorical (3 one-hot: `BOROUGH`, `TYPE`, `PROPERTY_CATEGORY`; 2 target-encoded: `ZIPCODE`, `SUBLOCALITY`). Full list in README "Feature engineering" section. Features deliberately **exclude** `PRICE_PER_SQFT` (target-derived; causes R² = 0.997 artefact — see ADR-001).

## Quantitative analyses

- **Unitary results:** top SHAP features (mean |SHAP|, averaged across the four classes; from `reports/training_metrics.json → classification.shap_top10`): `DIST_MANHATTAN_CENTER` (1.245), `BATH` (0.890), `PROPERTYSQFT` (0.874), `DIST_CENTRAL_PARK` (0.490). Full top-10 in README.
- **Intersectional results:** borough-level macro F1 (from the artifact's
  `fairness_by_borough`; a small missing-borough group is recorded as `nan`):
  - Staten Island 0.777
  - Brooklyn 0.670
  - Manhattan 0.634
  - Bronx 0.612
  - Queens 0.591

  Manhattan and Queens carry the largest class-distribution shift (more
  Very High / more Medium respectively), which depresses their F1 vs.
  Staten Island where the distribution is tighter. Not currently mitigated
  (would need per-borough calibration or reweighting).

## Ethical considerations

- **Real-estate price modelling can reinforce existing spatial inequality.** If this model were deployed for pricing or approval decisions, its borough-level F1 disparity would translate into systematically higher uncertainty for Manhattan and Queens — the opposite of the "fair across groups" property we usually want. The model IS NOT deployed for any such decision; the fairness analysis here is diagnostic, not claim of fairness.
- **Dataset age:** the training snapshot predates the 2025 NYC market shifts. Any prediction on current-market data is subject to distribution drift.
- **Sensitive attributes:** the dataset does not contain race, ethnicity, or income; demographic-fairness analysis is not possible. Borough is the only proxy.

## Caveats and recommendations

- **Caveats:**
  - `DIST_NEAREST_SUBWAY` is a proxy (equal to `DIST_MANHATTAN_CENTER`) **by design**: station-level data is not bundled with the repo, and training + serving must use identical feature semantics, so both sides compute the same proxy (`run_training.py` and `api/main.py`). Its marginal information value is zero; it remains in the schema for forward compatibility.
  - **Uncertainty is an empirical interval, not a distribution.** The served price range is the 10th-90th percentile of `actual / predicted` measured on the validation split (multipliers 0.671x / 1.390x, recorded in `models/price_interval.json`), and it covers **76.3%** of the test split against an 80% target. It is a marginal band: one interval for every property, so it does not widen for unusual inputs the way a quantile-regression or conformal-by-difficulty interval would. It previously was a fixed ±15% derived from nothing, which covered 32% of listings.
  - **Pre-split outlier capping (methodological leakage, small but real):** `src/data/cleaner.py` IQR-caps `PRICE` on the FULL dataset before the train/test split, so the cap bounds carry information about test rows into training preprocessing. The effect is bounded (the cap touches only distribution tails and the same bounds are applied to every row), but a strictly clean pipeline would fit the capper on the training fold only.
  - **The IQR cap factor (3.0) is inherited, not derived.** It clips PRICE at Q3 + 3·IQR = $4,483,000 on this snapshot, which compresses the genuine luxury tail (real listings at $56M-$195M land at the cap) rather than removing it. That is a defensible-but-unproven choice: no experiment in this repo establishes 3.0 over 1.5 or over dropping the tail outright, so it is recorded here as a known weakness rather than presented as tuned.
  - **One overflow sentinel is dropped, not capped.** The raw export contains a single `PRICE` of 2,147,483,647 (2³¹−1) where the next-highest real listing is $195M. It is removed before capping, because capping would convert it into a plausible-looking listing at the IQR bound instead of eliminating it.
  - **No naive-baseline column next to the headline metrics:** R²=0.835 (Kaggle split) and R²(log)=0.250 (external benchmark) are reported without a same-split naive baseline (e.g. borough-median price). The external benchmark's value is defensible on its own terms (sealed contract, weekly re-derivation on live data), but absolute skill vs. a trivial predictor is not quantified here.
- **Recommendations:**
  - Treat predictions as directional, not dollar-accurate.
  - Do not use for decisions that would materially affect a specific person (loan, rent, appraisal).
  - Retrain quarterly if the dataset can be refreshed.
  - Expand fairness analysis to calibration plots + per-group confusion matrices if this model were promoted beyond portfolio use.

## Production incidents (postmortem)

- **2026-04-19 — silent prediction corruption from a scikit-learn version
  mismatch.** A serving environment running `scikit-learn==1.5.2` loaded
  pipeline artifacts trained under `1.8.0`. sklearn emitted
  `InconsistentVersionWarning` — a warning, not an error — and the pipeline
  continued serving with corrupted internal state (a Manhattan condo
  predicted at **$2**). Nothing crashed; the failure mode was silence.
- **Root cause:** unpinned runtime sklearn + warning-level signal for a
  corruption-level problem.
- **Remediation (both layers now in place):**
  1. *Pin:* training and serving run the identical `scikit-learn==1.8.0`
     (`requirements.txt`).
  2. *Runtime guard:* `src/models/predict.py::_load_model` promotes
     `InconsistentVersionWarning` to a hard `ModelVersionError` — a
     cross-version artifact is **refused**, never served. Regression-tested
     in `tests/test_predict.py::test_version_mismatch_is_refused`.
