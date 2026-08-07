# Model Card, NYC Real Estate Price Prediction

Format loosely follows *"Model Cards for Model Reporting"* (Mitchell et al., 2019). Fields chosen for practical use by a hiring reviewer or downstream consumer, not for academic completeness.

---

## Model details

- **Persons or organisations developing the model:** Marwa Ben Salem (solo).
- **Model date:** 2026-07-20 (last trained, `run_date` in `reports/training_metrics.json` is authoritative).
- **Model version:** v1.0.0.
- **Model type:** ONE artefact.
  - **Regressor** - `XGBoost` on `LOG_PRICE`. Predictions converted back via `expm1()`.
  - The 4-class price zone is **derived by bucketing that prediction**, not predicted by a second model. `PRICE_ZONE` is a deterministic function of price, so a classifier was fitting the same features to the same signal - and could disagree with the served price on the same listing. Training scores zones through the same decode serving uses.
- **Additional models compared (not shipped):** Random Forest, LightGBM - compared on val, never scored on test.
- **Training / tuning (shipped artefacts):** `run_training.py`, fixed hyperparameters, best-of-candidates selection on the **val** split, with **test** scored exactly once by the already-selected model. Full record with provenance (commit SHA, sklearn version, seed, splits) in `reports/training_metrics.json`.
  - **Provenance-SHA caveat:** the `commit_sha` recorded in `reports/training_metrics.json` (and in `benchmarks/results.json`) is the PR-branch commit that produced the artefact. This repo squash-merges, which orphans branch commits, so those SHAs are **not ancestors of `main`**. What pins the artefacts instead is `models/MANIFEST.sha256`, which records a SHA-256 for each of the five governed files; `tests/test_artifact_manifest.py` recomputes them and fails on any drift. That ties the shipped bytes to the committed manifest. It does not tie them to the code that produced them.
- **Paper or resource:** architecture, feature engineering, and decisions documented in `README.md` + `docs/decisions/*.md` (ADRs 001-003).
- **Licence:** MIT.
- **Citation / contact:** `marwabensalem30@gmail.com`; include `[MODEL_CARD]` in subject.

## Intended use

- **Primary intended uses:**
  - Portfolio demonstration of end-to-end ML engineering (data cleaning → feature engineering → modelling → tuning → explainability → API → UI → deployment).
  - Educational: illustrate a data-leakage guard (`assert_no_leakage`) and honest R² (0.835) vs. the near-perfect R² a leaked `PRICE_PER_SQFT` produces (0.997, measured on the earlier dataset revision in ADR-001 and not re-measured since).
- **Primary intended users:** hiring managers / reviewers evaluating the author's ML engineering signal; engineers looking at how to structure a tabular-ML project with DL as an ablation.
- **Out-of-scope uses:** the model is **NOT suitable for real pricing decisions**. It is trained on a ~4,500-row public Kaggle snapshot of NYC listings; it will drift against current-market reality, has no per-user auth / SLA, and is not fairness-audited beyond borough-level F1 disparity.

## Factors

- **Relevant groupings:** NYC borough (Staten Island, Bronx, Brooklyn, Queens, Manhattan), F1 varies (0.696 → 0.529).
- **Evaluation factors:** price zone (4 classes, stratified), sublocality (target-encoded with smoothing), property type (one-hot).
- **Factors NOT evaluated:** seller/buyer demographics (not in the dataset); temporal drift across listing date (dataset is a single snapshot); accessibility amenities (not in features).

## Metrics

- **Model performance measures:**
  - Zones: macro F1 = **0.712** on the test split, derived by bucketing the regressor's predictions (naive borough-median baseline: 0.301). There is no classifier.
  - Regression: R² = **0.835** (XGBoost) on test; 0.774 on val (naive borough-median baseline: 0.177). Honest, no leakage (see ADR-001).
  - **That 0.835 is scored against a capped target.** The IQR bounds are fitted on train, but applied to every row, so 72 of 906 test prices are clipped at $4,487,000 before scoring. Against the listed prices the same model scores **0.7883**, and that gap is wider than the 0.028 seed-to-seed spread, so it is not noise. Both figures are in [`reports/cap_factor_study.json`](reports/cap_factor_study.json).
  - Split: 2,896 train / 724 val / 906 test, stratified on pooled price quartiles (a balancing key only; served zone labels come from train-derived cut-points).
- **Decision rule:** the zone is the predicted price bucketed through `PRICE_ZONE_BINS` (`src/models/decode.py`), the same function that labels the training data. There is no classifier, no label encoder and no argmax. Earlier versions shipped per-class tuned thresholds and advertised macro F1 0.724; those were fitted on the test labels and scored on the same labels, so the number was in-sample. Measured out-of-sample over 20 stratified splits, tuning was worth +0.0006 (std 0.0106), noise, and is gone.
- **Variation approaches:** the full protocol (split, train-only fitting, candidate selection) re-run over 20 seeds, test R² **0.814 ± 0.028**, zones macro F1 **0.717 ± 0.020**; XGBoost selected in 16/20 runs. Recorded in `reports/seed_variance.json`; the headline numbers above are the shipped `RANDOM_SEED=42` artefact, which sits inside 1 SD of the seed mean.

## Evaluation data

- **Datasets:** `Resources/NY-House-Dataset.csv` (Kaggle public snapshot, 4,801 rows cleaned to 4,526). The raw CSV is committed, so `python run_training.py` regenerates the cleaned dataset and every artefact below from a fresh clone.
- **Motivation:** illustrative; chosen for small-enough-to-experiment-with size while having enough geospatial and categorical signal to make feature engineering non-trivial.
- **Preprocessing:** `src/data/cleaner.py` handles dedupe, borough/ZIP derivation, borough-aware imputation, overflow-sentinel removal, outlier capping, and normalisation. Target `PRICE_ZONE` is derived at the **equal-frequency quartiles** of the training price distribution ($499,000 / $825,000 / $1,496,000), so the four zones carry ~1,130 listings each. The previous [0, 500k, 1M, 2M] were round numbers with no derivation and split the data 1610/1183/929/805.

  `BOROUGH` and `ZIPCODE` do not exist in the raw export and are derived: borough from `SUBLOCALITY` → `ADMINISTRATIVE_AREA_LEVEL_2` → `LOCALITY` through `BOROUGH_MAP` (78.5% / 0.8% / 47.0% individually, 99.2% chained), ZIP by extracting 5 digits from the misnamed `STATE` field (100%). The 36 rows no source resolves have shifted geocode columns (`LOCALITY` = "United States", `ADMINISTRATIVE_AREA_LEVEL_2` holding a ZIP) and are dropped rather than guessed at.

## Training data

- **Same as evaluation:** a stratified three-way 64/16/20 train/val/test split of the same cleaned dataset (20% test held out first, then 20% of the remainder as val). No separate external corpus.
- **Split strategy:** stratified on pooled price quartiles (`pd.qcut(PRICE, 4)`) as a balancing key only; the served `PRICE_ZONE` labels come from train-derived cut-points, not from the split key.
- **Feature set:** 12 total - 8 numeric + 4 categorical (2 one-hot: `BOROUGH`, `TYPE`; 2 target-encoded: `ZIPCODE`, `SUBLOCALITY`). `PROPERTY_CATEGORY` was removed: training, the API and the dashboard all hardcoded it to "residential", so the encoder only ever saw one level. Full list in README "Feature engineering" section. Features deliberately **exclude** `PRICE_PER_SQFT` (target-derived; causes R² = 0.997 artefact, see ADR-001).

## Quantitative analyses

- **Unitary results:** top SHAP features (mean |SHAP| over the regressor's test predictions; from `reports/training_metrics.json → classification.shap_top10`): `BATH` (0.369), `DIST_MANHATTAN_CENTER` (0.362), `PROPERTYSQFT` (0.144), `DIST_CENTRAL_PARK` (0.116). Full top-10 in README.
- **Intersectional results:** borough-level macro F1 (from the artefact's
  `fairness_by_borough`). There is no `nan` group: rows whose borough cannot
  be derived are dropped during cleaning rather than carried as an unnamed
  category.
  - Manhattan 0.696
  - Brooklyn 0.691
  - The Bronx 0.688
  - Queens 0.679
  - Staten Island 0.529

  Reported without a causal explanation, because none was measured. Not
  currently mitigated (would need per-borough calibration or reweighting).

## Ethical considerations

- **Real-estate price modelling can reinforce existing spatial inequality.** If this model were deployed for pricing or approval decisions, its borough-level F1 disparity would translate into systematically higher uncertainty for Staten Island and Queens, the two lowest scores in the table above, which is the opposite of the "fair across groups" property we usually want. The model IS NOT deployed for any such decision; the fairness analysis here is diagnostic, not a claim of fairness.
- **Dataset age:** the training snapshot predates the 2025 NYC market shifts. Any prediction on current-market data is subject to distribution drift.
- **Sensitive attributes:** the dataset does not contain race, ethnicity, or income; demographic-fairness analysis is not possible. Borough is the only proxy.

## Caveats and recommendations

- **Caveats:**
  - **Uncertainty is an empirical interval, not a distribution.** The served price range is the conformal 10th-90th band of `actual / predicted` measured on the validation split (multipliers 0.677x / 1.457x, recorded in `models/price_interval.json`), and it covers **77.9%** of the test split against an 80% target.

  That is 2.1 points below the target: the binomial standard error of a proportion at p=0.8 on the 906-row test split is 1.33 points, so the gap is 1.6 SE, within sampling noise. It is reported rather than widened to hit 80% exactly, which would make the target a fitted quantity. It is a marginal band: one interval for every property, so it does not widen for unusual inputs the way a quantile-regression or conformal-by-difficulty interval would.
  - **Cross-row statistics are train-fitted:** IQR cap bounds, zone cut-points and the category vocabulary are fitted on the train split only and applied to val/test (`run_training.build_splits`), enforced by `tests/test_train_only_fitting.py` and three pooled-fitting mutations in the CI harness.
  - **The IQR cap factor (3.0) is measured, and it bites harder than "tails".** Fitted on the train split, it clips PRICE at Q3 + 3·IQR = $4,487,000. **205 of 2,896 train rows (7.08%) sit at exactly that value**, every listing from $4,487,000 up is collapsed onto one price, not only the extreme ones. So the model cannot distinguish any property above ~$4.5M, and the top of the price distribution is a spike rather than a tail. Scored on a common evaluation set (listings under $2,991,500, so every variant faces the same target distribution), val MAE is 0.2750 at factor 1.5, 0.2792 at 3.0, 0.2812 at 5.0 and 0.2827 uncapped, capping beats not capping, and tighter is better on the metric. **1.5 measured better than the shipped 3.0** and is not used because it collapses 11.15% of train rows onto one price against 7.08%: the gain is 0.0042 MAE (1.5% relative), the cost is 118 more listings the model cannot tell apart. The lower clip bound never fires (it computes to −$2,492,000 for PRICE, so zero rows). Every figure here is measured with the Random Forest candidate, not the shipped XGBoost, and recorded in [`reports/cap_factor_study.json`](reports/cap_factor_study.json) by `scripts/measure_cap_factor.py`.
  - **One overflow sentinel is dropped, not capped.** The raw export contains a single `PRICE` of 2,147,483,647 (2³¹−1) where the next-highest real listing is $195M. It is removed before capping, because capping would convert it into a plausible-looking listing at the IQR bound instead of eliminating it.
  - **No naive-baseline column next to the headline metrics:** R²=0.835 (Kaggle split, baseline 0.177) and R²(log)=0.250 (external benchmark, baseline -0.016) are each reported beside the per-borough-median naive predictor scored on the same rows, recorded in the artefacts.
- **Recommendations:**
  - Treat predictions as directional, not dollar-accurate.
  - Do not use for decisions that would materially affect a specific person (loan, rent, appraisal).
  - Retrain quarterly if the dataset can be refreshed.
  - Expand fairness analysis to calibration plots + per-group confusion matrices if this model were promoted beyond portfolio use.

## Failure modes observed

- **2026-04-19, silent prediction corruption from a scikit-learn version
  mismatch.** Reproduced locally against a container built from an older
  `requirements.txt`, not on a live user-facing deployment; this model has
  never served real pricing decisions. A serving environment running
  `scikit-learn==1.5.2` loaded
  pipeline artefacts trained under `1.8.0`. sklearn emitted
  `InconsistentVersionWarning`, a warning, not an error, and the pipeline
  continued serving with corrupted internal state (a Manhattan condo
  predicted at **$2**). Nothing crashed; the failure mode was silence.
- **Root cause:** unpinned runtime sklearn + warning-level signal for a
  corruption-level problem.
- **Remediation (both layers now in place):**
  1. *Pin:* training and serving run the identical `scikit-learn==1.8.0`
     (`requirements.txt`).
  2. *Runtime guard:* `src/models/predict.py::_load_model` promotes
     `InconsistentVersionWarning` to a hard `ModelVersionError`, a
     cross-version artefact is **refused**, never served. Regression-tested
     in `tests/test_predict.py::test_version_mismatch_is_refused`.
