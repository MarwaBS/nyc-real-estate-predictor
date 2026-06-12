# Model Card — NYC Real Estate Price Prediction

Format loosely follows *"Model Cards for Model Reporting"* (Mitchell et al., 2019). Fields chosen for practical use by a hiring reviewer or downstream consumer, not for academic completeness.

---

## Model details

- **Persons or organisations developing the model:** Marwa Ben Salem (solo).
- **Model date:** 2026-04-xx (last trained; see `CHANGELOG.md`).
- **Model version:** v1.0.0.
- **Model types:** two artifacts trained jointly on the same feature set:
  - **Classifier** — `XGBoost` with per-class threshold tuning. 4-class price zone (Low / Medium / High / Very High).
  - **Regressor** — `XGBoost` on `LOG_PRICE` target. Point-estimate. Predictions converted back via `expm1()`.
- **Additional models compared (not shipped as primary):** LightGBM, Random Forest (regression). The extended training path (`src/models/train_classification.py`, `src/dl/`) also implements Optuna search, CatBoost, a stacking ensemble, SMOTE-ENN, and a multi-task PyTorch TabNet — runnable with `requirements-train.txt`, but not the source of the shipped artifacts.
- **Training / tuning (shipped artifacts):** `run_training.py` — fixed hyperparameters, best-of-candidates selection on the held-out split, per-class threshold tuning post-hoc. Full record with provenance (commit SHA, sklearn version, seed, splits) in `reports/training_metrics.json`.
- **Paper or resource:** architecture, feature engineering, and decisions documented in `README.md` + `docs/decisions/*.md` (ADRs 001–003).
- **Licence:** MIT.
- **Citation / contact:** `marwabensalem30@gmail.com`; include `[MODEL_CARD]` in subject.

## Intended use

- **Primary intended uses:**
  - Portfolio demonstration of end-to-end ML engineering (data cleaning → feature engineering → modelling → tuning → explainability → API → UI → deployment).
  - Educational: illustrate a data-leakage guard (`assert_no_leakage`) and honest R² vs. inflated R² (0.815 vs 0.997 when `PRICE_PER_SQFT` is leaked — ADR-001).
- **Primary intended users:** hiring managers / reviewers evaluating the author's ML engineering signal; engineers looking at how to structure a tabular-ML project with DL as an ablation.
- **Out-of-scope uses:** the model is **NOT suitable for real pricing decisions**. It is trained on a ~4,500-row public Kaggle snapshot of NYC listings; it will drift against current-market reality, has no per-user auth / SLA, and is not fairness-audited beyond borough-level F1 disparity.

## Factors

- **Relevant groupings:** NYC borough (Staten Island, Bronx, Brooklyn, Queens, Manhattan) — F1 varies materially (0.778 → 0.613).
- **Evaluation factors:** price zone (4 classes, stratified), sublocality (target-encoded with smoothing), property type (one-hot).
- **Factors NOT evaluated:** seller/buyer demographics (not in the dataset); temporal drift across listing date (dataset is a single snapshot); accessibility amenities (not in features).

## Metrics

- **Model performance measures:**
  - Classification: macro F1 = **0.724** (XGBoost + threshold tuning) on a stratified 20% hold-out (901 test / 3,603 train).
  - Regression: R² = **0.815** (XGBoost), honest, no leakage (see ADR-001).
- **Decision thresholds:** per-class probability thresholds tuned on validation split — Low=0.361, Medium=0.9, High=0.492, Very High=0.5. Improved macro F1 from 0.711 → 0.724 (+0.014).
- **Variation approaches:** none repeated across random seeds in the reported numbers. A single seed (`RANDOM_SEED=42`) is used. **Honest limitation:** a Staff-level submission would report mean ± std over N seeds; this project does not.

## Evaluation data

- **Datasets:** `Resources/NY-House-Dataset.csv` (Kaggle public snapshot, ~4,800 rows cleaned to 4,504).
- **Motivation:** illustrative; chosen for small-enough-to-experiment-with size while having enough geospatial and categorical signal to make feature engineering non-trivial.
- **Preprocessing:** `src/data/cleaner.py` handles dedupe, borough-aware imputation, outlier capping, and normalisation. Target `PRICE_ZONE` is derived via fixed cut-points (documented in `src/config.py`).

## Training data

- **Same as evaluation:** stratified 80/20 split from the same cleaned dataset. No separate external corpus.
- **Split strategy:** stratified on `PRICE_ZONE` to preserve class balance across train/test.
- **Feature set:** 10 numeric + 3 categorical. Full list in README "Feature engineering" section. Features deliberately **exclude** `PRICE_PER_SQFT` (target-derived; causes R² = 0.997 artefact — see ADR-001).

## Quantitative analyses

- **Unitary results:** top SHAP features (mean |SHAP|, averaged across the four classes; from `reports/training_metrics.json → classification.shap_top10`): `DIST_MANHATTAN_CENTER` (1.149), `PROPERTYSQFT` (0.858), `BATH` (0.786), `DIST_CENTRAL_PARK` (0.404). Full top-10 in README.
- **Intersectional results:** borough-level macro F1 (from the artifact's
  `fairness_by_borough`; a small missing-borough group is recorded as `nan`):
  - Staten Island 0.778
  - Bronx 0.681
  - Brooklyn 0.677
  - Manhattan 0.627
  - Queens 0.613

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
  - No uncertainty quantification — the predicted price has a fixed ±15% band, not a calibrated interval.
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
