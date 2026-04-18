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
- **Additional models compared (not shipped as primary):** LightGBM, Random Forest, Stacking ensemble, Multi-Task PyTorch TabNet. See `src/models/train_classification.py` + `train_regression.py`.
- **Training / tuning:** Optuna Bayesian search, 50 trials per model. Class-imbalance handled via SMOTE-ENN + class-weight reweighting.
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

- **Relevant groupings:** NYC borough (Staten Island, Bronx, Brooklyn, Queens, Manhattan) — F1 varies materially (0.795 → 0.619).
- **Evaluation factors:** price zone (4 classes, stratified), sublocality (target-encoded with smoothing), property type (one-hot).
- **Factors NOT evaluated:** seller/buyer demographics (not in the dataset); temporal drift across listing date (dataset is a single snapshot); accessibility amenities (not in features).

## Metrics

- **Model performance measures:**
  - Classification: macro F1 = **0.724** (XGBoost + threshold tuning) on a stratified 20% hold-out (901 test / 3,603 train).
  - Regression: R² = **0.815** (XGBoost), honest, no leakage (see ADR-001).
- **Decision thresholds:** per-class probability thresholds tuned on validation split — Low=0.165, Medium=0.704, High=0.5, Very High=0.5. Improved macro F1 from 0.704 → 0.724 (+0.020).
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

- **Unitary results:** top SHAP features (mean |SHAP|): `DIST_MANHATTAN_CENTER` (0.212), `PROPERTYSQFT` (0.184), `BATH` (0.166), `SUBLOCALITY` target-encoded (0.148). Full top-10 in README.
- **Intersectional results:** borough-level macro F1:
  - Staten Island 0.795
  - Bronx 0.680
  - Brooklyn 0.664
  - Queens 0.625
  - Manhattan 0.619
  
  Manhattan has the largest class-distribution shift (more Very High), which depresses F1 vs. Staten Island where the distribution is tighter. Not currently mitigated (would need per-borough calibration or reweighting).

## Ethical considerations

- **Real-estate price modelling can reinforce existing spatial inequality.** If this model were deployed for pricing or approval decisions, its borough-level F1 disparity would translate into systematically higher uncertainty for Manhattan and Queens — the opposite of the "fair across groups" property we usually want. The model IS NOT deployed for any such decision; the fairness analysis here is diagnostic, not claim of fairness.
- **Dataset age:** the training snapshot predates the 2025 NYC market shifts. Any prediction on current-market data is subject to distribution drift.
- **Sensitive attributes:** the dataset does not contain race, ethnicity, or income; demographic-fairness analysis is not possible. Borough is the only proxy.

## Caveats and recommendations

- **Caveats:**
  - `DIST_NEAREST_SUBWAY` is currently a proxy (equal to `DIST_MANHATTAN_CENTER`) until NYC Open Data subway-station coordinates are plumbed in. The feature is in the schema but its information value is zero at current state.
  - No uncertainty quantification — the predicted price has a fixed ±15% band, not a calibrated interval.
- **Recommendations:**
  - Treat predictions as directional, not dollar-accurate.
  - Do not use for decisions that would materially affect a specific person (loan, rent, appraisal).
  - Retrain quarterly if the dataset can be refreshed.
  - Expand fairness analysis to calibration plots + per-group confusion matrices if this model were promoted beyond portfolio use.
