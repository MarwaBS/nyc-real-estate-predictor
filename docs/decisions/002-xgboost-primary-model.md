# ADR-002: XGBoost as primary classification model with Optuna tuning

## Status
Superseded — see the dated status updates below. Final state (2026-07-19):
Random Forest selected for regression; the classifier ruling is void (no
classifier ships).

## Context
We evaluated 5 model families: Random Forest, XGBoost, LightGBM, CatBoost, and a stacking ensemble. Previous experiments used RandomizedSearchCV with limited parameter ranges.

## Decision
Use XGBoost as the primary classifier, tuned with Optuna (Bayesian optimization, 50 trials). Also train LightGBM and CatBoost as alternatives, and offer a stacking ensemble as the highest-accuracy option.

## Rationale
- Optuna finds better hyperparameters in 50 trials than GridSearchCV does in 250 (Bayesian vs. exhaustive)
- XGBoost handles class imbalance well via scale_pos_weight
- CatBoost handles categoricals natively but is slower to tune
- Stacking (XGB + LGBM + CatBoost -> LR) provides a 1-3% F1 lift at the cost of 3x training time

## Consequences
- Training time increases from ~2 minutes to ~15 minutes (Optuna trials)
- Model artifacts are larger (stacking = 3 models + meta-learner)
- All three boosting models are saved for comparison in MLflow

## Status update (2026-07-14)

The SHIPPED artifacts are NOT produced by this Optuna path. The serving
pipeline (`run_training.py`, per `MODEL_CARD.md`) uses fixed
hyperparameters with best-of-candidates selection — the Optuna/stacking
work described above lives in the extended training path
(`src/models/train_classification.py` + `requirements-train.txt`) and is
runnable, but its output is not what the API and dashboard serve. This
record originally read as if Optuna tuning was the shipped decision;
it is the explored decision, superseded for shipping by the simpler
fixed-hyperparameter pipeline (reproducibility over the last ~0.01 F1).

## Status update (2026-07-18)

"Best-of-candidates selection" above meant selection **on the test split**:
candidates were compared on test and the winner's test score was published as
a hold-out result. Selection now happens on a dedicated validation split and
test is scored once, by the already-chosen model.

The classifier decision is unchanged — XGBoost still wins, now on val
(macro F1 0.713 vs LightGBM 0.683). The **regressor** decision changed:
LightGBM (val R2 0.790) now wins over Random Forest (0.788) and XGBoost
(0.782), where XGBoost had won under test-split selection. Random Forest was
also bounded with `min_samples_leaf=10` — unbounded, its 500 fully-grown
trees produced a 129 MB artifact that exceeds GitHub's file limit and could
not be committed to the registry at all.

## Status update (2026-07-19) — supersedes the 2026-07-18 regressor ruling

The 2026-07-18 entry above is **no longer true of the shipped artifacts** and
is retained only as the record of what was decided then. It ruled LightGBM the
regressor on val R2 0.790, ahead of Random Forest 0.788 and XGBoost 0.782.

Those numbers were measured on a cleaned dataset that no code in this repo
could regenerate: `output/cleaned_house_dataset.csv` had no producer, carried
a 2,147,483,647 overflow sentinel, and held `BOROUGH`/`ZIPCODE` columns the
cleaner never created. Once the cleaner derived those columns for real and
training ran from the committed raw CSV, the candidate ordering changed.

Measured on the current data (`reports/training_metrics.json`,
`regression.candidates_val`):

| Regressor | val R2 |
|---|---|
| **Random Forest (selected)** | **0.7837** |
| LightGBM | 0.7799 |
| XGBoost | 0.7727 |

So the regressor ruling reverses: **Random Forest** is selected, and it scores
R2 0.8117 on test — read once, after selection.

The classifier ruling is void rather than revised. There is no classifier: the
price zone is the regressor's prediction bucketed through `PRICE_ZONE_BINS`
(`src/models/decode.py`), so the val macro F1 comparison this ADR recorded no
longer describes anything that runs. Zones are scored on test only, at macro
F1 0.699.

The `min_samples_leaf=10` bound on Random Forest stands — it is what keeps the
artifact committable regardless of which model wins.
