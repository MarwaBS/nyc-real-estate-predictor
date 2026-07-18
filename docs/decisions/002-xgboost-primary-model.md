# ADR-002: XGBoost as primary classification model with Optuna tuning

## Status
Accepted

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
