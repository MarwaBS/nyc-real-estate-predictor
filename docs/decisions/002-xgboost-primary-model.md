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
