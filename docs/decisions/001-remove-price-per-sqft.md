# ADR-001: Remove PRICE_PER_SQFT from all feature sets

## Status
Accepted

## Context
The original notebooks used PRICE_PER_SQFT (= PRICE / PROPERTYSQFT) as a training feature for both regression and classification models. The regression model achieved R2=0.997 — an unusually high score for real estate prediction.

## Decision
Remove PRICE_PER_SQFT from all feature sets. It is derived directly from the target variable (PRICE), which constitutes **data leakage**. The model was effectively given the answer.

## Consequences
- **R2 will drop significantly** (expected: 0.70-0.85 range without leakage). This is the honest performance.
- Feature importance shifts to PROPERTYSQFT, BOROUGH, and geospatial features as the real predictors.
- A `test_no_leakage.py` test suite enforces this constraint in CI — no regression possible.
- All future features must be validated through `assert_no_leakage()` before training.
