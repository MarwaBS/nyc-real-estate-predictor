# ADR-003: Multi-task deep learning with shared trunk

## Status
Reversed (2026-07-19) — see the status update below; the code is deleted.

## Context
Most tabular ML projects use only tree-based models. For a senior portfolio, demonstrating DL competence on tabular data is a differentiator.

## Decision
Build a multi-task PyTorch network with:
- Entity embeddings for categorical features (BOROUGH, TYPE, ZIPCODE)
- Shared trunk (Dense 256 -> 128 -> 64 with BatchNorm + Dropout)
- Classification head (4-class softmax with Focal Loss)
- Regression head (single linear output for LOG_PRICE)
- Combined loss: 0.6 * Focal + 0.4 * MSE

## Rationale
- Multi-task learning provides regularization — classification head prevents regression overfitting
- Entity embeddings learn richer representations than OneHot for high-cardinality features
- Focal Loss addresses class imbalance (Very High is the smallest test class: 161 of ~900 test samples, ~18% — an earlier revision of this line mistakenly recorded the percentage as a sample count)
- Shows architecture design skills beyond "call sklearn.fit()"

## Consequences
- Requires PyTorch dependency (~2GB install)
- Training is GPU-friendly but CPU-viable for this dataset size
- Performance may not exceed XGBoost on tabular data (expected) — the value is in demonstrating the approach
- The shipped DL model is a plain embedding + dense (MLP) multi-task net
  (`MultiTaskDenseNet`), NOT TabNet — it has no attention/sparsemax feature
  selection. TabNet remains a *future* alternative worth evaluating for its
  built-in attention-based interpretability, but it is not what this ADR builds.

## Status update (2026-07-19) — reversed; the code is deleted

`src/dl/` no longer exists. This record is retained because an ADR is a log of
what was decided, not a description of the current tree — deleting it would
erase the fact that the approach was tried.

What was measured before removal, on the same split as the tree models:

| | macro F1 |
|---|---|
| Multi-task dense net | 0.666 |
| Gradient-boosted tree | 0.727 |

The net scored below the tree, its results were quoted in no document, and its
139 statements sat at 0% test coverage. The prediction in "Consequences" above
— *"performance may not exceed XGBoost on tabular data (expected) — the value
is in demonstrating the approach"* — held on the first clause. The second no
longer justified the cost: on 4,526 rows the tree wins, trains in seconds, and
is interpretable via SHAP, while the net added a `torch` pin carrying two
security advisories for a capability nothing shipped.
