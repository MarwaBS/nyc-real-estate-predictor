# ADR-003: Multi-task deep learning with shared trunk

## Status
Accepted

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
- Focal Loss addresses class imbalance (Very High class has only ~18 test samples)
- Shows architecture design skills beyond "call sklearn.fit()"

## Consequences
- Requires PyTorch dependency (~2GB install)
- Training is GPU-friendly but CPU-viable for this dataset size
- Performance may not exceed XGBoost on tabular data (expected) — the value is in demonstrating the approach
- TabNet is offered as an alternative with built-in attention-based interpretability
