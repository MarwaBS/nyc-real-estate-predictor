"""Per-class threshold optimization — improves macro F1 over argmax."""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def optimize_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
    resolution: int = 50,
) -> tuple[dict[str, float], float]:
    """Find optimal per-class probability thresholds to maximize macro F1.

    Instead of argmax(proba), we find the threshold per class that maximizes
    the overall macro F1 score. This is especially useful when class distributions
    are imbalanced (e.g., "Very High" has only 18 test samples).

    Returns:
        (thresholds_dict, best_macro_f1)
    """
    n_classes = proba.shape[1]
    best_thresholds = np.full(n_classes, 0.5)
    best_f1 = _predict_with_thresholds(proba, y_true, best_thresholds)

    # Greedy per-class optimization
    for cls in range(n_classes):
        best_t = best_thresholds[cls]
        for t in np.linspace(0.1, 0.9, resolution):
            candidate = best_thresholds.copy()
            candidate[cls] = t
            f1 = _predict_with_thresholds(proba, y_true, candidate)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[cls] = best_t

    thresholds_dict = {label: round(float(t), 3) for label, t in zip(labels, best_thresholds, strict=False)}
    logger.info("Optimized thresholds: %s -> macro_f1=%.4f", thresholds_dict, best_f1)
    return thresholds_dict, float(best_f1)


def predict_with_thresholds(
    proba: np.ndarray,
    thresholds: dict[str, float],
    labels: list[str],
) -> np.ndarray:
    """Predict classes using optimized per-class thresholds."""
    threshold_array = np.array([thresholds.get(label, 0.5) for label in labels])
    adjusted = proba / threshold_array
    # np.argmax with axis=1 always returns an ndarray; the explicit np.asarray
    # narrows mypy's `Any` return-type inference.
    return np.asarray(np.argmax(adjusted, axis=1))


def _predict_with_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    """Internal: compute macro F1 with given thresholds."""
    adjusted = proba / np.clip(thresholds, 1e-6, None)
    y_pred = np.argmax(adjusted, axis=1)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
