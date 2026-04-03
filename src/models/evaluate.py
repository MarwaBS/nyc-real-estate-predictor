"""Model evaluation — metrics, confusion matrix, reports, plots."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute all classification metrics and return as dict."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=labels, output_dict=True,
        ),
    }

    logger.info(
        "Classification: accuracy=%.3f, macro_f1=%.3f, kappa=%.3f",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["cohen_kappa"],
    )
    return metrics


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_target: bool = True,
) -> dict[str, float]:
    """Compute all regression metrics. If log_target, also report on original scale."""
    metrics: dict[str, float] = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

    if log_target:
        # Convert back to original scale for interpretable metrics
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
        metrics["rmse_usd"] = float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)))
        metrics["mae_usd"] = float(mean_absolute_error(y_true_orig, y_pred_orig))
        metrics["mape"] = float(mean_absolute_percentage_error(y_true_orig, y_pred_orig))

    logger.info(
        "Regression: R2=%.4f, RMSE=%.4f, MAE=%.4f",
        metrics["r2"],
        metrics["rmse"],
        metrics["mae"],
    )
    return metrics


def evaluate_fairness_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    metric_fn: str = "macro_f1",
) -> dict[str, float]:
    """Evaluate model performance per demographic/geographic group."""
    results: dict[str, float] = {}
    for group_name in groups.unique():
        mask = groups == group_name
        if mask.sum() < 5:
            continue
        if metric_fn == "macro_f1":
            score = float(f1_score(y_true[mask], y_pred[mask], average="macro"))
        else:
            score = float(accuracy_score(y_true[mask], y_pred[mask]))
        results[str(group_name)] = score

    logger.info("Fairness by group: %s", results)
    return results
