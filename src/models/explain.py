"""Model explainability — SHAP values, feature importance, per-prediction explanations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> tuple[Any, Any]:
    """Compute SHAP values for a tree-based model.

    Returns (shap_values, explainer) for downstream plotting.
    """
    import shap

    # Sample for performance if dataset is large
    X_sample = X.sample(n=max_samples, random_state=42) if len(X) > max_samples else X

    # Use TreeExplainer for tree models, KernelExplainer as fallback
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        logger.info("TreeExplainer failed — falling back to KernelExplainer")
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 100))
        shap_values = explainer.shap_values(X_sample)

    logger.info("SHAP values computed for %d samples", len(X_sample))
    return shap_values, explainer


def get_top_features_for_prediction(
    explainer: Any,
    shap_values: np.ndarray,
    feature_names: list[str],
    idx: int = 0,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Get top N contributing features for a single prediction."""
    if isinstance(shap_values, list):
        # Legacy multi-class shape: list of (n_samples, n_features) per class
        values = shap_values[0][idx]
    else:
        values = np.asarray(shap_values)[idx]
        if values.ndim == 2:
            # Modern shap multi-class: (n_features, n_classes) per sample —
            # aggregate magnitude across classes for a single ranking.
            values = values.mean(axis=1)

    # strict=True: one SHAP value per feature name. A mismatch means the
    # explainer's output shape disagrees with the transformed feature list,
    # and truncating would silently mis-pair names to values -- the resulting
    # top-10 table would look plausible and attribute importance to the wrong
    # features.
    importance = list(zip(feature_names, values, strict=True))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            "feature": name,
            "shap_value": round(float(val), 4),
            "direction": "+" if val > 0 else "-",
        }
        for name, val in importance[:top_n]
    ]


def global_feature_importance(
    shap_values: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute mean absolute SHAP values per feature (global importance)."""
    if isinstance(shap_values, list):
        # Legacy multi-class shape: list of (n_samples, n_features) per class
        vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        vals = np.abs(np.asarray(shap_values))
        if vals.ndim == 3:
            # Modern shap multi-class: (n_samples, n_features, n_classes) —
            # average magnitude across classes first.
            vals = vals.mean(axis=2)

    mean_importance = vals.mean(axis=0)
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_importance,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Top 5 features by SHAP: %s", importance_df.head(5).to_dict("records"))
    return importance_df
