"""Model explainability: SHAP values, feature importance, per-prediction explanations."""

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
        # TreeExplainer raises assorted, version-dependent errors on an
        # unsupported model; any failure degrades to the model-agnostic
        # KernelExplainer rather than losing the explanation.
        logger.info("TreeExplainer failed, falling back to KernelExplainer")
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
    # Single-output regressor: TreeExplainer returns (n_samples, n_features), so
    # one row indexes to a 1-D per-feature vector.
    values = np.asarray(shap_values)[idx]

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
    vals = np.abs(np.asarray(shap_values))
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
