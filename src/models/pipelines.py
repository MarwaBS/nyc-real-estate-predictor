"""sklearn Pipeline + ColumnTransformer definitions, reproducible preprocessing."""

from __future__ import annotations

import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import NUMERIC_FEATURES, ONEHOT_FEATURES, TARGET_ENCODED_FEATURES

logger = logging.getLogger(__name__)

# Unguarded import: category-encoders is a pinned hard dependency.
from category_encoders import TargetEncoder  # noqa: E402


def build_preprocessor(
    numeric_features: list[str] | None = None,
    onehot_features: list[str] | None = None,
    target_encoded_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build the ColumnTransformer for the ML pipeline.

    Target encoding is applied inside the pipeline so it is fit on the train
    split only, preventing target leakage from encoding.
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    onehot_features = onehot_features or ONEHOT_FEATURES
    target_encoded_features = target_encoded_features or TARGET_ENCODED_FEATURES

    transformers = [
        ("num", StandardScaler(), numeric_features),
        (
            "cat_onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            onehot_features,
        ),
    ]

    if target_encoded_features:
        transformers.append(
            ("cat_target", TargetEncoder(smoothing=10.0), target_encoded_features),
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        "Built preprocessor: %d numeric, %d onehot, %d target-encoded features",
        len(numeric_features),
        len(onehot_features),
        len(target_encoded_features),
    )
    return preprocessor


def build_regression_pipeline(
    model: object,
    preprocessor: ColumnTransformer | None = None,
) -> Pipeline:
    """Wrap preprocessor + model into a single Pipeline."""
    preprocessor = preprocessor or build_preprocessor()
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )
