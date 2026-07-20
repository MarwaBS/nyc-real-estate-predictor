"""Behavioural changes that must make a named gate go red.

``verify_gates.py`` applies these one at a time in CI. A constant that shapes
training, serving or a published number belongs here; if no test goes red for
it, write the gate rather than an entry pointing at a loosely related test.
"""

from __future__ import annotations

from typing import NamedTuple


class Mutation(NamedTuple):
    """One behavioural change and the gate that must catch it."""

    name: str
    path: str
    old: str
    new: str
    # pytest target, scoped to one file so each mutation runs in seconds.
    gate: str


MUTATIONS: list[Mutation] = [
    Mutation(
        name="conformal-correction-removed",
        path="run_training.py",
        old="corrected_hi = min(math.ceil((n_cal + 1) * hi_q) / n_cal, 1.0)",
        new="corrected_hi = hi_q",
        gate="tests/test_price_interval.py",
    ),
    Mutation(
        name="zone-tie-break-flipped",
        path="src/models/decode.py",
        old="bisect.bisect_left(_CUTS, float(price))",
        new="bisect.bisect_right(_CUTS, float(price))",
        gate="tests/test_decode.py",
    ),
    Mutation(
        name="interval-target-halved",
        path="run_training.py",
        old="PRICE_INTERVAL_TARGET = 0.80",
        new="PRICE_INTERVAL_TARGET = 0.50",
        gate="tests/test_price_interval.py",
    ),
    Mutation(
        name="outlier-cap-factor-loosened",
        path="src/data/cleaner.py",
        old="    factor: float = 3.0,",
        new="    factor: float = 5.0,",
        gate="tests/test_data_cleaner.py",
    ),
    # SQFT_BINS and the fallback encoder's max_categories are absent because
    # both were deleted: dead constants need removing, not gating.
    Mutation(
        name="training-category-cap-slashed",
        path="src/data/features.py",
        old="    max_categories: int = 50,",
        new="    max_categories: int = 5,",
        gate="tests/test_features.py",
    ),
    Mutation(
        name="drift-threshold-disabled",
        path="src/models/drift.py",
        old="    threshold: float = 0.15,\n) -> dict[str, dict[str, float]]:",
        new="    threshold: float = 0.9,\n) -> dict[str, dict[str, float]]:",
        gate="tests/test_drift.py",
    ),
    Mutation(
        name="price-zone-bins-shifted",
        path="src/config.py",
        old="PRICE_ZONE_BINS: list[float] = [0, 499_000, 825_000, 1_496_000,",
        new="PRICE_ZONE_BINS: list[float] = [0, 400_000, 800_000, 1_200_000,",
        gate="tests/test_config_artefact_agreement.py",
    ),
    Mutation(
        name="feature-dropped",
        path="src/config.py",
        old='    "TOTAL_ROOMS",',
        new="",
        gate="tests/test_config_artefact_agreement.py",
    ),
    Mutation(
        name="random-seed-changed",
        path="src/config.py",
        old="RANDOM_SEED: int = 42",
        new="RANDOM_SEED: int = 7",
        gate="tests/test_config_artefact_agreement.py",
    ),
    Mutation(
        name="test-split-resized",
        path="src/config.py",
        old="TEST_SIZE: float = 0.2",
        new="TEST_SIZE: float = 0.3",
        gate="tests/test_config_artefact_agreement.py",
    ),
    Mutation(
        name="val-split-resized",
        path="src/config.py",
        old="VAL_SIZE: float = 0.2",
        new="VAL_SIZE: float = 0.1",
        gate="tests/test_config_artefact_agreement.py",
    ),
    Mutation(
        name="cap-bounds-fitted-on-pooled",
        path="run_training.py",
        old="bounds = fit_cap_bounds(df.loc[idx_train])",
        new="bounds = fit_cap_bounds(df)",
        gate="tests/test_train_only_fitting.py",
    ),
    Mutation(
        name="zone-bins-fitted-on-pooled",
        path="run_training.py",
        old='train_prices = df.loc[idx_train, "PRICE"]',
        new='train_prices = df["PRICE"]',
        gate="tests/test_train_only_fitting.py",
    ),
    Mutation(
        name="category-vocab-fitted-on-pooled",
        path="run_training.py",
        old="df.loc[idx_train], columns=",
        new="df, columns=",
        gate="tests/test_train_only_fitting.py",
    ),
    Mutation(
        name="borough-floor-check-neutered",
        path="run_training.py",
        # The comparison, not the def name: renaming breaks the import and
        # pytest exits 2 without ever judging the change.
        old="        if actual <= baseline:",
        new="        if actual < -1.0:",
        gate="tests/test_borough_floor.py",
    ),
    Mutation(
        name="borough-floor-scored-with-accuracy",
        path="run_training.py",
        # Single-line patterns: several working-tree files are CRLF, so a
        # pattern containing "\n" never matches them.
        old='group["true"], group["pred"], average="macro", zero_division=0',
        new='group["true"], group["pred"], average="micro", zero_division=0',
        gate="tests/test_borough_floor.py",
    ),
    Mutation(
        name="earth-radius-shifted",
        path="src/utils/geo.py",
        old="earth_radius = 6_371.0  # km",
        new="earth_radius = 6_400.0  # km",
        gate="tests/test_geo.py",
    ),
    Mutation(
        name="type-suffix-strip-removed",
        path="src/data/cleaner.py",
        old='.str.replace(r"\\s+for\\s+sale$", "", regex=True)',
        new='.str.replace(r"\\s+never\\s+matches$", "", regex=True)',
        gate="tests/test_data_cleaner.py",
    ),
    Mutation(
        name="beds-bound-removed",
        path="api/schemas.py",
        old='    beds: int = Field(ge=0, le=20, description="Number of bedrooms")',
        new='    beds: int = Field(ge=0, le=2000, description="Number of bedrooms")',
        gate="tests/test_api.py",
    ),
    Mutation(
        name="borough-validation-removed",
        path="api/schemas.py",
        old="        if normalized not in VALID_BOROUGHS:",
        new="        if False:",
        gate="tests/test_api.py",
    ),
    Mutation(
        name="served-band-from-unrounded-price",
        path="src/models/predict.py",
        old='"price_range": price_range(rounded)',
        new='"price_range": price_range(price)',
        gate="tests/test_predict.py::test_served_band_reproduces_from_the_rounded_price",
    ),
    Mutation(
        name="near-duplicate-dedup-removed",
        path="src/data/cleaner.py",
        old='    df = df.drop_duplicates(subset=["_lat_round", "_lon_round", "PRICE"], keep="first")',
        new="",
        gate="tests/test_cleaned_dataset_provenance.py",
    ),
]
