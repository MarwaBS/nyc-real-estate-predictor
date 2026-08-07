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
        name="serving-cap-skipped-in-predict",
        # The cap lives only in predict_listings, so this block is its
        # single site.
        path="src/models/predict.py",
        old=(
            "    reg = get_regressor()\n"
            "    features = apply_serving_cap(features, "
            "learned_capped_categories(reg))\n"
            "    prices = np.expm1(np.asarray(reg.predict(features), dtype=float))"
        ),
        new=(
            "    reg = get_regressor()\n"
            "    prices = np.expm1(np.asarray(reg.predict(features), dtype=float))"
        ),
        gate="tests/test_predict.py::test_predict_applies_the_serving_cap",
    ),
    Mutation(
        name="near-duplicate-dedup-removed",
        path="src/data/cleaner.py",
        old='    df = df.drop_duplicates(subset=["_lat_round", "_lon_round", "PRICE"], keep="first")',
        new="",
        gate="tests/test_cleaned_dataset_provenance.py",
    ),
    Mutation(
        name="xgboost-version-guard-removed",
        path="src/models/predict.py",
        old='"error", category=UserWarning, message=_XGB_CROSS_VERSION',
        new='"ignore", category=UserWarning, message=_XGB_CROSS_VERSION',
        gate="tests/test_predict.py::test_xgboost_version_mismatch_is_refused",
    ),
    Mutation(
        name="pyproject-description-rots-to-losing-model",
        path="pyproject.toml",
        old=(
            'description = "NYC real estate price prediction with derived '
            'price zones (one XGBoost regressor)"'
        ),
        new=(
            'description = "NYC real estate price prediction with derived '
            'price zones (one Random Forest regressor)"'
        ),
        gate="tests/test_documented_numbers.py",
    ),
    Mutation(
        name="readme-revives-argmax-serving-claim",
        path="README.md",
        old=(
            "**Per-class threshold tuning was removed; serving has no "
            "thresholds to tune.**"
        ),
        new="**Serving decodes with plain argmax.**",
        gate="tests/test_documented_numbers.py",
    ),
    Mutation(
        name="logging-force-reconfigure-removed",
        path="src/utils/logging_config.py",
        old="        force=True,",
        new="",
        gate="tests/test_logging_config.py",
    ),
    Mutation(
        name="drift-baseline-failure-swallowed",
        path="run_training.py",
        old=(
            "    from src.models.drift import save_baseline\n"
            "\n"
            '    save_baseline(X_train, MODELS_DIR / "drift_baseline.json")'
        ),
        new=(
            "    from src.models.drift import save_baseline\n"
            "\n"
            "    try:\n"
            '        save_baseline(X_train, MODELS_DIR / "drift_baseline.json")\n'
            "    except Exception as exc:\n"
            '        logger.warning("drift baseline failed (non-critical): %s", exc)'
        ),
        gate="tests/test_artifact_manifest.py",
    ),
    Mutation(
        name="benchmark-baseline-ungoverned",
        # Dropping it from the producer's governed set must fail a test at
        # HEAD, not only silently on the next retrain.
        path="run_training.py",
        old='    "benchmark_baseline.json",\n',
        new="",
        gate="tests/test_artifact_manifest.py::test_producer_governs_exactly_this_set",
    ),
    Mutation(
        name="tree-state-sampled-but-discarded",
        # The call stays, so anything reading the source still finds it. Only
        # the recorded value changes, and it can now report one outcome.
        path="run_training.py",
        old="    tree_clean_at_start = _git_working_tree_clean()",
        new="    _git_working_tree_clean()\n    tree_clean_at_start = True",
        gate="tests/test_training_provenance.py",
    ),
    Mutation(
        name="baseline-figure-rewritten",
        # Rephrased as well as falsified: a check anchored on the word next to
        # the number stops matching and the false figure ships.
        path="README.md",
        old="naive borough-median baseline: 0.177",
        new="naive borough-median baseline of 0.577",
        gate="tests/test_documented_numbers.py",
    ),
    Mutation(
        name="coverage-scope-narrowed",
        # Drops a module from measurement. The percentage rises and the floor
        # passes, so the gate goes greener by covering less.
        path=".github/workflows/ci.yml",
        old=" --cov=streamlit_app",
        new="",
        gate="tests/test_gate_scope.py",
    ),
    Mutation(
        name="tree-state-resampled-for-the-writer",
        # The early sample stays, so call order still checks out. Only the value
        # handed to the writer changes, and by then the models are on disk.
        path="run_training.py",
        old="    _write_training_metrics(\n        tree_clean_at_start,",
        new="    _write_training_metrics(\n        _git_working_tree_clean(),",
        gate="tests/test_training_provenance.py",
    ),
    Mutation(
        name="baseline-figure-moved-left-of-the-metric",
        # Same falsification as baseline-figure-rewritten, written before the
        # metric name instead of after it.
        path="README.md",
        old="R² = 0.835 on the 20% test split (naive borough-median baseline: 0.177)",
        new="naive borough-median baseline 0.577; R² = 0.835 on the 20% test split",
        gate="tests/test_documented_numbers.py",
    ),
    Mutation(
        name="numpy-reaching-train-package-unfrozen",
        path=".github/dependabot.yml",
        old=(
            '      - dependency-name: "shap"\n'
            '        update-types: ["version-update:semver-minor"]\n'
        ),
        new="",
        gate="tests/test_dependency_freeze.py",
    ),
    Mutation(
        name="coverage-omit-swallows-a-module",
        # The other half of the same knob: source keeps the module, omit removes
        # it, and the percentage rises.
        path="pyproject.toml",
        old='omit = [\n    "tests/*",',
        new='omit = [\n    "streamlit_app/*",\n    "tests/*",',
        gate="tests/test_gate_scope.py",
    ),
    Mutation(
        name="type-check-scope-narrowed",
        path=".github/workflows/ci.yml",
        old=" streamlit_app/ scripts/ run_training.py --ignore-missing-imports",
        new=" run_training.py --ignore-missing-imports",
        gate="tests/test_gate_scope.py",
    ),
    Mutation(
        name="cap-study-figure-rots",
        path="MODEL_CARD.md",
        old="0.2792 at 3.0",
        new="0.2892 at 3.0",
        gate="tests/test_documented_numbers.py",
    ),
    Mutation(
        name="module-left-with-no-production-importer",
        # Drop the one production import and the module survives on its own
        # test file alone, which is how the drift helpers stayed.
        path="run_training.py",
        old="    from src.models.drift import save_baseline\n",
        new="",
        gate="tests/test_gate_scope.py::test_no_shipped_module_has_zero_importers",
    ),
    Mutation(
        name="coverage-exclude-lines-widened",
        # One level above source and omit: it removes statements from the
        # denominator, so measuring less raises the percentage.
        path="pyproject.toml",
        old='exclude_lines = [\n    "pragma: no cover",',
        new='exclude_lines = [\n    "def ",\n    "pragma: no cover",',
        gate="tests/test_gate_scope.py::test_coverage_excludes_only_the_three_justified_lines",
    ),
    Mutation(
        name="doc-link-points-at-nothing",
        path="README.md",
        old="SCHEMA_MAP_VERSIONS.json`](benchmarks/SCHEMA_MAP_VERSIONS.json)",
        new="SCHEMA_MAP_VERSIONS.json`](benchmarks/SCHEMA_VERSIONS.json)",
        gate="tests/test_documented_numbers.py::test_every_relative_link_in_the_docs_resolves",
    ),
    Mutation(
        name="numpy-reaching-package-unfrozen",
        path=".github/dependabot.yml",
        old=(
            '      - dependency-name: "lightgbm"\n'
            '        update-types: ["version-update:semver-minor"]\n'
        ),
        new="",
        gate="tests/test_dependency_freeze.py",
    ),
    Mutation(
        name="coverage-floor-decoyed-by-a-comment",
        # The floor a comment advertises, above the one the step runs.
        path=".github/workflows/ci.yml",
        old=(
            "        run: pytest tests/ -v --tb=short --cov=src --cov=benchmarks "
            "--cov=api --cov=run_training --cov=streamlit_app "
            "--cov-report=term-missing --cov-report=xml --cov-fail-under=85"
        ),
        new=(
            "        # --cov-fail-under=85\n"
            "        run: pytest tests/ -v --tb=short --cov=src --cov=benchmarks "
            "--cov=api --cov=run_training --cov=streamlit_app "
            "--cov-report=term-missing --cov-report=xml --cov-fail-under=10"
        ),
        gate="tests/test_gate_scope.py::test_the_stated_coverage_gate_matches_ci",
    ),
    Mutation(
        name="type-check-narrowed-by-a-flag",
        # Same reach as deleting a path, in a token no path list can see.
        path=".github/workflows/ci.yml",
        old="run_training.py --ignore-missing-imports\n",
        new="run_training.py --ignore-missing-imports --exclude 'src/models/.*'\n",
        gate="tests/test_gate_scope.py::test_the_tool_commands_carry_no_unapproved_argument",
    ),
    Mutation(
        name="coverage-omit-written-as-an-absolute-pattern",
        # coverage matches omit against the absolute path, so this drops the
        # inference path while the repo-relative form matches nothing.
        path="pyproject.toml",
        old='omit = [\n    "tests/*",',
        new='omit = [\n    "*/src/models/*",\n    "tests/*",',
        gate="tests/test_gate_scope.py::test_omit_skips_only_the_files_it_names",
    ),
    Mutation(
        name="served-multiplier-rewritten",
        # A figure at or above 1: the fractional scan once started at `0.`.
        path="MODEL_CARD.md",
        old="0.677x / 1.457x",
        new="0.677x / 1.99x",
        gate="tests/test_documented_numbers.py",
    ),
    Mutation(
        name="false-figure-behind-a-historical-clause",
        # One leading historical sentence used to exempt the rest of the line.
        path="README.md",
        old="R² = 0.835 on the 20% test split",
        new=(
            "An earlier revision used a different split. Calibration slope 0.91. "
            "R² = 0.835 on the 20% test split"
        ),
        gate="tests/test_documented_numbers.py",
    ),
    Mutation(
        name="gate-replay-step-dropped-from-ci",
        # CI then proves the code passes its tests and nothing else.
        path=".github/workflows/ci.yml",
        old="        run: python scripts/verify_gates.py\n",
        new="",
        gate="tests/test_gate_scope.py::test_ci_runs_the_whole_mutation_replay",
    ),
    Mutation(
        name="benchmark-runner-step-dropped",
        # Leaves run_benchmark.py with no runner and no importer. A __main__
        # guard used to excuse that on its own.
        path=".github/workflows/benchmark.yml",
        old="        run: python -m benchmarks.run_benchmark\n",
        new="",
        gate="tests/test_gate_scope.py::test_no_shipped_module_has_zero_importers",
    ),
    Mutation(
        name="coverage-floor-set-twice",
        # pytest honours the last flag, a reader the first.
        path=".github/workflows/ci.yml",
        old="--cov-report=xml --cov-fail-under=85",
        new="--cov-report=xml --cov-fail-under=85 --cov-fail-under=10",
        gate="tests/test_gate_scope.py::test_the_stated_coverage_gate_matches_ci",
    ),
    Mutation(
        name="mypy-scope-excluded-in-config",
        # Narrows the check to 27 files with no command-line token to see.
        path="pyproject.toml",
        old='[tool.mypy]\npython_version = "3.12"',
        new='[tool.mypy]\nexclude = ["src/models/.*"]\npython_version = "3.12"',
        gate="tests/test_gate_scope.py::test_the_mypy_config_narrows_nothing",
    ),
    Mutation(
        name="mypy-errors-silenced-over-shipped-code",
        # Still reports 34 files checked, and finds nothing in any of them.
        path="pyproject.toml",
        old='[[tool.mypy.overrides]]\nmodule = ["tests.*", "notebooks.*"]',
        new=(
            "[[tool.mypy.overrides]]\n"
            'module = ["src.*"]\n'
            "ignore_errors = true\n\n"
            "[[tool.mypy.overrides]]\n"
            'module = ["tests.*", "notebooks.*"]'
        ),
        gate="tests/test_gate_scope.py::test_the_mypy_config_narrows_nothing",
    ),
    Mutation(
        name="coverage-omit-written-as-a-relative-path",
        # coverage makes this absolute; neither plain form matches it.
        path="pyproject.toml",
        old='omit = [\n    "tests/*",',
        new='omit = [\n    "./src/models/*",\n    "tests/*",',
        gate="tests/test_gate_scope.py::test_omit_skips_only_the_files_it_names",
    ),
    Mutation(
        name="benchmark-runner-step-commented-out",
        # The module name survives in the comment, so raw text still found it.
        path=".github/workflows/benchmark.yml",
        old="        run: python -m benchmarks.run_benchmark\n",
        new="        # run: python -m benchmarks.run_benchmark\n",
        gate="tests/test_gate_scope.py::test_no_shipped_module_has_zero_importers",
    ),
    Mutation(
        name="replay-narrowed-and-its-exit-code-swallowed",
        path=".github/workflows/ci.yml",
        old="        run: python scripts/verify_gates.py\n",
        new=(
            "        run: python scripts/verify_gates.py "
            "--name conformal-correction-removed || true\n"
        ),
        gate="tests/test_gate_scope.py::test_ci_runs_the_whole_mutation_replay",
    ),
    Mutation(
        name="false-figure-behind-a-comma-joined-clause",
        # The marker sits one comma-clause left of the figure.
        path="README.md",
        old="R² = 0.835 on the 20% test split",
        new=(
            "R² = 0.835 on the 20% test split, unlike the earlier ensemble, "
            "which reached 0.9997"
        ),
        gate="tests/test_documented_numbers.py",
    ),
    Mutation(
        name="one-place-percentage-rewritten",
        # Below the two-place floor the fractional scan used to start at.
        path="MODEL_CARD.md",
        old="99.2% chained",
        new="11.1% chained",
        gate="tests/test_documented_numbers.py",
    ),
    Mutation(
        name="readme-undercounts-the-ci-jobs",
        path="README.md",
        old="CI runs 5 jobs:",
        new="CI runs 4 jobs:",
        gate="tests/test_gate_scope.py::test_readme_names_every_ci_job",
    ),
]
