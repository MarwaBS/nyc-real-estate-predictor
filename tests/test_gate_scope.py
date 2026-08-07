"""Every gate's declared scope must match the tree it covers.

Coverage source, omit and exclude_lines; the mypy and bandit path lists; the
--cov flags in three files; the import graph. Each is compared to git ls-files
or to a named set, never to another document agreeing with it.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


#: Tracked Python that coverage does not measure. tests/ is the suite itself;
#: scripts/ runs as subprocesses and by hand, so its line coverage means nothing;
#: notebooks/ is not executed by CI.
_UNMEASURED_TREES = {"tests", "scripts", "notebooks"}


def _tracked_heads() -> set[str]:
    """First path segment of every tracked Python file: `src`, `run_training.py`."""
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return {path.split("/")[0] for path in tracked}


def _shipped_modules() -> set[str]:
    return {head.removesuffix(".py") for head in _tracked_heads()} - _UNMEASURED_TREES


def _ci_step_command(step_name: str) -> str:
    """The ``run:`` scalar of a named CI step, read as YAML.

    Line matching cannot do this. Any line holding the marker satisfies it,
    including a comment, so a decoy above the real command reads as correct
    while the command that executes covers less.
    """
    workflow = yaml.safe_load(_read(".github/workflows/ci.yml"))
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == step_name:
                return str(step["run"])
    raise AssertionError(f"ci.yml has no step named {step_name!r}")


def _make_recipe(target: str) -> str:
    """The tab-indented recipe body of a Makefile target. Comments sit at column
    zero, so they are structurally outside it."""
    body, inside = [], False
    for line in _read("Makefile").splitlines():
        if line.startswith(f"{target}:"):
            inside = True
            continue
        if inside:
            if line.startswith("\t"):
                body.append(line.lstrip("\t"))
            elif line.strip():
                break
    return "\n".join(body)


def _readme_pytest_command() -> str:
    lines = [
        row
        for row in _read("README.md").splitlines()
        if row.startswith("pytest ") and "--cov-fail-under=" in row
    ]
    assert len(lines) == 1, f"README shows {len(lines)} pytest commands, expected 1"
    return lines[0]


def _tool_arguments(command: str, tool: str) -> list[str]:
    return command.split(f"{tool} ", 1)[1].split()


def _is_path(argument: str) -> bool:
    return argument.endswith("/") or argument.endswith(".py")


def _tool_paths(command: str, tool: str) -> set[str]:
    return {a.rstrip("/") for a in _tool_arguments(command, tool) if _is_path(a)}


TOOL_STEPS = {
    "mypy": ("mypy type check", "typecheck"),
    "bandit": ("bandit security scan", "security"),
}

#: Every non-path token each command may carry. A path list cannot see a scope
#: flag: `mypy --exclude 'src/models/.*'` drops the check from 34 files to 27.
_TOOL_ARGUMENTS = {
    "mypy": ["--ignore-missing-imports"],
    "bandit": ["-r", "-n", "3", "-ll"],
}


@pytest.mark.parametrize("tool", sorted(TOOL_STEPS))
def test_the_tool_commands_carry_no_unapproved_argument(tool: str) -> None:
    step, target = TOOL_STEPS[tool]
    for label, command in (
        ("ci.yml", _ci_step_command(step)),
        ("Makefile", _make_recipe(target)),
    ):
        flags = [a for a in _tool_arguments(command, tool) if not _is_path(a)]
        assert flags == _TOOL_ARGUMENTS[tool], (
            f"{label} runs {tool} with {flags}; approved is {_TOOL_ARGUMENTS[tool]}"
        )


@pytest.mark.parametrize("tool", sorted(TOOL_STEPS))
def test_type_and_security_checks_cover_every_tracked_module(tool: str) -> None:
    """The path lists are hardcoded in two files, so a new top-level module is
    invisible to both tools until someone remembers it. Commands are read
    structurally: from the step's ``run:`` scalar and the Makefile recipe body,
    not from any line that mentions the tool."""
    expected = {head.rstrip("/") for head in _tracked_heads()} - {"tests"}
    step, target = TOOL_STEPS[tool]
    for label, command in (
        ("ci.yml", _ci_step_command(step)),
        ("Makefile", _make_recipe(target)),
    ):
        assert _tool_paths(command, tool) == expected, (
            f"{label} runs {tool} over {sorted(_tool_paths(command, tool))}; "
            f"tracked Python outside tests/ is {sorted(expected)}"
        )


#: Every setting [tool.mypy] may carry, and the only module patterns an
#: override may name. `exclude` there narrows the check with no command-line
#: token to see, and `ignore_errors` silences it while still reporting 34 files.
_MYPY_SETTINGS = {
    "python_version",
    "ignore_missing_imports",
    "disallow_untyped_defs",
    "warn_return_any",
}
_MYPY_OVERRIDABLE = {"tests.*", "notebooks.*"}


def test_the_mypy_config_narrows_nothing() -> None:
    config = tomllib.loads(_read("pyproject.toml"))["tool"]["mypy"]
    overrides = config.pop("overrides", [])
    assert set(config) == _MYPY_SETTINGS, f"[tool.mypy] carries {sorted(config)}"
    for override in overrides:
        assert set(override["module"]) <= _MYPY_OVERRIDABLE, (
            f"a mypy override reaches shipped code: {override['module']}"
        )
        assert "ignore_errors" not in override, "a mypy override silences errors"


def _declared_source() -> set[str]:
    config = tomllib.loads(_read("pyproject.toml"))["tool"]["coverage"]["run"]
    return {entry.removesuffix(".py") for entry in config["source"]}


def _cov_flags(command: str) -> set[str]:
    return set(re.findall(r"--cov=([\w./]+)", command))


def test_coverage_measures_every_shipped_module() -> None:
    """Anchored to the tree, not to agreement between documents. Four files can
    be edited to agree on a smaller set, which raises the percentage and passes
    the floor while the module that was just brought into scope leaves it."""
    assert _declared_source() == _shipped_modules(), (
        f"pyproject declares {sorted(_declared_source())}; tracked Python outside "
        f"{sorted(_UNMEASURED_TREES)} is {sorted(_shipped_modules())}"
    )


def test_coverage_excludes_only_the_three_justified_lines() -> None:
    """`exclude_lines` sits above `source` and `omit`, and no gate watched it.
    Adding "def ", "return" and "if " takes 997 measurable statements to 36 and
    the reported percentage rises to 97%. The justified set is closed, so
    widening it is a visible edit here."""
    config = tomllib.loads(_read("pyproject.toml"))["tool"]["coverage"]["report"]
    assert set(config["exclude_lines"]) == {
        "pragma: no cover",
        "if TYPE_CHECKING:",
        "if __name__ == .__main__.:",
    }, f"exclude_lines widened to {sorted(config['exclude_lines'])}"


#: The only shipped files coverage may skip, each with the reason it is skipped
#: rather than measured. Closed set: widening it is an edit here, in review.
OMITTED_BY_DESIGN = {
    "benchmarks/run_benchmark.py": "downloads live NYC.gov data; run end to end by the benchmark workflow",
    "benchmarks/train_benchmark_model.py": "one-off producer of the committed benchmark model",
    "benchmarks/datasets/__init__.py": "package marker for the download adapters",
    "benchmarks/datasets/nyc_rolling_sales_2024.py": "downloads live NYC.gov data",
}


def _omits(pattern: str, path: str) -> bool:
    """Whether coverage's omit ``pattern`` would skip repo-relative ``path``.

    coverage makes a pattern absolute unless it opens with a wildcard, so
    `./src/models/*` matches there while matching neither plain form here."""
    if not pattern.startswith(("*", "?")):
        pattern = (ROOT / pattern).as_posix()
    return fnmatch(path, pattern) or fnmatch((ROOT / path).as_posix(), pattern)


#: The other files coverage would read instead of pyproject.toml.
_COMPETING_COVERAGE_CONFIG = (".coveragerc", "tox.ini", "setup.cfg")


def test_pyproject_is_the_only_coverage_config() -> None:
    """`.coveragerc` replaces the pyproject section wholesale, so every gate
    below would be reading settings that no longer apply."""
    present = [n for n in _COMPETING_COVERAGE_CONFIG if (ROOT / n).exists()]
    assert not present, f"coverage would read these before pyproject.toml: {present}"


def test_omit_skips_only_the_files_it_names() -> None:
    """`omit` is the other half of the same knob and the only one of the four
    that was not pinned to a set. Requiring one measurable file per module left
    it open: omitting `src/models/*` and `src/data/*` drops 352 statements, the
    cleaning pipeline and the inference path, and the headline rises."""
    config = tomllib.loads(_read("pyproject.toml"))["tool"]["coverage"]["run"]
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    shipped = [
        path
        for path in tracked
        if path.split("/")[0].removesuffix(".py") in _declared_source()
    ]
    skipped = {
        path
        for path in shipped
        if any(_omits(pattern, path) for pattern in config["omit"])
    }
    assert skipped == set(OMITTED_BY_DESIGN), (
        f"coverage skips {sorted(skipped)}; the justified set is "
        f"{sorted(OMITTED_BY_DESIGN)}"
    )


def test_every_invocation_measures_the_declared_module_set() -> None:
    invocations = {
        "ci.yml": _ci_step_command("Run tests with coverage"),
        "Makefile": _make_recipe("test"),
        "README.md": _readme_pytest_command(),
    }
    for label, command in invocations.items():
        measured = _cov_flags(command)
        assert measured == _declared_source(), (
            f"{label} measures {sorted(measured)}; pyproject declares "
            f"{sorted(_declared_source())}"
        )


def test_the_stated_coverage_gate_matches_ci() -> None:
    """The floor comes from the step's ``run:`` scalar. Read as text, a
    `# --cov-fail-under=85` comment satisfied this while the command ran 10.
    Exactly one flag, because pytest honours the last and a reader the first."""
    floors = re.findall(
        r"--cov-fail-under=(\d+)", _ci_step_command("Run tests with coverage")
    )
    assert len(floors) == 1, f"ci.yml sets the coverage floor {len(floors)} times"

    readme = _read("README.md")
    stated = set(re.findall(r"(\d+)% coverage gate", readme))
    stated |= set(re.findall(r"CI gate: (\d+)%", readme))
    stated |= set(re.findall(r"--cov-fail-under=(\d+)", readme))
    assert stated <= set(floors), f"CI gates at {floors[0]}%; README states {stated}"


def test_ci_runs_the_whole_mutation_replay() -> None:
    """The exact command. A substring check accepted `--name one-mutation` and
    a trailing `|| true`, which replays 1 of 48 and swallows the exit code."""
    workflow = yaml.safe_load(_read(".github/workflows/ci.yml"))
    runs = [
        step.get("run", "").strip()
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
    ]
    assert "python scripts/verify_gates.py" in runs, (
        "no ci.yml step runs the full replay; steps are " + repr(runs)
    )


def test_readme_names_every_ci_job() -> None:
    """It said 4 jobs and listed 4, omitting `reproducibility` entirely."""
    jobs = list(yaml.safe_load(_read(".github/workflows/ci.yml"))["jobs"])
    sentence = re.search(r"CI runs (\d+) jobs:([^\n]*)", _read("README.md"))
    assert sentence is not None, "README no longer describes the CI jobs"
    assert int(sentence.group(1)) == len(jobs), (
        f"README says {sentence.group(1)} jobs; ci.yml defines {len(jobs)}"
    )
    unnamed = [job for job in jobs if f"`{job}`" not in sentence.group(2)]
    assert not unnamed, f"README does not name these CI jobs: {unnamed}"


#: Files a runner invokes by path rather than importing.
_RUNNER_FILES = (".github/workflows", "Dockerfile", "Dockerfile.streamlit", "Makefile")

#: Scripts run by hand, each with the committed artefact it writes.
ARTEFACT_PRODUCERS = {
    "benchmarks/train_benchmark_model.py": "models/benchmark_regressor.joblib",
    "scripts/measure_cap_factor.py": "reports/cap_factor_study.json",
    "scripts/measure_seed_variance.py": "reports/seed_variance.json",
}


def test_every_producer_still_has_the_artefact_it_writes() -> None:
    """Otherwise the exemption above outlives the reason for it."""
    missing = [a for a in ARTEFACT_PRODUCERS.values() if not (ROOT / a).exists()]
    assert not missing, f"producers exempted for artefacts that are gone: {missing}"


def _runner_text() -> str:
    """What the runners actually execute, comments stripped. Read as raw text,
    a `# see src/models/drift.py` line kept an orphan module looking alive."""
    parts = []
    for name in _RUNNER_FILES:
        path = ROOT / name
        files = sorted(path.glob("*.yml")) if path.is_dir() else [path]
        for file in files:
            if not file.exists():
                continue
            parts += [
                line.split("#", 1)[0]
                for line in file.read_text(encoding="utf-8").splitlines()
            ]
    return "\n".join(parts)


def _imported_modules(source: str) -> set[str]:
    """Dotted module names an ``import`` statement names. Read from the parsed
    tree: grepping the leaf name matched `save_drift_baseline` and kept
    `src.models.drift` looking alive after its only import was deleted."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
    return names


def test_no_shipped_module_has_zero_importers() -> None:
    """A module nothing imports and no runner invokes is weight the reader
    still has to carry. The drift comparison helpers sat that way behind a green
    suite, kept alive by their own test file."""
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    shipped = [f for f in tracked if not f.startswith(("tests/", "notebooks/"))]
    runners = _runner_text()

    orphans = []
    for path in shipped:
        module = path.removesuffix(".py").replace("/", ".")
        if module.endswith("__init__"):
            continue
        # A runner names it either by path (`streamlit run streamlit_app/app.py`)
        # or dotted (`uvicorn api.main:app`).
        if path in runners or module in runners or path in ARTEFACT_PRODUCERS:
            continue
        # Tests are not importers: a module reachable only from its own test
        # file is exactly what this catches.
        imported_by = any(
            module in _imported_modules((ROOT / other).read_text(encoding="utf-8"))
            for other in shipped
            if other != path
        )
        if not imported_by:
            orphans.append(path)
    assert not orphans, f"shipped modules nothing imports or runs: {orphans}"
