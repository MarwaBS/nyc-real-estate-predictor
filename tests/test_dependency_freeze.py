"""A pin marked ``# frozen:`` must be one Dependabot is configured to leave alone.

Dependabot does not read the comment at the pin. So the two lists drift, and a
bump lands that requirements.txt forbids. Checked both directions: a frozen pin
with no ignore entry fails, and an ignore entry for a package nobody froze fails
too.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPENDABOT = REPO_ROOT / ".github" / "dependabot.yml"
REQUIREMENTS = ("requirements.txt", "requirements-train.txt", "requirements-dev.txt")

#: ``name==version  # frozen: why`` — the marker carries its own reason.
FROZEN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*[^\s#]+\s*#\s*frozen:\s*(\S.*)$")


def _normalise(name: str) -> str:
    """PEP 503 name, so ``category_encoders`` and ``category-encoders`` agree."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _frozen_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for filename in REQUIREMENTS:
        path = REPO_ROOT / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = FROZEN.match(line)
            if match:
                pins[_normalise(match.group(1))] = match.group(2).strip()
    return pins


def _pip_ignore_entries() -> list[dict]:
    config = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))
    pip = next(u for u in config["updates"] if u["package-ecosystem"] == "pip")
    return [e for e in pip.get("ignore", []) if e.get("dependency-name") != "*"]


def _pip_ignored() -> set[str]:
    """Named packages held at their current version, whatever the release."""
    return {
        _normalise(entry["dependency-name"])
        for entry in _pip_ignore_entries()
        if "update-types" not in entry
    }


def _minor_ignored() -> set[str]:
    return {
        _normalise(entry["dependency-name"])
        for entry in _pip_ignore_entries()
        if "version-update:semver-minor" in entry.get("update-types", [])
    }


def _managed_pins() -> list[str]:
    """Every pin Dependabot manages. Its pip entry sets `directory: "/"` with no
    file restriction, and CI installs requirements-dev.txt, which pulls the other
    two into one resolve, so a bump in any of them can break it."""
    names = []
    for filename in REQUIREMENTS:
        path = REPO_ROOT / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            # `pkg[extra]==v` is ordinary; a name-only pattern skips the line and
            # the package leaves the scan without failing anything.
            match = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*[=<>~!]", line)
            if match:
                names.append(_normalise(match.group(1)))
    return names


def _direct_dependencies(package: str) -> set[str]:
    """Installed runtime dependencies of ``package``. Extras are skipped: pip does
    not install them, so they cannot constrain the resolve."""
    try:
        specs = requires(package) or []
    except PackageNotFoundError:
        return set()
    names = set()
    for spec in specs:
        requirement, _, marker = spec.partition(";")
        if "extra" in marker:
            continue
        match = re.match(r"^\s*([A-Za-z0-9_.\-]+)", requirement)
        if match:
            names.add(_normalise(match.group(1)))
    return names


def _reaches_numpy(package: str) -> bool:
    seen: set[str] = set()
    stack = [package]
    while stack:
        for dependency in _direct_dependencies(stack.pop()):
            if dependency == "numpy":
                return True
            if dependency not in seen:
                seen.add(dependency)
                stack.append(dependency)
    return False


def test_the_scan_finds_the_frozen_pins():
    """A renamed marker would leave the two checks below comparing empty sets."""
    pins = _frozen_pins()
    assert len(pins) >= 8, (
        f"only {len(pins)} frozen pins found — the marker or the files moved"
    )
    assert all(reason for reason in pins.values()), "a frozen pin carries no reason"


@pytest.mark.parametrize("package", sorted(_frozen_pins()))
def test_every_frozen_pin_is_ignored_by_dependabot(package: str):
    assert package in _pip_ignored(), (
        f"{package} is pinned '# frozen:' but Dependabot may still bump it — "
        f"add '- dependency-name: \"{package}\"' to the pip ignore list"
    )


@pytest.mark.parametrize("package", sorted(_pip_ignored()))
def test_every_ignored_package_is_a_frozen_pin(package: str):
    assert package in _frozen_pins(), (
        f"Dependabot ignores {package}, but no requirements file marks it '# frozen:'. "
        f"Either mark the pin with its reason, or drop the ignore so updates resume"
    )


def test_the_scan_finds_the_managed_pins():
    """_reaches_numpy reads installed metadata, so a package the scan misses or
    that is absent from the environment answers False and fails nothing."""
    pins = _managed_pins()
    assert len(pins) >= 20, (
        f"only {len(pins)} managed pins found; the files or the pattern moved"
    )
    missing = []
    for package in pins:
        try:
            version(package)
        except PackageNotFoundError:
            missing.append(package)
    assert not missing, (
        f"not installed, so their dependency trees are invisible: {missing}"
    )


def test_numpy_is_still_pinned_below_2():
    """The reason the next check exists. Once the models are refitted under
    numpy 2 it stops applying and both can go."""
    pin = re.search(
        r"^numpy==(\d+)\.", (REPO_ROOT / "requirements.txt").read_text("utf-8"), re.M
    )
    assert pin is not None and pin.group(1) == "1", "numpy is no longer held at 1.x"


@pytest.mark.parametrize("package", sorted(set(_managed_pins())))
def test_a_package_that_reaches_numpy_cannot_take_a_minor_bump(package: str):
    """numpy 2 entered the tree through scipy, then through shap, and each time
    the resolve failed at `pip install`. The set is recomputed here rather than
    listed, so a new dependency that reaches numpy fails until it is handled."""
    if package == "numpy" or not _reaches_numpy(package):
        return
    assert package in _pip_ignored() | _minor_ignored(), (
        f"{package} has numpy in its dependency tree, so a minor bump can require "
        f"numpy>=2 and leave requirements.txt unsolvable. Add it to the pip ignore "
        f"list, with update-types semver-minor if it is not otherwise frozen"
    )
