"""A pin marked ``# frozen:`` must be one Dependabot is configured to leave alone.

Two Dependabot PRs (#83, #88) died at ``pip install`` with ResolutionImpossible,
because numpy was held at 1.x by the major-version ignore while scipy was free to
take a minor bump to a release that requires numpy>=2. The same PRs also proposed
scikit-learn and schema-firewall bumps that a comment forbade — comments the bot
cannot read.

The fix was a per-package ignore list. This test stops that list drifting away
from the pins it protects, in both directions: a new frozen pin with no ignore
entry fails, and an ignore entry for a package nobody froze fails too.
"""

from __future__ import annotations

import re
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


def _pip_ignored() -> set[str]:
    config = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))
    pip = next(u for u in config["updates"] if u["package-ecosystem"] == "pip")
    return {
        _normalise(entry["dependency-name"])
        for entry in pip.get("ignore", [])
        if entry.get("dependency-name") != "*"
    }


def test_the_scan_finds_the_frozen_pins():
    """A renamed marker would leave the two checks below comparing empty sets."""
    pins = _frozen_pins()
    assert len(pins) >= 8, f"only {len(pins)} frozen pins found — the marker or the files moved"
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
