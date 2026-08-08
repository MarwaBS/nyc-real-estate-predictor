"""The committed model artefacts must match models/MANIFEST.sha256.

The manifest pins the exact bytes of every governed artefact, so an artefact
swapped without a recorded retrain fails CI. Regenerate it only as part of a
deliberate retrain (one sha256 line per file in models/).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MANIFEST = MODELS_DIR / "MANIFEST.sha256"

# Every committed model artefact whose bytes back a published number: the
# served model + interval, plus the benchmark pair (not served, but README
# quotes their R2(log), so an unpinned vintage there is the same defect). Kept
# independent of run_training.GOVERNED_ARTIFACTS so this catches the producer's
# list rotting rather than merely agreeing with it.
GOVERNED_ARTIFACTS = {
    "benchmark_baseline.json",
    "benchmark_regressor.joblib",
    "price_regressor_best.joblib",
    "price_interval.json",
}


def _manifest_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        entries[name.strip()] = digest
    return entries


def test_manifest_covers_exactly_the_governed_set() -> None:
    assert set(_manifest_entries()) == GOVERNED_ARTIFACTS


def test_producer_governs_exactly_this_set() -> None:
    """run_training's manifest producer must cover this same set, so a retrain
    cannot silently drop an artefact from governance."""
    assert set(run_training.GOVERNED_ARTIFACTS) == GOVERNED_ARTIFACTS


def _artifact_bytes(path: Path) -> bytes:
    """Bytes to hash: text artifacts are LF-normalised first.

    Same rationale as ``benchmarks.invariants.schema_map_sha256``: a
    Windows checkout with ``core.autocrlf`` would otherwise hash different
    bytes than the LF bytes git stores (this exact mismatch failed the
    first CI run of this gate, the manifest had been generated from a
    CRLF working copy). Binary artifacts are hashed raw: normalising a
    pickle would corrupt the comparison.
    """
    data = path.read_bytes()
    if path.suffix == ".json":
        data = data.replace(b"\r\n", b"\n")
    return data


def test_every_serving_artifact_matches_its_manifest_hash() -> None:
    for name, expected in _manifest_entries().items():
        path = MODELS_DIR / name
        assert path.exists(), f"{name} is in the manifest but not in models/"
        actual = hashlib.sha256(_artifact_bytes(path)).hexdigest()
        assert actual == expected, (
            f"{name}: sha256 {actual} != manifest {expected}, the committed "
            f"artifact changed without a manifest update (or vice versa). "
            f"Regenerate the manifest only as part of a deliberate retrain."
        )


def test_manifest_is_byte_identical_to_what_the_producer_writes() -> None:
    """The committed file must be exactly what run_training step 9 emits.
    Same hashes in a different order passed the per-line checks while the
    next retrain would rewrite the file with a pure-reorder diff."""
    lines = [
        f"{hashlib.sha256(_artifact_bytes(MODELS_DIR / name)).hexdigest()}  {name}"
        for name in sorted(GOVERNED_ARTIFACTS)
    ]
    produced = ("\n".join(lines) + "\n").encode("ascii")
    committed = MANIFEST.read_bytes().replace(b"\r\n", b"\n")
    assert committed == produced
