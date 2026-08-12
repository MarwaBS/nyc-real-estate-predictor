"""Break each behaviour in ``tests/mutations.py``; require its gate to go red.

The suite proves the code passes its tests. This proves the tests can fail.

    python scripts/verify_gates.py [--name <mutation>] [--allow-dirty]

Files are restored from an in-memory copy in a ``finally``, never from git, so
an interrupted run cannot discard uncommitted work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SUBPROCESS_TIMEOUT_S  # noqa: E402
from tests.mutations import MUTATIONS, Mutation  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _require_clean_tree(allow_dirty: bool) -> None:
    """A leftover mutation from a crashed run would be read as the baseline."""
    dirty = subprocess.run(
        # -uno: mutations only touch tracked files.
        ["git", "status", "--porcelain", "-uno"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    ).stdout.strip()
    if dirty and allow_dirty:
        print(
            "warning: dirty tree (--allow-dirty). A leftover mutation from a "
            "crashed run would be read as the baseline. CI never passes it.\n",
            file=sys.stderr,
        )
        return
    if dirty:
        sys.exit(
            "verify_gates refuses to run on a dirty tree -- it cannot tell an "
            f"edit in progress from a leftover mutation:\n{dirty}"
        )


def _run_gate(mutation: Mutation) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", mutation.gate, "-x", "-q", "--no-cov"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )


def check(mutation: Mutation) -> tuple[bool, str]:
    """Apply one mutation, run its gate, restore. True if the gate caught it."""
    target = REPO / mutation.path
    # Bytes, not text. Path.write_text translates "\n" to os.linesep, so on
    # Windows restoring an LF-committed file rewrites it with CRLF -- an
    # identical diff that still leaves the tree dirty and trips the guard
    # above on the next run.
    original = target.read_bytes().decode("utf-8")

    if mutation.old not in original:
        return False, (
            f"mutation text not found in {mutation.path} -- the code moved and "
            "this entry is now checking nothing"
        )

    mutated = original.replace(mutation.old, mutation.new, 1)
    try:
        target.write_bytes(mutated.encode("utf-8"))
        result = _run_gate(mutation)
    finally:
        target.write_bytes(original.encode("utf-8"))
        # Drop the module's compiled cache: the gate subprocess compiles a
        # .pyc from the MUTATED source, and when the restore lands within the
        # same mtime second Python considers that .pyc fresh, later runs then
        # import the mutation while the source reads clean.
        cache = target.parent / "__pycache__"
        if cache.exists():
            for pyc in cache.glob(f"{target.stem}.*.pyc"):
                pyc.unlink(missing_ok=True)

    tail = (result.stdout.strip().splitlines() or [""])[-1]

    # Exit codes, not output matching: a real failure prints "E AssertionError",
    # so a search for "error" cannot tell a failing gate from a broken one. Only
    # 1 is detection; 2/3/4/5 mean pytest never judged the change.
    if result.returncode == 1:
        return True, "caught"
    if result.returncode == 0:
        return False, f"{mutation.gate} still passes ({tail})"
    if result.returncode == 5:
        return False, f"{mutation.gate} collected no tests -- the gate is missing"
    return False, f"{mutation.gate} exited {result.returncode} without judging ({tail})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="run a single mutation by name")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run with uncommitted changes (local iteration only; not CI)",
    )
    args = parser.parse_args()

    _require_clean_tree(args.allow_dirty)

    selected = [m for m in MUTATIONS if args.name in (None, m.name)]
    if not selected:
        sys.exit(f"no mutation named {args.name!r}")

    survivors: list[tuple[Mutation, str]] = []
    for i, mutation in enumerate(selected, 1):
        caught, detail = check(mutation)
        mark = "CAUGHT " if caught else "SURVIVED"
        print(
            f"[{i:2}/{len(selected)}] {mark}  {mutation.name}  ({detail})", flush=True
        )
        if not caught:
            survivors.append((mutation, detail))

    print()
    if survivors:
        print(f"{len(survivors)}/{len(selected)} mutations survived:")
        for mutation, detail in survivors:
            print(f"  - {mutation.name}: {detail}")
            print(f"      expected {mutation.gate} to fail")
        return 1
    print(f"All {len(selected)} mutations caught.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
