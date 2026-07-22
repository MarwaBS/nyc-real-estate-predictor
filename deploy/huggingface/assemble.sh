#!/usr/bin/env bash
# Assemble the HF Space runtime snapshot from a checkout of this repo.
#
#   assemble.sh <repo_dir> <space_dir>
#
# Overlays onto <space_dir> (a clone of the Space repo) everything the
# Space needs from <repo_dir>, and ONLY that:
#
#   Dockerfile, README.md, start.sh   <- deploy/huggingface/ (Space-specific)
#   requirements.txt                  <- runtime-only dependency set
#   api/ src/ streamlit_app/          <- runtime code (fresh copy, so files
#                                        deleted from the repo are deleted
#                                        from the Space too)
#   models/ serving artifacts         <- the committed, MANIFEST-pinned
#                                        model files (see below)
#
# MODELS ARE SYNCED, NOT PRESERVED. Leaving <space_dir>/models/ alone
# once let the live demo run current code over a stale model vintage
# (21.4% of a probe grid decoded differently).
# The serving artifacts are committed to this repo and pinned by
# models/MANIFEST.sha256 (enforced by tests/test_artifact_manifest.py),
# so the overlay ships them like any other file and the weekly drift
# guard fails on ANY divergence — code or model. The Space's
# .gitattributes stays untouched (its LFS rules pick the .joblib files
# up on add). The Space-only benchmark leftovers (e.g. old *.pt) are
# removed so the Space carries exactly the manifest set.
#
# Used by BOTH the deploy job (overlay -> commit -> push) and the weekly
# drift guard (overlay -> any diff means the Space is stale -> fail), so
# the definition of "what the Space should contain" exists exactly once.
set -euo pipefail

REPO_DIR="${1:?usage: assemble.sh <repo_dir> <space_dir>}"
SPACE_DIR="${2:?usage: assemble.sh <repo_dir> <space_dir>}"

rm -rf "$SPACE_DIR/api" "$SPACE_DIR/src" "$SPACE_DIR/streamlit_app"

cp "$REPO_DIR/deploy/huggingface/Dockerfile" "$SPACE_DIR/Dockerfile"
cp "$REPO_DIR/deploy/huggingface/README.md" "$SPACE_DIR/README.md"
cp "$REPO_DIR/deploy/huggingface/start.sh" "$SPACE_DIR/start.sh"
cp "$REPO_DIR/requirements.txt" "$SPACE_DIR/requirements.txt"
cp -r "$REPO_DIR/api" "$SPACE_DIR/api"
cp -r "$REPO_DIR/src" "$SPACE_DIR/src"
cp -r "$REPO_DIR/streamlit_app" "$SPACE_DIR/streamlit_app"

# Serving artifacts: exactly the MANIFEST set, nothing else.
rm -rf "$SPACE_DIR/models"
mkdir -p "$SPACE_DIR/models"
while read -r _digest name; do
    [ -n "$name" ] || continue
    cp "$REPO_DIR/models/$name" "$SPACE_DIR/models/$name"
done < "$REPO_DIR/models/MANIFEST.sha256"
cp "$REPO_DIR/models/MANIFEST.sha256" "$SPACE_DIR/models/MANIFEST.sha256"

# Never ship bytecode caches (the original hand-assembled Space did).
find "$SPACE_DIR" -type d -name "__pycache__" -not -path "$SPACE_DIR/.git/*" -exec rm -rf {} +

echo "Snapshot assembled into $SPACE_DIR"
