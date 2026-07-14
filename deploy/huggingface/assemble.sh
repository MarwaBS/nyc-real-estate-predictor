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
#
# Deliberately untouched: <space_dir>/models/ (the trained .joblib/.json
# artifacts live only on the Space — this repo does not track them) and
# .gitattributes (the Space's LFS config for those artifacts).
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

# Never ship bytecode caches (the original hand-assembled Space did).
find "$SPACE_DIR" -type d -name "__pycache__" -not -path "$SPACE_DIR/.git/*" -exec rm -rf {} +

echo "Snapshot assembled into $SPACE_DIR"
