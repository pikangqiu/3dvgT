#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-VggT}"
VISIBILITY="${VISIBILITY:-private}"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI 'gh' is required. Install it with: brew install gh" >&2
  exit 1
fi

gh auth status

if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin "$(git branch --show-current)"
else
  gh repo create "${REPO_NAME}" "--${VISIBILITY}" --source=. --remote=origin --push
fi

