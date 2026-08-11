#!/usr/bin/env bash
# Delta drift metric for the eval-harness forks: counts additive (new) vs
# invasive (upstream-file-editing) files relative to the upstream merge-base.
# Run at each sync; the invasive count should trend down. See DELTA.md in each fork.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for SUB in third_party/lmms-eval third_party/VLMEvalKit; do
  cd "$ROOT/$SUB"
  git rev-parse --verify -q upstream/main >/dev/null || { echo "$SUB: no upstream remote"; continue; }
  MB=$(git merge-base HEAD upstream/main)
  ADD=0; INV=0
  while read -r f; do
    if git cat-file -e "$MB:$f" 2>/dev/null; then INV=$((INV+1)); else ADD=$((ADD+1)); fi
  done < <(git diff --name-only "$MB" HEAD)
  BEHIND=$(git rev-list --count HEAD..upstream/main)
  echo "$SUB: $ADD additive, $INV invasive, $BEHIND commits behind upstream"
done
