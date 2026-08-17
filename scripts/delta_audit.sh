#!/usr/bin/env bash
# Delta drift metric for the eval-harness forks: counts additive (new) vs
# invasive (upstream-file-editing) files relative to the upstream merge-base.
# Run at each sync; the invasive count should trend down. See DELTA.md in each fork.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS=0
for SUB in third_party/lmms-eval third_party/VLMEvalKit; do
  cd "$ROOT/$SUB"
  if ! git rev-parse --verify -q upstream/main >/dev/null; then
    echo "$SUB: no upstream remote — add one, this fork is unmeasured" >&2
    STATUS=1
    continue
  fi
  MB=$(git merge-base HEAD upstream/main)
  ADD=$(git diff --name-only --diff-filter=A "$MB" HEAD | wc -l)
  INV=$(git diff --name-only --diff-filter=MDR "$MB" HEAD | wc -l)
  BEHIND=$(git rev-list --count HEAD..upstream/main)
  echo "$SUB: $ADD additive, $INV invasive, $BEHIND commits behind upstream"
done
exit "$STATUS"
