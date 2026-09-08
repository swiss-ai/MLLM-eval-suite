#!/usr/bin/env bash
# Merge gate for a harness sync. Given a merge commit (fork side = ^1, upstream = ^2),
# report every fork-changed file whose merged content is byte-identical to upstream
# (the fork's change was dropped) or is missing lines the fork added (partial).
# Each DROPPED or PARTIAL entry must be either restored or recorded as superseded
# in the sync record before the submodule pointer moves.
#
# An accept file lists paths whose finding was reviewed and recorded as superseded
# or intentionally restored, one per line: "<path>  <reason>". Accepted findings
# are printed but do not fail the gate.
#
#   scripts/sync_audit.sh <worktree> [merge-commit] [accept-file]
set -u
W=${1:?worktree}; M=${2:-}; A=${3:-}
[ -n "$A" ] && A=$(readlink -f "$A")
cd "$W" || exit 2
[ -n "$M" ] || M=$(git log --merges -1 --format=%H)
accepted() { [ -n "$A" ] && awk -v p="$1" '$1 == p { found = 1 } END { exit !found }' "$A"; }
reason() { awk -v p="$1" '$1 == p { $1 = ""; sub(/^[ \t]+/, ""); print; exit }' "$A"; }
P1=$(git rev-parse "$M^1"); P2=$(git rev-parse "$M^2"); B=$(git merge-base "$P1" "$P2")
echo "worktree=$W merge=$(git rev-parse --short "$M") fork=$(git rev-parse --short "$P1") upstream=$(git rev-parse --short "$P2") base=$(git rev-parse --short "$B")"
n=0; d=0; m=0
mapfile -d '' -t changed < <(git diff -z --name-only "$B" "$P1")
for f in "${changed[@]}"; do
  n=$((n+1))
  if ! git cat-file -e HEAD:"$f" 2>/dev/null; then
    if git cat-file -e "$P2":"$f" 2>/dev/null; then echo "DELETED-IN-MERGE $f"; else echo "DELETED-UPSTREAM-TOO $f"; fi
    continue
  fi
  if git diff --quiet "$P2" HEAD -- "$f" 2>/dev/null && ! git diff --quiet "$B" "$P1" -- "$f"; then
    if accepted "$f"; then echo "ACCEPTED $f  ($(reason "$f"))"; continue; fi
    echo "DROPPED $f  (fork +$(git diff "$B" "$P1" -- "$f" | grep -c '^+[^+]') -$(git diff "$B" "$P1" -- "$f" | grep -c '^-[^-]'))"
    d=$((d+1)); continue
  fi
  if ! git diff --quiet "$P1" HEAD -- "$f"; then
    missing=$(comm -23 <(git diff "$B" "$P1" -- "$f" | grep '^+[^+]' | cut -c2- | sed 's/^[[:space:]]*//' | sort -u) \
                      <(git show HEAD:"$f" | sed 's/^[[:space:]]*//' | sort -u) | grep -cv '^$')
    if [ "$missing" -gt 0 ]; then
      if accepted "$f"; then echo "ACCEPTED $f  ($(reason "$f"))"; else echo "PARTIAL $f  (fork-added lines absent from merge: $missing)"; m=$((m+1)); fi
    fi
  fi
done
echo "fork-changed files: $n, dropped: $d, partial: $m"
[ "$d" -eq 0 ] && [ "$m" -eq 0 ]
