#!/usr/bin/env bash
# Regenerate the dual-harness eval dashboard for GitHub Pages. Both eval kits
# land on one page: lmms-eval results from RUNS_ROOT plus VLMEvalKit results
# merged in by checkpoint identity through a symlink bridge.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE="$(cd "$HERE/.." && pwd)"
PY="${PY:-python3}"
RUNS_ROOT="${RUNS_ROOT:-/capstor/store/cscs/swissai/infra01/users/xyixuan/apertus-1p5-eval/runs}"
VLMEVAL_OUTPUTS="${VLMEVAL_OUTPUTS:-/capstor/store/cscs/swissai/infra01/vision-datasets/benchmark/VLMEval_Outputs}"
SUITE_VLMEVAL="${SUITE_VLMEVAL:-$SUITE/results/VLMEvalKit}"
BRIDGE="${BRIDGE:-$SUITE/cache/vlmeval_bridge}"
OUT="${OUT:-$SUITE/docs/index.html}"

# one --vlmeval-root unioning the shared VLMEval_Outputs with the suite's own
# VLMEvalKit runs (results/VLMEvalKit/<run-id>/<model>); make_dashboard follows
# the symlinks.
rm -rf "$BRIDGE"; mkdir -p "$BRIDGE"
for d in "$VLMEVAL_OUTPUTS"/*/; do [[ -d "$d" ]] && ln -sfn "$d" "$BRIDGE/$(basename "$d")"; done
[[ -d "$SUITE_VLMEVAL" ]] && for rid in "$SUITE_VLMEVAL"/*/; do
  for md in "$rid"*/; do [[ -d "$md" ]] && ln -sfn "$md" "$BRIDGE/$(basename "$md")"; done
done

mkdir -p "$(dirname "$OUT")"
"$PY" "$HERE/make_dashboard.py" --runs-root "$RUNS_ROOT" --vlmeval-root "$BRIDGE" -o "$OUT"
# internal checkpoint results: keep out of search indexes
"$PY" - "$OUT" <<'PYEOF'
import re,sys
p=sys.argv[1]; h=open(p).read()
if 'name="robots"' not in h:
    open(p,"w").write(re.sub(r'(<head[^>]*>)', r'\1\n<meta name="robots" content="noindex, nofollow">', h, count=1))
PYEOF
echo "dashboard -> $OUT"
