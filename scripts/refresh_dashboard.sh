#!/usr/bin/env bash
# Regenerate the eval dashboard for GitHub Pages. The eval kits land on one
# page: lmms-eval results from RUNS_ROOT/SUITE_LMMS, VLMEvalKit results through
# a symlink bridge, and direct lm-evaluation-harness results from LM_EVAL_ROOT.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE="$(cd "$HERE/.." && pwd)"
PY="${PY:-python3}"
RUNS_ROOT="${RUNS_ROOT:-/capstor/store/cscs/swissai/infra01/users/xyixuan/apertus-1p5-eval/runs}"
SUITE_LMMS="${SUITE_LMMS:-$SUITE/results/lmms-eval}"
LM_EVAL_ROOT="${LM_EVAL_ROOT:-$SUITE/results/lm-evaluation-harness}"
VLMEVAL_OUTPUTS="${VLMEVAL_OUTPUTS:-/capstor/store/cscs/swissai/infra01/vision-datasets/benchmark/VLMEval_Outputs}"
SUITE_VLMEVAL="${SUITE_VLMEVAL:-$SUITE/results/VLMEvalKit}"
BRIDGE="${BRIDGE:-$SUITE/cache/vlmeval_bridge}"
OUT="${OUT:-$SUITE/docs/index.html}"

# one --vlmeval-root unioning the shared VLMEval_Outputs with the suite's own
# VLMEvalKit runs (results/VLMEvalKit/<run-id>/<model>); make_dashboard follows
# the symlinks.
rm -rf "$BRIDGE"; mkdir -p "$BRIDGE"
# Link per-benchmark, not per-model, so a checkpoint evaluated across several
# run-ids (e.g. a full-spatial run + a single-benchmark re-fire) merges instead
# of the last run-id overwriting the others. Later run-ids win per benchmark.
link_model() {
  local md="$1" model; model="$(basename "$md")"; mkdir -p "$BRIDGE/$model"
  for bench in "$md"/*/; do [[ -d "$bench" ]] && ln -sfn "$bench" "$BRIDGE/$model/$(basename "$bench")"; done
}
for d in "$VLMEVAL_OUTPUTS"/*/; do [[ -d "$d" ]] && link_model "$d"; done
[[ -d "$SUITE_VLMEVAL" ]] && for rid in "$SUITE_VLMEVAL"/*/; do
  for md in "$rid"*/; do [[ -d "$md" ]] && link_model "$md"; done
done

# curated checkpoint set (exact canonical keys): SFT 4200 + RL stage2, both
# direct and thinking, plus the sDPO alignment checkpoint.
ONLY=(
  "sft-capfilter-linear-it8816"
  "sft-capfilter-constant-it8816"
  "sft-256k-4200"
  "sft-256k-4200 [thinking-32k]"
  "rl_1p5-8b-stage2_notools_mixthink_1606_480it"
  "rl_1p5-8b-stage2_notools_mixthink_1606_480it [thinking-32k]"
  "sdpo-mix-less-refuse-feedback"
  "sdpo-mix-less-refuse-feedback [thinking-32k]"
)
LABELS=(
  "sft-capfilter-linear-it8816=it8816-linear"
  "sft-capfilter-constant-it8816=it8816-const"
  "sft-256k-4200=SFT-4200"
  "sft-256k-4200 [thinking-32k]=SFT-4200 (think)"
  "rl_1p5-8b-stage2_notools_mixthink_1606_480it=RL-mixthink"
  "rl_1p5-8b-stage2_notools_mixthink_1606_480it [thinking-32k]=RL-mixthink (think)"
  "sdpo-mix-less-refuse-feedback=sDPO"
  "sdpo-mix-less-refuse-feedback [thinking-32k]=sDPO (think)"
)

# Per-task truncation rates for thinking runs (the ⌁ subscripts), recomputed
# only when a run's samples are newer than its cache (the samples are huge).
TRUNC_TOOL="${TRUNC_TOOL:-/iopsstor/scratch/cscs/xyixuan/apertus/lmms-eval/examples/apertus-vllm/scripts/truncation_report.py}"
mkdir -p "$SUITE/cache/truncation"
for md in "$RUNS_ROOT"/*-thinking-32k; do
  [[ -d "$md" ]] || continue
  cache="$SUITE/cache/truncation/$(basename "$md").json"
  # point at the whole model dir (all run-roots) so re-fired tasks are covered,
  # not just the original sweep's run-root.
  if [[ ! -f "$cache" || -n "$(find "$md" -name '*_samples_*.jsonl' -newer "$cache" 2>/dev/null | head -1)" ]]; then
    "$PY" "$TRUNC_TOOL" --run-root "$md" --max-new-tokens 32768 --json > "$cache" 2>/dev/null || true
  fi
done

mkdir -p "$(dirname "$OUT")"
"$PY" "$HERE/make_dashboard.py" --runs-root "$RUNS_ROOT" "$SUITE_LMMS" --vlmeval-root "$BRIDGE" --lm-eval-root "$LM_EVAL_ROOT" --only "${ONLY[@]}" --label "${LABELS[@]}" -o "$OUT"
# internal checkpoint results: keep out of search indexes
"$PY" - "$OUT" <<'PYEOF'
import re,sys
p=sys.argv[1]; h=open(p).read()
if 'name="robots"' not in h:
    open(p,"w").write(re.sub(r'(<head[^>]*>)', r'\1\n<meta name="robots" content="noindex, nofollow">', h, count=1))
PYEOF
echo "dashboard -> $OUT"
