#!/bin/bash
# eval.sh — text-benchmark eval CLI (EleutherAI lm-eval-harness, one job per task).
#
# Usage:
#   bash eval.sh <model> [--tasks T | --suite text-smoke|text-full|text-requested] [--submit-mode batch|interactive] [--confirm-run-unsafe-code] [--dry-run] [--help] [-- job args...]
#
# <model> forms:
#   /path/to/ckpt                       single path
#   /a,/b,/c                            comma-separated paths
#   @models.txt                         one path per line
#
# --tasks   comma-separated, @file.txt, or direct path to a suite file.
#           Task names are upstream lm-eval names; suite-local task YAMLs under
#           task_suites/lm-eval/custom/ are loaded via --include_path and may
#           be referenced by their YAML task name.
# Post-`--` args are forwarded verbatim to slurm/lm-eval/eval_job.slurm.
#
# --confirm-run-unsafe-code enables HumanEval/MBPP scoring (executes model code).
# AlpacaEval requires exported ALPACA_EVAL_ANNOTATORS_CONFIG; scorer credentials
# and endpoints follow that configuration and are inherited by the job.
# The harness is the pinned Swiss AI submodule on PYTHONPATH — never a
# job-time install. Custom tasks belong in task_suites/lm-eval/custom/, not in
# the harness tree.
set -euo pipefail

REPO_ROOT="${ORCH_REPO_ROOT:?ORCH_REPO_ROOT must point to the suite checkout}"
source "${REPO_ROOT}/slurm/shared/sbatch_overrides.sh"

MODELS_RAW=""
TASKS_RAW=""
SUITE=""
SUBMIT_MODE="batch"
DRY_RUN=0
CONFIRM_RUN_UNSAFE_CODE=0
PASSTHROUGH=()

usage() { sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)       TASKS_RAW="$2"; shift 2 ;;
    --suite)       SUITE="$2"; shift 2 ;;
    --submit-mode) SUBMIT_MODE="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --confirm-run-unsafe-code) CONFIRM_RUN_UNSAFE_CODE=1; shift ;;
    --)            shift; PASSTHROUGH+=("$@"); break ;;
    -h|--help)     usage; exit 0 ;;
    --*)           echo "unknown flag: $1" >&2; usage; exit 1 ;;
    *)             MODELS_RAW="$1"; shift ;;
  esac
done

[[ -n "$MODELS_RAW" ]] || { echo "model path required" >&2; usage; exit 1; }

resolve_list() {
  local raw="$1"
  if [[ "$raw" == @* ]]; then
    grep -vE '^\s*(#|$)' "${raw#@}"
  elif [[ -f "$raw" && "$raw" == *.txt ]]; then
    grep -vE '^\s*(#|$)' "$raw"
  else
    tr ',' '\n' <<< "$raw"
  fi
}

if [[ -n "$SUITE" ]]; then
  SUITE_FILE="${REPO_ROOT}/task_suites/lm-eval/${SUITE//-/_}.txt"
  [[ -f "$SUITE_FILE" ]] || { echo "unknown suite: $SUITE (no $SUITE_FILE)" >&2; exit 1; }
  TASKS="$(grep -vE '^\s*(#|$)' "$SUITE_FILE")"
elif [[ -n "$TASKS_RAW" ]]; then
  TASKS="$(resolve_list "$TASKS_RAW")"
else
  echo "one of --tasks / --suite is required" >&2; exit 1
fi
MODELS="$(resolve_list "$MODELS_RAW")"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$RANDOM}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/results/lm-eval}"
LOG_DIR="${REPO_ROOT}/logs/lm-eval/${RUN_ID}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SLURM_TEMPLATE="${REPO_ROOT}/slurm/lm-eval/eval_job.slurm"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "lm-eval text submit"
echo "  models:      $(echo "$MODELS" | tr '\n' ',' | sed 's/,$//')"
echo "  tasks:       $(echo "$TASKS"  | tr '\n' ',' | sed 's/,$//')"
echo "  dp workers:  ${NUM_PROCESSES}"
echo "  run id:      ${RUN_ID}"
echo "  logs:        ${LOG_DIR}"
echo "========================================"

while IFS= read -r TASK; do
  while IFS= read -r MODEL_PATH; do
    MODEL_LABEL="$(basename "$MODEL_PATH")"
    OUT_DIR="${OUTPUT_BASE}/${MODEL_LABEL}/${RUN_ID}/${TASK}"
    preflight_or_die lm-eval "$MODEL_PATH" \
      --model "$MODEL_PATH" --tasks "$(echo "$TASKS" | tr '\n' ',')" --tokenizer "$MODEL_PATH" \
      --container-image "${SUITE_CONTAINER_IMAGE:-}" --max-model-len "${MAX_MODEL_LEN:-65536}"
    export SUITE_JOB_OUTPUT="${LOG_DIR}/eval_${TASK}_%j.out" SUITE_JOB_ERROR="${LOG_DIR}/eval_${TASK}_%j.err"
    [[ "$SUBMIT_MODE" == "interactive" ]] && export SUITE_JOB_OUTPUT="${LOG_DIR}/eval_${TASK}_interactive.out" SUITE_JOB_ERROR=""
    JOB_ARGS=(
      --model-path "$MODEL_PATH"
      --tasks "$TASK"
      --output-path "$OUT_DIR"
      --num-processes "$NUM_PROCESSES"
      --hf-home "${HF_HOME:-${REPO_ROOT}/cache/hf}"
    )
    [[ "$CONFIRM_RUN_UNSAFE_CODE" -eq 1 ]] && JOB_ARGS+=(--confirm-run-unsafe-code)
    # The registry decides the prompting protocol; LM_EVAL_CHAT_TEMPLATE=0 overrides it for a base model.
    CHAT_TEMPLATE="${LM_EVAL_CHAT_TEMPLATE:-}"
    if [[ -z "$CHAT_TEMPLATE" ]]; then
      if ! CHAT_TEMPLATE="$(cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT" python3 -m suite.tasks --framework lm-eval --chat-template "$TASK")"; then
        echo "chat-template registry lookup failed for ${TASK}; not submitting" >&2
        exit 2
      fi
    fi
    if [[ "$CHAT_TEMPLATE" == "true" || "$CHAT_TEMPLATE" == "1" ]]; then
      JOB_ARGS+=(--apply-chat-template)
    fi
    JOB_ARGS+=("${PASSTHROUGH[@]}")
    echo "--- submit: task=$TASK model=$MODEL_LABEL ---"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf '    dry-run:'; printf ' %q' "$SLURM_TEMPLATE" "${JOB_ARGS[@]}"; printf '\n'
    elif [[ "$SUBMIT_MODE" == "interactive" ]]; then
      bash "$SLURM_TEMPLATE" "${JOB_ARGS[@]}" >"${LOG_DIR}/eval_${TASK}_interactive.out" 2>&1
    else
      sbatch \
        --export=ALL \
        "${SBATCH_OVERRIDES[@]}" \
        --time "$SBATCH_TIME" \
        --job-name "lm-eval-${TASK}" \
        --output "${LOG_DIR}/eval_${TASK}_%j.out" \
        --error  "${LOG_DIR}/eval_${TASK}_%j.err" \
        "$SLURM_TEMPLATE" \
        "${JOB_ARGS[@]}"
    fi
  done <<< "$MODELS"
done <<< "$TASKS"
