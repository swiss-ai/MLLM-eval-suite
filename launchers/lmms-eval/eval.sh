#!/usr/bin/env bash
# eval.sh — Apertus VLM eval CLI (per-task SQLite cache, single user entry point).
#
# Usage:
#   bash eval.sh <model> [--tasks T | --suite full|smoke|audio-full|audio-smoke|audio-llm-eval] [--mode fill|readonly] [--submit-mode batch|interactive] [--help]
#
# <model> forms:
#   /path/to/ckpt                       single path
#   /a,/b,/c                            comma-separated paths
#   @file.txt                           one path per line (comments # and blanks OK)
#
# --tasks   comma-separated, @file.txt, or direct path to a suite file.
# --suite   named curation: full (default), smoke, audio-full, audio-smoke, audio-llm-eval.
# --mode    fill|readonly. Both modes use the shared cache directly with preload
#           on and writes enabled.
# --submit-mode  batch|interactive. Batch submits one sbatch per task/model pair.
#                Interactive runs the job script directly with bash so it uses
#                the current shell's node allocation.
# --extra-framework-config  Extra lmms-eval argv token. Repeat for each token.
# --dry-run  Print submissions without executing them.
#
# Examples:
#   bash eval.sh /path/to/ckpt                          # full suite, fill mode
#   bash eval.sh /path/to/ckpt --suite smoke            # 3-task sanity
#   bash eval.sh /path/to/ckpt --tasks mmmu_val,chartqa # specific tasks
#   bash eval.sh /a,/b,/c                               # multiple models
#   bash eval.sh @models.txt --tasks @custom.txt
#   bash eval.sh /path/to/ckpt --mode fill              # explicit default
#   bash eval.sh /path/to/ckpt --submit-mode interactive --suite audio-smoke
#
# Cache layout (per-task SQLite, bounded growth per file):
#   $CACHE_BASE/{task}/image_tokens/apertus_image_token_cache.sqlite3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${ORCH_REPO_ROOT:-}" ]]; then
  echo "ORCH_REPO_ROOT must be set by the orchestration launcher." >&2
  exit 2
fi
REPO_ROOT="${ORCH_REPO_ROOT}"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-${REPO_ROOT}/slurm/lmms-eval/eval_job.slurm}"
LMMS_CACHE_ROOT="${LMMS_CACHE_ROOT:-${REPO_ROOT}/cache/lmms-eval}"
CACHE_BASE="${CACHE_BASE:-${LMMS_CACHE_ROOT}/image_token_cache}"
LOG_BASE="${LOG_DIR:-${REPO_ROOT}/logs/lmms-eval}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/results/lmms-eval}"
HF_HOME_PATH="${HF_HOME:-${REPO_ROOT}/cache/hf}"
NLTK_DATA_PATH="${NLTK_DATA:-${REPO_ROOT}/cache/nltk_data}"
XDG_CACHE_HOME_PATH="${XDG_CACHE_HOME:-${REPO_ROOT}/cache/xdg}"
VLLM_CACHE_ROOT_PATH="${VLLM_CACHE_ROOT:-${REPO_ROOT}/cache/vllm}"
LMMS_EVAL_MODELS_CACHE_PATH="${LMMS_EVAL_MODELS_CACHE:-${REPO_ROOT}/cache/models}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
OUTPUT_BASE="${OUTPUT_PATH}"
OUTPUT_PATH="${OUTPUT_BASE}/${RUN_ID}"
LOG_DIR="${LOG_BASE}/${RUN_ID}"

# ------------------------------------------------------------------
# Canonical task suites.
# The launch wrapper reads these files so suites are maintained in one place.
# ------------------------------------------------------------------
SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/task_suites/lmms-eval}"
SUITE_SMOKE="${SUITE_SMOKE:-${SUITE_DIR}/visual_smoke.txt}"
SUITE_FULL="${SUITE_FULL:-${SUITE_DIR}/visual_full.txt}"
SUITE_AUDIO_SMOKE="${SUITE_AUDIO_SMOKE:-${SUITE_DIR}/audio_smoke.txt}"
SUITE_AUDIO_FULL="${SUITE_AUDIO_FULL:-${SUITE_DIR}/audio_full.txt}"
SUITE_AUDIO_LLM_EVAL="${SUITE_AUDIO_LLM_EVAL:-${SUITE_DIR}/audio_llm_eval.txt}"

# ------------------------------------------------------------------
# CLI parsing
# ------------------------------------------------------------------
usage() {
  awk 'NR>1 && /^#/{sub(/^# ?/, ""); print; next} NR>1 && !/^#/{exit}' "${BASH_SOURCE[0]}"
}

if [[ $# -lt 1 ]]; then usage; exit 1; fi

MODELS_RAW=""
TASKS_RAW=""
SUITE=""
MODE="fill"
SUBMIT_MODE="batch"
ENABLE_THINKING=""
GEN_KWARGS_OVERRIDE=""
LABEL_SUFFIX=""
EXTRA_FRAMEWORK_ARGS=()
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)  usage; exit 0 ;;
    --tasks)    TASKS_RAW="$2"; shift 2 ;;
    --suite)    SUITE="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --submit-mode) SUBMIT_MODE="$2"; shift 2 ;;
    --enable-thinking) ENABLE_THINKING=1; shift ;;
    --gen-kwargs) GEN_KWARGS_OVERRIDE="$2"; shift 2 ;;
    --label-suffix) LABEL_SUFFIX="$2"; shift 2 ;;
    --extra-framework-config|--extra-framework-arg) EXTRA_FRAMEWORK_ARGS+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --*)        echo "unknown flag: $1" >&2; usage; exit 1 ;;
    *)
      # First positional = model(s)
      if [[ -z "$MODELS_RAW" ]]; then MODELS_RAW="$1"; shift
      else echo "unexpected positional: $1" >&2; usage; exit 1
      fi
      ;;
  esac
done

if [[ -z "$MODELS_RAW" ]]; then echo "missing <model> argument" >&2; usage; exit 1; fi

case "$MODE" in fill|readonly) ;; *) echo "--mode must be fill|readonly (got: $MODE)" >&2; exit 1 ;; esac
case "$SUBMIT_MODE" in batch|interactive) ;; *) echo "--submit-mode must be batch|interactive (got: $SUBMIT_MODE)" >&2; exit 1 ;; esac

# ------------------------------------------------------------------
# Resolve models: inline path, comma-separated, or @file
# ------------------------------------------------------------------
resolve_list() {
  # $1 = raw value ("/path", "/a,/b,/c", "@file.txt", or direct suite path)
  local raw="$1"
  if [[ "$raw" == @* ]]; then
    local f="${raw#@}"
    [[ -f "$f" ]] || { echo "file not found: $f" >&2; exit 1; }
    # Strip comments and blanks; keep one per line
    sed -E 's/[[:space:]]*#.*$//' "$f" | tr ',' '\n' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  elif [[ -f "$raw" ]]; then
    sed -E 's/[[:space:]]*#.*$//' "$raw" | tr ',' '\n' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  else
    echo "$raw" | tr ',' '\n' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  fi
}

MODELS=$(resolve_list "$MODELS_RAW")
[[ -z "$MODELS" ]] && { echo "no models resolved from: $MODELS_RAW" >&2; exit 1; }

# ------------------------------------------------------------------
# Resolve tasks: --tasks takes priority, else --suite, else default full
# ------------------------------------------------------------------
if [[ -n "$TASKS_RAW" ]]; then
  TASKS=$(resolve_list "$TASKS_RAW")
else
  case "${SUITE:-full}" in
    full) TASKS=$(resolve_list "$SUITE_FULL") ;;
    smoke) TASKS=$(resolve_list "$SUITE_SMOKE") ;;
    audio-full) TASKS=$(resolve_list "$SUITE_AUDIO_FULL") ;;
    audio-smoke) TASKS=$(resolve_list "$SUITE_AUDIO_SMOKE") ;;
    audio-llm-eval) TASKS=$(resolve_list "$SUITE_AUDIO_LLM_EVAL") ;;
    *) echo "--suite must be full|smoke|audio-full|audio-smoke|audio-llm-eval (got: $SUITE)" >&2; exit 1 ;;
  esac
fi
[[ -z "$TASKS" ]] && { echo "no tasks resolved" >&2; exit 1; }

# ------------------------------------------------------------------
# Container-environment fixes for sbatch from inside Pyxis container.
#
# This script is invoked from inside a container. Two things bite sbatch here:
#   1. The container inherits SLURM_SPANK_* env vars from the parent job which
#      conflict with the new submission's --environment flag. Strip them.
#   2. libjson-c.so.5 (needed by pyxis) is missing from default library path
#      inside the container; we keep a copy in the team wheelhouse.
# ------------------------------------------------------------------
unset $(env | awk -F= '/^SLURM_SPANK/{print $1}') 2>/dev/null || true
export LD_LIBRARY_PATH="/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}"

# ------------------------------------------------------------------
# Auto-load WANDB_API_KEY from ~/.netrc and HF_TOKEN from HF cache.
# /users isn't mounted inside the job container so token files there are
# unreachable from the inner slurm job; we re-export here.
# ------------------------------------------------------------------
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
  WANDB_API_KEY=$(awk '/api.wandb.ai/{flag=1;next} flag && /password/{print $2; exit}' "$HOME/.netrc")
  export WANDB_API_KEY
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN=$(<"$HOME/.cache/huggingface/token")
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# ------------------------------------------------------------------
# Eval defaults (override via env if needed)
# ------------------------------------------------------------------
DEFAULT_TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
TOKENIZER_PATH="${TOKENIZER_PATH:-${DEFAULT_TOKENIZER_PATH}}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
if [[ -z "${CHAT_TEMPLATE}" && "${TOKENIZER_PATH}" == "${DEFAULT_TOKENIZER_PATH}" ]]; then
  CHAT_TEMPLATE="${TOKENIZER_PATH}/chat_template.jinja"
fi
# 8192 covers ~85% of reasoning-task generations without truncation. Math/reasoning
# tasks at lower budgets show 30-50% mid-response truncation. MCQ hits EOS well
# before this, so no cost for short-answer tasks.
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=8192,temperature=0}"
if [[ -n "$GEN_KWARGS_OVERRIDE" ]]; then GEN_KWARGS="$GEN_KWARGS_OVERRIDE"; fi
# 4 vLLM workers per node = 1 per GH200 GPU (4 GPUs). Per-task SQLite handles
# 4 concurrent writers via WAL with sub-ms lock overhead.
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"

# WandB config
ENABLE_WANDB="${ENABLE_WANDB:-true}"
WANDB_ENTITY="${WANDB_ENTITY:-alvor}"
WANDB_PROJECT="${WANDB_PROJECT:-apertus-1p5-eval}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-}"
WANDB_LOG_SAMPLES="${WANDB_LOG_SAMPLES:-false}"

if [[ "$ENABLE_WANDB" == "true" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: ENABLE_WANDB=true but WANDB_API_KEY is empty (not in ~/.netrc). Jobs will fail fast."
fi

# ------------------------------------------------------------------
# Logging — Anunay's inner default uses scripts/logs which is group-readable
# but NOT writable for us (signal-53). Use the repo logs dir instead.
# ------------------------------------------------------------------
mkdir -p "$LOG_DIR" "$OUTPUT_PATH" "$CACHE_BASE" "$HF_HOME_PATH" "$NLTK_DATA_PATH" "$XDG_CACHE_HOME_PATH" "$VLLM_CACHE_ROOT_PATH" "$LMMS_EVAL_MODELS_CACHE_PATH"
cd "$REPO_ROOT"
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Pretty header
# ------------------------------------------------------------------
echo "========================================"
echo "Apertus eval"
echo "  mode:       $MODE"
echo "  models:     $(echo "$MODELS" | tr '\n' ',' | sed 's/,$//')"
echo "  tasks:      $(echo "$TASKS"  | tr '\n' ',' | sed 's/,$//')"
echo "  tokenizer:  $TOKENIZER_PATH"
echo "  template:   ${CHAT_TEMPLATE:-<tokenizer default>}"
echo "  cache base: $CACHE_BASE"
echo "  output:     $OUTPUT_PATH"
echo "  run id:     $RUN_ID"
echo "  logs:       $LOG_DIR"
echo "  slurm:      $SLURM_TEMPLATE"
echo "  wandb:      $ENABLE_WANDB ($WANDB_ENTITY/$WANDB_PROJECT)"
echo "========================================"

# ------------------------------------------------------------------
# Main loop: one submission per (task, model) tuple.
# Per-task cache dir is unique → trivially parallel writes during fill,
# bounded blast radius if any single task's cache is corrupted.
# ------------------------------------------------------------------
while IFS= read -r TASK; do
  [[ -z "$TASK" ]] && continue
  TASK_CACHE_DIR="$CACHE_BASE/$TASK"
  mkdir -p "$TASK_CACHE_DIR"

  while IFS= read -r MODEL_PATH; do
    [[ -z "$MODEL_PATH" ]] && continue
    if [[ ! -d "$MODEL_PATH" ]]; then
      echo "WARNING: model path not found, skipping: $MODEL_PATH" >&2
      continue
    fi

    # Derive a stable model label: parent dir name if path ends in /HF, else basename.
    MODEL_LABEL="$(basename "$MODEL_PATH")"
    [[ "$MODEL_LABEL" == "HF" ]] && MODEL_LABEL="$(basename "$(dirname "$MODEL_PATH")")"
    MODEL_LABEL="${MODEL_LABEL}${LABEL_SUFFIX}"
    WANDB_GROUP="${WANDB_GROUP_PREFIX}${MODEL_LABEL}"

    # The dashboard's run identity is the dir one level under runs-root, so nest
    # <MODEL_LABEL>/<RUN_ID>: the suffix becomes the canonical key and two models
    # in one call never share a dir.
    MODEL_OUTPUT_PATH="${OUTPUT_BASE}/${MODEL_LABEL}/${RUN_ID}"
    mkdir -p "$MODEL_OUTPUT_PATH"

    if [[ "$SUBMIT_MODE" == "interactive" ]]; then
      JOB_OUTPUT="${LOG_DIR}/eval_${MODE}_${TASK}_${MODEL_LABEL}_interactive.out"
      JOB_ERROR="${LOG_DIR}/eval_${MODE}_${TASK}_${MODEL_LABEL}_interactive.err"
    else
      JOB_OUTPUT="${LOG_DIR}/eval_${MODE}_%j.out"
      JOB_ERROR="${LOG_DIR}/eval_${MODE}_%j.err"
    fi

    echo "--- submit: task=$TASK  model=$MODEL_LABEL ---"
    echo "    cache: $TASK_CACHE_DIR"
    echo "    logs:  $JOB_OUTPUT / $JOB_ERROR"

    JOB_ARGS=(
      --model-path "$MODEL_PATH"
      --tokenizer-path "$TOKENIZER_PATH"
      --chat-template "$CHAT_TEMPLATE"
      --tasks "$TASK"
      --output-path "$MODEL_OUTPUT_PATH"
      --log-dir "$LOG_DIR"
      --hf-home "$HF_HOME_PATH"
      --nltk-data "$NLTK_DATA_PATH"
      --xdg-cache-home "$XDG_CACHE_HOME_PATH"
      --vllm-cache-root "$VLLM_CACHE_ROOT_PATH"
      --models-cache "$LMMS_EVAL_MODELS_CACHE_PATH"
      --num-processes "$NUM_PROCESSES"
      --batch-size "$BATCH_SIZE"
      --gen-kwargs "$GEN_KWARGS"
      --enable-image-token-cache true
      --image-token-cache-dir "$TASK_CACHE_DIR"
      --image-token-cache-mode "$MODE"
      --enable-wandb "$ENABLE_WANDB"
      --wandb-project "$WANDB_PROJECT"
      --wandb-entity "$WANDB_ENTITY"
      --wandb-group "$WANDB_GROUP"
      --wandb-log-samples "$WANDB_LOG_SAMPLES"
      --wandb-api-key "${WANDB_API_KEY:-}"
    )

    if [[ -n "$ENABLE_THINKING" ]]; then
      JOB_ARGS+=(--extra-model-args "enable_thinking=True" --wandb-run-name "$MODEL_LABEL")
    fi
    for ARG in "${EXTRA_FRAMEWORK_ARGS[@]}"; do
      JOB_ARGS+=(--extra-framework-config "${ARG}")
    done

    if [[ "$SUBMIT_MODE" == "interactive" ]]; then
      CMD=(bash "$SLURM_TEMPLATE" "${JOB_ARGS[@]}")
      echo "    submit: interactive bash ${SLURM_TEMPLATE}"
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf ' %q' "${CMD[@]}"
        printf ' >%q 2>%q\n' "$JOB_OUTPUT" "$JOB_ERROR"
      else
        "${CMD[@]}" >"$JOB_OUTPUT" 2>"$JOB_ERROR"
      fi
    else
      CMD=(sbatch --output "$JOB_OUTPUT" --error "$JOB_ERROR" "$SLURM_TEMPLATE" "${JOB_ARGS[@]}")
      echo "    submit: sbatch ${SLURM_TEMPLATE}"
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf ' %q' "${CMD[@]}"
        printf '\n'
      else
        "${CMD[@]}"
      fi
    fi
  done <<< "$MODELS"
done <<< "$TASKS"

echo "========================================"
echo "All submissions complete. Tail logs in: $LOG_DIR"
echo "========================================"
