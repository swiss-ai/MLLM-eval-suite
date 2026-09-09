#!/usr/bin/env bash
# eval.sh — Apertus VLM eval CLI (per-task SQLite cache, single user entry point).
#
# Usage:
#   bash eval.sh <model> [--tasks T | --suite full|smoke|audio-full|audio-smoke|audio-llm-eval|visual-llm-judge|geospatial-full|geospatial-smoke] [--mode fill|readonly] [--submit-mode batch|interactive] [--help] [-- job args...]
#
# <model> forms:
#   /path/to/ckpt                       single path
#   /a,/b,/c                            comma-separated paths
#   @file.txt                           one path per line (comments # and blanks OK)
#
# --tasks   comma-separated, @file.txt, or direct path to a suite file.
# --suite   named curation: full (default), smoke, audio-full, audio-smoke, audio-llm-eval,
#           visual-llm-judge, geospatial-full, geospatial-smoke.
#           visual-llm-judge holds the judge-scored visual tasks; it is disjoint
#           from full and the launcher refuses to submit it without a judge.
# --mode    fill|readonly. Both modes use the shared cache directly with preload
#           on and writes enabled.
# --size         8b|70b parallelism profile: 8b = 4 DP workers (TP=1); 70b = 1 worker,
#                model sharded across 4 GPUs (TP=4, gpu-mem 0.85). Default: 8b.
# --submit-mode  batch|interactive. Batch submits one sbatch per task/model pair.
#                Interactive runs the job script directly with bash so it uses
#                the current shell's node allocation.
#
# Examples:
#   bash eval.sh /path/to/ckpt                          # full suite, fill mode
#   bash eval.sh /path/to/ckpt --suite smoke            # 3-task sanity
#   bash eval.sh /path/to/ckpt --tasks mmmu_val,chartqa # specific tasks
#   bash eval.sh /a,/b,/c                               # multiple models
#   bash eval.sh @models.txt --tasks @custom.txt
#   bash eval.sh /path/to/ckpt --mode fill              # explicit default
#   bash eval.sh /path/to/ckpt --submit-mode interactive --suite audio-smoke
#   bash eval.sh /path/to/ckpt --tasks google_fleurs -- --gpu-memory-utilization 0.75
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
source "${ORCH_REPO_ROOT}/slurm/shared/sbatch_overrides.sh"
LMMS_CACHE_ROOT="${LMMS_CACHE_ROOT:-${REPO_ROOT}/cache/lmms-eval}"
CACHE_BASE="${CACHE_BASE:-${LMMS_CACHE_ROOT}/image_token_cache}"
declare -A HF_AUTH_CHECKED
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
SUITE_VISUAL_LLM_JUDGE="${SUITE_VISUAL_LLM_JUDGE:-${SUITE_DIR}/visual_llm_judge.txt}"
SUITE_GEOSPATIAL_FULL="${SUITE_GEOSPATIAL_FULL:-${SUITE_DIR}/geospatial_full.txt}"
SUITE_GEOSPATIAL_SMOKE="${SUITE_GEOSPATIAL_SMOKE:-${SUITE_DIR}/geospatial_smoke.txt}"

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
SIZE="8b"
SUBMIT_MODE="batch"
ENABLE_THINKING=""
GEN_KWARGS_OVERRIDE=""
LABEL_SUFFIX=""
PASSTHROUGH=()
EXTRA_FRAMEWORK_ARGS=()
DEBUG_MODE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)  usage; exit 0 ;;
    --tasks)    TASKS_RAW="$2"; shift 2 ;;
    --suite)    SUITE="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --size)     SIZE="$2"; shift 2 ;;
    --submit-mode) SUBMIT_MODE="$2"; shift 2 ;;
    --enable-thinking) ENABLE_THINKING=1; shift ;;
    --gen-kwargs) GEN_KWARGS_OVERRIDE="$2"; shift 2 ;;
    --label-suffix) LABEL_SUFFIX="$2"; shift 2 ;;
    --extra-model-args) EXTRA_MODEL_ARGS="$2"; shift 2 ;;
    --debug-mode) DEBUG_MODE=1; shift ;;
    --extra-framework-config|--extra-framework-arg)
      if [[ "$2" == "--debug-mode" ]]; then
        DEBUG_MODE=1
      else
        EXTRA_FRAMEWORK_ARGS+=("$2")
      fi
      shift 2
      ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --)         shift; PASSTHROUGH+=("$@"); break ;;
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
# Parallelism profile, mirroring the VLMEvalKit launcher: 8b = 4 DP workers on one
# node; 70b = one worker with the model tensor-sharded across all 4 GPUs.
case "$SIZE" in
  8b)  SIZE_NUM_PROCESSES=4; SIZE_EXTRA_MODEL_ARGS="";                       SIZE_GPU_MEM="" ;;
  70b) SIZE_NUM_PROCESSES=1; SIZE_EXTRA_MODEL_ARGS="tensor_parallel_size=4"; SIZE_GPU_MEM="0.85" ;;
  *) echo "--size must be 8b|70b (got: $SIZE)" >&2; exit 1 ;;
esac
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-$SIZE_EXTRA_MODEL_ARGS}"
if [[ -n "$ENABLE_THINKING" ]]; then
  EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:+$EXTRA_MODEL_ARGS,}enable_thinking=True"
fi
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
    visual-llm-judge) TASKS=$(resolve_list "$SUITE_VISUAL_LLM_JUDGE") ;;
    geospatial-full) TASKS=$(resolve_list "$SUITE_GEOSPATIAL_FULL") ;;
    geospatial-smoke) TASKS=$(resolve_list "$SUITE_GEOSPATIAL_SMOKE") ;;
    *) echo "--suite must be full|smoke|audio-full|audio-smoke|audio-llm-eval|visual-llm-judge|geospatial-full|geospatial-smoke (got: $SUITE)" >&2; exit 1 ;;
  esac
fi
[[ -z "$TASKS" ]] && { echo "no tasks resolved" >&2; exit 1; }

# Judge-scored tasks silently mark every sample wrong when the judge is absent
# (dummy provider, or a client that never constructs). Fail loud instead, the
# same contract the VLMEvalKit launcher enforces. ALLOW_NO_JUDGE=1 overrides.
if [[ -f "$SUITE_VISUAL_LLM_JUDGE" && "${ALLOW_NO_JUDGE:-0}" != "1" ]]; then
  JUDGE_HITS="$(comm -12 <(echo "$TASKS" | sort -u) <(resolve_list "@${SUITE_VISUAL_LLM_JUDGE}" | sort -u) || true)"
  if [[ -n "$JUDGE_HITS" ]]; then
    for t in $JUDGE_HITS; do
      case "$t" in
        babyvision)
          [[ -n "${BABYVISION_API_KEY:-}" ]] || { echo "ERROR: task 'babyvision' needs BABYVISION_API_KEY (set ALLOW_NO_JUDGE=1 to override)" >&2; exit 1; } ;;
        healthbench)
          [[ -n "${HEALTHBENCH_GRADER_BASE_URL:-}" ]] || { echo "ERROR: task 'healthbench' needs HEALTHBENCH_GRADER_BASE_URL (a live grader endpoint)" >&2; exit 1; } ;;
        *)
          if [[ "${API_TYPE:-}" != "openai" || -z "${OPENAI_API_KEY:-}" ]]; then
            echo "ERROR: judge task '$t' needs API_TYPE=openai and OPENAI_API_KEY; without them it is scored by the dummy judge (all-zero)." >&2
            exit 1
          fi ;;
      esac
    done
  fi
fi

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
# 16384 is the Artificial Analysis standard cap for non-thinking evals; MCQ hits
# EOS well before this, so no cost for short-answer tasks.
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=16384,temperature=0}"
if [[ -n "$GEN_KWARGS_OVERRIDE" ]]; then GEN_KWARGS="$GEN_KWARGS_OVERRIDE"; fi
# 4 vLLM workers per node = 1 per GH200 GPU (4 GPUs). Per-task SQLite handles
# 4 concurrent writers via WAL with sub-ms lock overhead.
NUM_PROCESSES="${NUM_PROCESSES:-$SIZE_NUM_PROCESSES}"
BATCH_SIZE="${BATCH_SIZE:-512}"
# The image-token cache memoizes the discrete image->VQ-token conversion shared
# by every Apertus checkpoint; foreign continuous-encoder models run once and
# have nothing to amortize.
if [[ "${MODEL_BACKEND:-apertus_1p5_vllm}" == apertus* ]]; then
  ENABLE_IMAGE_TOKEN_CACHE="${ENABLE_IMAGE_TOKEN_CACHE:-true}"
else
  ENABLE_IMAGE_TOKEN_CACHE="${ENABLE_IMAGE_TOKEN_CACHE:-false}"
fi

# WandB config
ENABLE_WANDB="${ENABLE_WANDB:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-alvor}"
WANDB_PROJECT="${WANDB_PROJECT:-apertus-1p5-eval}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-}"
WANDB_LOG_SAMPLES="${WANDB_LOG_SAMPLES:-false}"

if [[ "$ENABLE_WANDB" == "true" ]]; then
  echo "NOTE: the slurm template currently forces W&B off; --enable-wandb has no effect."
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
echo "  gen kwargs: $GEN_KWARGS"
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
    # Foreign backends may take HF ids; only path-shaped models must exist.
    if [[ ! -d "$MODEL_PATH" && ( "$MODEL_PATH" == /* || "${MODEL_BACKEND:-apertus_1p5_vllm}" == apertus* ) ]]; then
      echo "WARNING: model path not found, skipping: $MODEL_PATH" >&2
      continue
    fi

    # A gated repo we cannot read 401s at load time, which slurm records as a
    # two-minute COMPLETED with no score. Check the hub once per model before
    # spending an allocation, against the same HF_HOME the job will use.
    if [[ ! -d "$MODEL_PATH" && -z "${HF_AUTH_CHECKED[$MODEL_PATH]:-}" ]]; then
      if HF_HOME="$HF_HOME_PATH" python3 -c \
          'import sys; from huggingface_hub import auth_check; auth_check(sys.argv[1])' \
          "$MODEL_PATH" 2>/dev/null; then
        HF_AUTH_CHECKED[$MODEL_PATH]=ok
      else
        HF_AUTH_CHECKED[$MODEL_PATH]=denied
      fi
    fi
    if [[ "${HF_AUTH_CHECKED[$MODEL_PATH]:-}" == "denied" ]]; then
      echo "ERROR: no read access to '$MODEL_PATH' (gated repo, or no token at \$HF_HOME/token); skipping" >&2
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
    # Per-task subdir: lmms-eval names results.json by wall-clock timestamp, so two
    # single-task jobs finishing in the same second clobber each other in a shared dir.
    MODEL_OUTPUT_PATH="${OUTPUT_BASE}/${MODEL_LABEL}/${RUN_ID}/${TASK}"
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
      --enable-image-token-cache "$ENABLE_IMAGE_TOKEN_CACHE"
      --image-token-cache-dir "$TASK_CACHE_DIR"
      --image-token-cache-mode "$MODE"
      --enable-wandb "$ENABLE_WANDB"
      --wandb-project "$WANDB_PROJECT"
      --wandb-entity "$WANDB_ENTITY"
      --wandb-group "$WANDB_GROUP"
      --wandb-log-samples "$WANDB_LOG_SAMPLES"
      --wandb-api-key "${WANDB_API_KEY:-}"
    )
    JOB_ARGS+=("${PASSTHROUGH[@]}")

    if [[ -n "$EXTRA_MODEL_ARGS" ]]; then
      JOB_ARGS+=(--extra-model-args "$EXTRA_MODEL_ARGS")
    fi
    if [[ -n "$SIZE_GPU_MEM" ]]; then
      JOB_ARGS+=(--gpu-memory-utilization "$SIZE_GPU_MEM")
    fi
    if [[ -n "$ENABLE_THINKING" ]]; then
      JOB_ARGS+=(--wandb-run-name "$MODEL_LABEL")
    fi

    if [[ "${DEBUG_MODE}" -eq 1 ]]; then
      JOB_ARGS+=(--debug-mode)
    fi
    for ARG in "${EXTRA_FRAMEWORK_ARGS[@]}"; do
      JOB_ARGS+=(--extra-framework-config "${ARG}")
    done

    if [[ "$DRY_RUN" -eq 1 ]]; then
      REDACTED=(); MASK_NEXT=0
      for a in "$SLURM_TEMPLATE" "${JOB_ARGS[@]}"; do
        if [[ "$MASK_NEXT" -eq 1 ]]; then REDACTED+=("***"); MASK_NEXT=0
        else REDACTED+=("$a"); [[ "$a" == "--wandb-api-key" ]] && MASK_NEXT=1; fi
      done
      printf '    dry-run:'; printf ' %q' "${REDACTED[@]}"; printf '\n'
    elif [[ "$SUBMIT_MODE" == "interactive" ]]; then
      echo "    submit: interactive bash ${SLURM_TEMPLATE}"
      bash "$SLURM_TEMPLATE" "${JOB_ARGS[@]}" >"$JOB_OUTPUT" 2>"$JOB_ERROR"
    else
      echo "    submit: sbatch ${SLURM_TEMPLATE}"
      TIME_ARGS=()
      [[ -n "${SBATCH_TIME:-}" ]] && TIME_ARGS=(--time "${SBATCH_TIME}")
      sbatch \
        "${SBATCH_OVERRIDES[@]}" \
        "${TIME_ARGS[@]}" \
        --output "$JOB_OUTPUT" \
        --error  "$JOB_ERROR" \
        "$SLURM_TEMPLATE" \
        "${JOB_ARGS[@]}"
    fi
  done <<< "$MODELS"
done <<< "$TASKS"

echo "========================================"
echo "All submissions complete. Tail logs in: $LOG_DIR"
echo "========================================"
