#!/usr/bin/env bash
# Submit EleutherAI lm-evaluation-harness text benchmarks with Apertus.
#
# Defaults target the Apertus 1.5 70B checkpoint through lm_eval's HuggingFace
# backend. Multi-GPU runs use accelerate data parallelism.

set -euo pipefail

if [[ -z "${ORCH_REPO_ROOT:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ORCH_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  export ORCH_REPO_ROOT
fi

REPO_ROOT="${ORCH_REPO_ROOT}"
HARNESS_REPO_DIR="${LM_EVAL_HARNESS_REPO_DIR:-${REPO_ROOT}/third_party/lm-evaluation-harness}"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-${REPO_ROOT}/slurm/lm-evaluation-harness/eval_job.slurm}"
SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/task_suites/lm-evaluation-harness}"
OUTPUT_BASE="${OUTPUT_PATH:-${REPO_ROOT}/results/lm-evaluation-harness}"
LOG_BASE="${LOG_DIR:-${REPO_ROOT}/logs/lm-evaluation-harness}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
RUN_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_ID}"
RUN_LOG_DIR="${LOG_BASE}/${RUN_ID}"

DEFAULT_MODEL="/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-70b-sft-262k-3000"
DEFAULT_TOKENIZER="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"

MODEL_RAW=""
TASKS_RAW=""
SUITE="smoke"
SUBMIT_MODE="batch"
TOKENIZER_PATH="${LM_EVAL_TOKENIZER:-${TOKENIZER_PATH:-${DEFAULT_TOKENIZER}}}"
HF_DTYPE="${LM_EVAL_HF_DTYPE:-auto}"
HF_TRUST_REMOTE_CODE="${LM_EVAL_HF_TRUST_REMOTE_CODE:-false}"
BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
NUM_PROCESSES="${LM_EVAL_NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${LM_EVAL_MAIN_PROCESS_PORT:-29500}"
PARALLELIZE="${LM_EVAL_PARALLELIZE:-true}"
NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-}"
LIMIT="${LM_EVAL_LIMIT:-}"
GEN_KWARGS="${LM_EVAL_GEN_KWARGS:-}"
APPLY_CHAT_TEMPLATE="${LM_EVAL_APPLY_CHAT_TEMPLATE:-false}"
LOG_SAMPLES="${LM_EVAL_LOG_SAMPLES:-true}"
USE_CACHE="${LM_EVAL_USE_CACHE:-}"
CACHE_REQUESTS="${LM_EVAL_CACHE_REQUESTS:-}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
DRY_RUN=0
EXTRA_ARGS=()
EXTRA_FRAMEWORK_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  launchers/lm-evaluation-harness/eval.sh [--model <path>|<path>] [options] [-- lm_eval args...]

Options:
  --model <path>                 Model checkpoint. Default: Apertus 1.5 70B checkpoint.
  --tasks <csv|@file|file>       lm-evaluation-harness task names.
  --suite <name>                 Suite in task_suites/lm-evaluation-harness/. Default: smoke.
  --submit-mode <batch|interactive>
                                 Batch submits one Slurm job per task; interactive runs the job script directly.
  --backend <hf|huggingface>     Optional compatibility flag. Only hf is supported.
  --tokenizer-path <path>        Tokenizer path. Default: Apertus tokenizer.
  --hf-dtype <value>             HuggingFace dtype. Default: auto.
  --hf-trust-remote-code <bool>  HuggingFace trust_remote_code. Default: false.
  --batch-size <value>           lm_eval --batch_size for hf. Default: auto.
  --num-processes <int>          HF accelerate process count inside each task job. Default: 1.
  --main-process-port <int>      HF accelerate main process port. Default: 29500.
  --parallelize <true|false>     Shard the model across all visible GPUs. Default: true.
  --num-fewshot <int>            Optional lm_eval --num_fewshot override.
  --limit <n|fraction>           Optional lm_eval --limit.
  --gen-kwargs <json|kv>         Optional lm_eval --gen_kwargs for generation tasks.
  --apply-chat-template <true|false>
                                 Pass --apply_chat_template. Default: false.
  --log-samples <true|false>     Pass --log_samples. Default: true.
  --use-cache <dir>              lm_eval --use_cache directory.
  --cache-requests <true|refresh|delete>
                                 lm_eval --cache_requests value.
  --output-dir <path>            Shared run output root. Default: results/lm-evaluation-harness/$RUN_ID.
  --log-dir <path>               Shared run log root. Default: logs/lm-evaluation-harness/$RUN_ID.
  --time <hh:mm:ss>              Slurm walltime. Default: 12:00:00.
  --extra-framework-config <arg> Extra lm_eval argv token. Repeat for each token.
  --dry-run                      Print commands without executing.
  -h, --help                     Show this help.

Examples:
  bash launchers/lm-evaluation-harness/eval.sh --suite smoke --submit-mode batch
  bash launchers/lm-evaluation-harness/eval.sh --suite text --limit 100
  bash launchers/lm-evaluation-harness/eval.sh --num-processes 4 --tasks hellaswag
EOF
}

resolve_list() {
  local raw="$1"
  if [[ "${raw}" == @* ]]; then
    local file="${raw#@}"
    [[ -f "${file}" ]] || { echo "file not found: ${file}" >&2; exit 1; }
    sed -E 's/[[:space:]]*#.*$//' "${file}" | tr ',' '\n' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  elif [[ -f "${raw}" ]]; then
    sed -E 's/[[:space:]]*#.*$//' "${raw}" | tr ',' '\n' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  else
    echo "${raw}" | tr ',' '\n' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  fi
}

safe_name() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '_' | sed -E 's/^_+//; s/_+$//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_RAW="$2"; shift 2 ;;
    --tasks|--data) TASKS_RAW="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --submit-mode) SUBMIT_MODE="$2"; shift 2 ;;
    --backend)
      case "$2" in hf|huggingface) ;; *) echo "--backend only supports hf for now" >&2; exit 2 ;; esac
      shift 2
      ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --hf-dtype) HF_DTYPE="$2"; shift 2 ;;
    --hf-trust-remote-code) HF_TRUST_REMOTE_CODE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    --main-process-port) MAIN_PROCESS_PORT="$2"; shift 2 ;;
    --parallelize) PARALLELIZE="$2"; shift 2 ;;
    --num-fewshot) NUM_FEWSHOT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --gen-kwargs) GEN_KWARGS="$2"; shift 2 ;;
    --apply-chat-template) APPLY_CHAT_TEMPLATE="$2"; shift 2 ;;
    --log-samples) LOG_SAMPLES="$2"; shift 2 ;;
    --use-cache) USE_CACHE="$2"; shift 2 ;;
    --cache-requests) CACHE_REQUESTS="$2"; shift 2 ;;
    --output-dir|--output-path) RUN_OUTPUT_DIR="$2"; shift 2 ;;
    --log-dir) RUN_LOG_DIR="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --extra-framework-config|--extra-framework-arg) EXTRA_FRAMEWORK_ARGS+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -z "${MODEL_RAW}" ]]; then
        MODEL_RAW="$1"
        shift
      else
        echo "unexpected positional argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

case "${SUBMIT_MODE}" in batch|interactive) ;; *) echo "--submit-mode must be batch or interactive" >&2; exit 2 ;; esac

if [[ -z "${MODEL_RAW}" ]]; then
  MODEL_RAW="${DEFAULT_MODEL}"
fi

case "${RUN_OUTPUT_DIR}" in /*) ;; *) RUN_OUTPUT_DIR="${REPO_ROOT}/${RUN_OUTPUT_DIR}" ;; esac
case "${RUN_LOG_DIR}" in /*) ;; *) RUN_LOG_DIR="${REPO_ROOT}/${RUN_LOG_DIR}" ;; esac
case "${HARNESS_REPO_DIR}" in /*) ;; *) HARNESS_REPO_DIR="${REPO_ROOT}/${HARNESS_REPO_DIR}" ;; esac
case "${SLURM_TEMPLATE}" in /*) ;; *) SLURM_TEMPLATE="${REPO_ROOT}/${SLURM_TEMPLATE}" ;; esac

[[ -d "${HARNESS_REPO_DIR}" ]] || { echo "lm-evaluation-harness repo not found: ${HARNESS_REPO_DIR}" >&2; exit 1; }
[[ -f "${SLURM_TEMPLATE}" ]] || { echo "slurm template not found: ${SLURM_TEMPLATE}" >&2; exit 1; }

MODELS="$(resolve_list "${MODEL_RAW}")"
if [[ -z "${TASKS_RAW}" ]]; then
  [[ "${SUITE}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid suite name: ${SUITE}" >&2; exit 1; }
  suite_file="${SUITE_DIR}/${SUITE//-/_}.txt"
  [[ -f "${suite_file}" ]] || {
    available="$(cd "${SUITE_DIR}" && ls *.txt 2>/dev/null | sed 's/\.txt$//' | tr '_' '-' | paste -sd, -)"
    echo "--suite must be one of: ${available} (got: ${SUITE})" >&2
    exit 1
  }
  TASKS="$(resolve_list "@${suite_file}")"
else
  TASKS="$(resolve_list "${TASKS_RAW}")"
fi

[[ -n "${MODELS}" ]] || { echo "no models resolved" >&2; exit 1; }
[[ -n "${TASKS}" ]] || { echo "no tasks resolved" >&2; exit 1; }

while IFS='=' read -r key _; do
  [[ "${key}" == SLURM_SPANK* ]] && unset "${key}"
done < <(env)
export LD_LIBRARY_PATH="/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}"

mkdir -p "${RUN_OUTPUT_DIR}" "${RUN_LOG_DIR}"

echo "========================================"
echo "lm-evaluation-harness submit"
echo "  repo:       ${HARNESS_REPO_DIR}"
echo "  backend:    hf"
echo "  processes:  ${NUM_PROCESSES}"
echo "  models:     $(echo "${MODELS}" | tr '\n' ',' | sed 's/,$//')"
echo "  tasks:      $(echo "${TASKS}" | tr '\n' ',' | sed 's/,$//')"
echo "  output:     ${RUN_OUTPUT_DIR}"
echo "  logs:       ${RUN_LOG_DIR}"
echo "  slurm:      ${SLURM_TEMPLATE}"
echo "========================================"

while IFS= read -r TASK; do
  [[ -z "${TASK}" ]] && continue
  TASK_SLUG="$(safe_name "${TASK}")"
  while IFS= read -r MODEL_PATH; do
    [[ -z "${MODEL_PATH}" ]] && continue
    MODEL_LABEL="$(basename "${MODEL_PATH%/}")"
    MODEL_SLUG="$(safe_name "${MODEL_LABEL}")"
    TASK_OUTPUT_DIR="${RUN_OUTPUT_DIR}/${MODEL_SLUG}/${TASK_SLUG}"
    JOB_OUTPUT="${RUN_LOG_DIR}/lm-eval-${MODEL_SLUG}-${TASK_SLUG}_${SUBMIT_MODE}.out"
    JOB_ERROR="${RUN_LOG_DIR}/lm-eval-${MODEL_SLUG}-${TASK_SLUG}_${SUBMIT_MODE}.err"
    mkdir -p "${TASK_OUTPUT_DIR}"

    JOB_ARGS=(
      --harness-repo-dir "${HARNESS_REPO_DIR}"
      --model "${MODEL_PATH}"
      --task "${TASK}"
      --output-dir "${TASK_OUTPUT_DIR}"
      --tokenizer-path "${TOKENIZER_PATH}"
      --hf-dtype "${HF_DTYPE}"
      --hf-trust-remote-code "${HF_TRUST_REMOTE_CODE}"
      --batch-size "${BATCH_SIZE}"
      --num-processes "${NUM_PROCESSES}"
      --main-process-port "${MAIN_PROCESS_PORT}"
      --parallelize "${PARALLELIZE}"
      --apply-chat-template "${APPLY_CHAT_TEMPLATE}"
      --log-samples "${LOG_SAMPLES}"
    )
    [[ -n "${NUM_FEWSHOT}" ]] && JOB_ARGS+=(--num-fewshot "${NUM_FEWSHOT}")
    [[ -n "${LIMIT}" ]] && JOB_ARGS+=(--limit "${LIMIT}")
    [[ -n "${GEN_KWARGS}" ]] && JOB_ARGS+=(--gen-kwargs "${GEN_KWARGS}")
    [[ -n "${USE_CACHE}" ]] && JOB_ARGS+=(--use-cache "${USE_CACHE}")
    [[ -n "${CACHE_REQUESTS}" ]] && JOB_ARGS+=(--cache-requests "${CACHE_REQUESTS}")
    for ARG in "${EXTRA_FRAMEWORK_ARGS[@]}"; do
      JOB_ARGS+=(--extra-framework-config "${ARG}")
    done
    [[ "${DRY_RUN}" -eq 1 ]] && JOB_ARGS+=(--dry-run)
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && JOB_ARGS+=(-- "${EXTRA_ARGS[@]}")

    echo "--- submit: task=${TASK} model=${MODEL_LABEL} output=${TASK_OUTPUT_DIR} ---"
    echo "    logs: ${JOB_OUTPUT} / ${JOB_ERROR}"
    if [[ "${SUBMIT_MODE}" == "interactive" ]]; then
      CMD=(bash "${SLURM_TEMPLATE}" "${JOB_ARGS[@]}")
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf ' %q' "${CMD[@]}"
        printf ' >%q 2>%q\n' "${JOB_OUTPUT}" "${JOB_ERROR}"
      else
        "${CMD[@]}" >"${JOB_OUTPUT}" 2>"${JOB_ERROR}"
      fi
    else
      CMD=(sbatch --job-name "lm-eval-${TASK_SLUG}" --output "${JOB_OUTPUT}" --error "${JOB_ERROR}" --time "${SBATCH_TIME}" "${SLURM_TEMPLATE}" "${JOB_ARGS[@]}")
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf ' %q' "${CMD[@]}"
        printf '\n'
      else
        "${CMD[@]}"
      fi
    fi
  done <<< "${MODELS}"
done <<< "${TASKS}"

echo "========================================"
echo "lm-evaluation-harness submissions complete. Logs: ${RUN_LOG_DIR}"
echo "========================================"
