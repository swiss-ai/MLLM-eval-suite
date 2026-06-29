#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=lm-eval-harness
#SBATCH --environment=/capstor/store/cscs/swissai/infra01/multimodal-eval/MLLM-eval-suite/toml/shared/apertus-vllm-vision-eval-prod.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=logs/lm-evaluation-harness/lm_eval_%j.out
#SBATCH --error=logs/lm-evaluation-harness/lm_eval_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ORCH_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ORCH_REPO_ROOT="${ORCH_REPO_ROOT:-${DEFAULT_ORCH_REPO_ROOT}}"

HARNESS_REPO_DIR="${LM_EVAL_HARNESS_REPO_DIR:-${ORCH_REPO_ROOT}/third_party/lm-evaluation-harness}"
MODEL_PATH=""
TASK_NAME=""
OUTPUT_DIR=""
TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
HF_DTYPE="auto"
HF_TRUST_REMOTE_CODE="false"
BATCH_SIZE="auto"
NUM_PROCESSES=4
MAIN_PROCESS_PORT=29500
PARALLELIZE="true"
NUM_FEWSHOT=""
LIMIT=""
GEN_KWARGS=""
APPLY_CHAT_TEMPLATE="false"
LOG_SAMPLES="true"
USE_CACHE=""
CACHE_REQUESTS=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  eval_job.slurm --model <path> --task <task> --output-dir <path> [options] [-- lm_eval args...]

Options:
  --backend <hf|huggingface>
  --harness-repo-dir <path>
  --model <path>
  --task <name>
  --output-dir <path>
  --tokenizer-path <path>
  --hf-dtype <value>
  --hf-trust-remote-code <bool>
  --batch-size <value>
  --num-processes <int>
  --main-process-port <int>
  --parallelize <true|false>
  --num-fewshot <int>
  --limit <n|fraction>
  --gen-kwargs <json|kv>
  --apply-chat-template <true|false>
  --log-samples <true|false>
  --use-cache <dir>
  --cache-requests <true|refresh|delete>
  --extra-framework-config <arg>
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      case "$2" in hf|huggingface) ;; *) echo "--backend only supports hf for now" >&2; exit 2 ;; esac
      shift 2
      ;;
    --harness-repo-dir) HARNESS_REPO_DIR="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --task|--tasks) TASK_NAME="$2"; shift 2 ;;
    --output-dir|--output-path) OUTPUT_DIR="$2"; shift 2 ;;
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
    --extra-framework-config|--extra-framework-arg) EXTRA_ARGS+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || { echo "--model is required" >&2; exit 2; }
[[ -n "${TASK_NAME}" ]] || { echo "--task is required" >&2; exit 2; }
[[ -n "${OUTPUT_DIR}" ]] || { echo "--output-dir is required" >&2; exit 2; }

case "${HARNESS_REPO_DIR}" in /*) ;; *) HARNESS_REPO_DIR="${ORCH_REPO_ROOT}/${HARNESS_REPO_DIR}" ;; esac
case "${OUTPUT_DIR}" in /*) ;; *) OUTPUT_DIR="${ORCH_REPO_ROOT}/${OUTPUT_DIR}" ;; esac
[[ -d "${HARNESS_REPO_DIR}" ]] || { echo "lm-evaluation-harness repo not found: ${HARNESS_REPO_DIR}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}" "${ORCH_REPO_ROOT}/cache/hf" "${ORCH_REPO_ROOT}/cache/xdg" "${ORCH_REPO_ROOT}/cache/models"

export PATH="/opt/venv/bin:${PATH}"
export PYTHONPATH="${HARNESS_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${ORCH_REPO_ROOT}/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ORCH_REPO_ROOT}/cache/xdg}"
export LMMS_EVAL_MODELS_CACHE="${LMMS_EVAL_MODELS_CACHE:-${ORCH_REPO_ROOT}/cache/models}"
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}"

MODEL_ARGS=(
  "pretrained=${MODEL_PATH}"
  "dtype=${HF_DTYPE}"
  "trust_remote_code=${HF_TRUST_REMOTE_CODE}"
  "parallelize=${PARALLELIZE}"
)
if [[ -n "${TOKENIZER_PATH}" ]]; then
  MODEL_ARGS+=("tokenizer=${TOKENIZER_PATH}")
fi

LMEVAL_ARGS=(-m lm_eval --model hf --model_args "${MODEL_ARGS[@]}" --tasks "${TASK_NAME}" --output_path "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}")
[[ -n "${NUM_FEWSHOT}" ]] && LMEVAL_ARGS+=(--num_fewshot "${NUM_FEWSHOT}")
[[ -n "${LIMIT}" ]] && LMEVAL_ARGS+=(--limit "${LIMIT}")
[[ -n "${GEN_KWARGS}" ]] && LMEVAL_ARGS+=(--gen_kwargs "${GEN_KWARGS}")
[[ -n "${USE_CACHE}" ]] && LMEVAL_ARGS+=(--use_cache "${USE_CACHE}")
[[ -n "${CACHE_REQUESTS}" ]] && LMEVAL_ARGS+=(--cache_requests "${CACHE_REQUESTS}")
case "$(printf '%s' "${APPLY_CHAT_TEMPLATE}" | tr '[:upper:]' '[:lower:]')" in
  1|true|t|yes|y|on) LMEVAL_ARGS+=(--apply_chat_template) ;;
esac
case "$(printf '%s' "${LOG_SAMPLES}" | tr '[:upper:]' '[:lower:]')" in
  1|true|t|yes|y|on) LMEVAL_ARGS+=(--log_samples) ;;
esac
LMEVAL_ARGS+=("${EXTRA_ARGS[@]}")

case "$(printf '%s' "${PARALLELIZE}" | tr '[:upper:]' '[:lower:]')" in
  1|true|t|yes|y|on)
    CMD=(python "${LMEVAL_ARGS[@]}")
    ;;
  *)
    CMD=(accelerate launch --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}" "${LMEVAL_ARGS[@]}")
    ;;
esac

echo "========== lm-evaluation-harness launch =========="
echo "ORCH_REPO_ROOT=${ORCH_REPO_ROOT}"
echo "HARNESS_REPO_DIR=${HARNESS_REPO_DIR}"
echo "BACKEND=hf"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TASK_NAME=${TASK_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PARALLELIZE=${PARALLELIZE}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
echo "HF_HOME=${HF_HOME}"
printf 'CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "=================================================="

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

cd "${HARNESS_REPO_DIR}"
"${CMD[@]}"
