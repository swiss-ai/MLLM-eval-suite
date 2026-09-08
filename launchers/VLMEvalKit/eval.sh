#!/usr/bin/env bash
# Submit Apertus VLMEvalKit jobs with the dedicated Apertus vLLM runtime.
#
# Invoked by launchers/eval.sh (which sets ORCH_REPO_ROOT). Direct examples:
#   ORCH_REPO_ROOT=$PWD bash launchers/VLMEvalKit/eval.sh --data 3DSRBench --model Apertus-1p5-8B
#   ORCH_REPO_ROOT=$PWD bash launchers/VLMEvalKit/eval.sh --data 3DSRBench --mode infer
#   ORCH_REPO_ROOT=$PWD bash launchers/VLMEvalKit/eval.sh --data @datasets.txt --model @models.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${ORCH_REPO_ROOT:-}" ]]; then
  echo "ORCH_REPO_ROOT must be set by the orchestration launcher." >&2
  exit 2
fi
REPO_ROOT="${ORCH_REPO_ROOT}"
REPO_DIR="${REPO_DIR:-${REPO_ROOT}/third_party/VLMEvalKit}"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-${REPO_ROOT}/slurm/VLMEvalKit/eval_job.slurm}"
source "${REPO_ROOT}/slurm/shared/sbatch_overrides.sh"

RESPONSE_CACHE="${VLMEVAL_RESPONSE_CACHE:-${REPO_ROOT}/cache/VLMEvalKit}"
IMAGE_TOKEN_CACHE_BASE="${IMAGE_TOKEN_CACHE_BASE:-}"
LMU_DATA="${LMUData:-${REPO_ROOT}/cache/VLMEvalKit/LMUData}"
WORK_BASE="${WORK_BASE:-${REPO_ROOT}/results/VLMEvalKit}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
WORK_BASE="${WORK_BASE}/${RUN_ID}"
RUNTIME_CACHE="${RUNTIME_CACHE:-${REPO_ROOT}/cache}"
[[ -n "${APERTUS_MAX_MODEL_LEN+x}" ]] && USER_APERTUS_MAX_MODEL_LEN="${APERTUS_MAX_MODEL_LEN}"
LOG_BASE="${LOG_DIR:-${REPO_ROOT}/logs/VLMEvalKit}"
LOG_DIR="${LOG_BASE}/${RUN_ID}"

DEFAULT_MODEL="Apertus-1p5-8B"

MODELS_RAW="${DEFAULT_MODEL}"
DATA_RAW=""
SUITE="smoke"
MODE="all"
SUBMIT_MODE="batch"
NODES="${NODES:-1}"
SIZE="${SIZE:-8b}"
BATCH_SIZE="${BATCH_SIZE:-512}"
IMAGE_TOKEN_CACHE_MODE="${IMAGE_TOKEN_CACHE_MODE:-fill}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"
DRY_RUN=0

usage() {
  sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --model <name[,name]|@file>       VLMEvalKit model name(s). Default: Apertus-1p5-8B.
  --data, --tasks <name[,name]|@file>
                                    Dataset name(s). Default suite smoke = 3DSRBench.
  --suite <name>                    Resolves to task_suites/VLMEvalKit/<name>.txt
                                    (hyphens in <name> map to underscores in
                                    the filename). Drop a new .txt in suites/
                                    to add a suite; no script edit needed.
  --mode all|infer|eval             VLMEvalKit run mode. Default: all.
  --submit-mode batch|interactive
                                    Batch submits via sbatch; interactive runs
                                    the job script directly with bash.
  --nodes <int>                     Number of slurm nodes (multi-node DP). Default: 1.
                                    Total world size = nodes * num-processes.
  --num-processes <int>             DP workers per node (= GPUs per node). Default: 4.
  --size <8b|70b>                   Parallelism profile: 8b (TP=1, 4 DP workers) | 70b
                                    (TP=4, 1 worker, model sharded across 4 GPUs). Default: 8b.
  --tensor-parallel-size <int>      Override vLLM tensor_parallel_size (advanced; --size sets it).
  --gpu-memory-utilization <float>  Override vLLM gpu_memory_utilization (advanced; --size sets it).
  --batch-size <int>                Batch size value passed through/logged for the framework. Default: 512.
  --work-base <path>                Root for VLMEvalKit outputs.
  --response-cache <path>           SQLite response cache root.
  --image-token-cache-base <path>   Apertus image-token cache base. Default: response-cache/image_token_cache.
  --enable-image-token-cache <true|false>
                                    Export Apertus vLLM image-token cache env. Default: true.
  --image-token-cache-mode <fill|readonly>
                                    fill builds the cache; readonly uses a sealed one. Default: fill.
  --lmu-data <path>                 Persistent LMUData root for VLMEvalKit datasets.
  --runtime-cache <path>            HF/XDG/vLLM runtime cache root.
  --log-dir <path>                  Slurm stdout/stderr directory.
  --time <hh:mm:ss>                 Slurm time limit. Default: 04:00:00.
  --main-process-port <int>         torch.distributed master port.
  --dry-run                         Print the submission command without executing it.
  -h, --help                        Show this help.
EOF
}

resolve_list() {
  local raw="$1"
  if [[ "${raw}" == @* ]]; then
    local file="${raw#@}"
    [[ -f "${file}" ]] || { echo "file not found: ${file}" >&2; exit 1; }
    sed -E 's/[[:space:]]*#.*$//' "${file}" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  elif [[ -f "${raw}" ]]; then
    sed -E 's/[[:space:]]*#.*$//' "${raw}" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  else
    echo "${raw}" | tr ',' '\n' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  fi
}

safe_name() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '_' | sed -E 's/^_+//; s/_+$//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODELS_RAW="$2"; shift 2 ;;
    --data|--tasks)
      DATA_RAW="$2"; shift 2 ;;
    --suite)
      SUITE="$2"; shift 2 ;;
    --mode)
      MODE="$2"; shift 2 ;;
    --submit-mode)
      SUBMIT_MODE="$2"; shift 2 ;;
    --nodes)
      NODES="$2"; shift 2 ;;
    --num-processes)
      NUM_PROCESSES="$2"; shift 2 ;;
    --size)
      SIZE="$2"; shift 2 ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --work-base)
      WORK_BASE="$2"; shift 2 ;;
    --response-cache)
      RESPONSE_CACHE="$2"; shift 2 ;;
    --image-token-cache-base)
      IMAGE_TOKEN_CACHE_BASE="$2"; shift 2 ;;
    --enable-image-token-cache)
      ENABLE_IMAGE_TOKEN_CACHE="$2"; shift 2 ;;
    --image-token-cache-mode)
      IMAGE_TOKEN_CACHE_MODE="$2"; shift 2 ;;
    --lmu-data)
      LMU_DATA="$2"; shift 2 ;;
    --runtime-cache)
      RUNTIME_CACHE="$2"; shift 2 ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    --time)
      SBATCH_TIME="$2"; shift 2 ;;
    --main-process-port)
      MAIN_PROCESS_PORT="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Parallelism profile by model size, identical to the lmms-eval launcher: 8b
# fits one GH200 (4 DP workers, TP=1); 70b shards one model across all 4 GPUs
# (TP=4, one worker). Explicit env/flag overrides any single knob.
case "${SIZE}" in
  8b)  _NUM_PROCESSES=4; _TENSOR_PARALLEL_SIZE=1; _GPU_MEMORY_UTILIZATION=0.6 ;;
  70b) _NUM_PROCESSES=1; _TENSOR_PARALLEL_SIZE=4; _GPU_MEMORY_UTILIZATION=0.85 ;;
  *)   echo "--size must be 8b|70b (got: ${SIZE})" >&2; exit 1 ;;
esac
NUM_PROCESSES="${NUM_PROCESSES:-$_NUM_PROCESSES}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$_TENSOR_PARALLEL_SIZE}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$_GPU_MEMORY_UTILIZATION}"

case "${MODE}" in
  all|infer|eval) ;;
  *) echo "--mode must be all, infer, or eval (got: ${MODE})" >&2; exit 1 ;;
esac

case "${SUBMIT_MODE}" in
  batch|interactive) ;;
  *) echo "--submit-mode must be batch or interactive (got: ${SUBMIT_MODE})" >&2; exit 1 ;;
esac

# Validate; the mode -> {readonly, write-misses, strict} mapping lives in
# slurm/shared/image_token_cache_env.sh (single source of truth).
case "${IMAGE_TOKEN_CACHE_MODE}" in
  fill|readonly) ;;
  *) echo "--image-token-cache-mode must be fill or readonly (got: ${IMAGE_TOKEN_CACHE_MODE})" >&2; exit 1 ;;
esac

if [[ -n "${DATA_RAW}" ]]; then
  DATASETS="$(resolve_list "${DATA_RAW}")"
else
  # A suite is a file in task_suites/VLMEvalKit/. Hyphens in the CLI flag map to underscores in
  # the filename so e.g. `--suite llm-judge` -> suites/llm_judge.txt.
  [[ "${SUITE}" =~ ^[a-z][a-z0-9_-]*$ ]] \
    || { echo "--suite must be a simple lowercase name (got: ${SUITE})" >&2; exit 1; }
  suite_file="${REPO_ROOT}/task_suites/VLMEvalKit/${SUITE//-/_}.txt"
  if [[ ! -f "${suite_file}" ]]; then
    available="$(cd "${REPO_ROOT}/task_suites/VLMEvalKit" && ls *.txt 2>/dev/null | sed 's/\.txt$//' | tr '_' '-' | paste -sd, -)"
    echo "--suite must be one of: ${available} (got: ${SUITE})" >&2
    exit 1
  fi
  DATASETS="$(resolve_list "@${suite_file}")"
fi

MODELS="$(resolve_list "${MODELS_RAW}")"
[[ -n "${DATASETS}" ]] || { echo "no datasets resolved" >&2; exit 1; }

# Derived after flag parsing so --response-cache moves the image-token cache
# with it (a fresh-inference canary must not replay prod VQ frames).
IMAGE_TOKEN_CACHE_BASE="${IMAGE_TOKEN_CACHE_BASE:-${RESPONSE_CACHE}/image_token_cache}"

# Image-token caching and the job-side Apertus registry override apply only to
# Apertus-native models. Classified per model at submit time (jobs are per
# model x dataset, so a mixed list must not share one classification); the
# orchestrator's explicit APERTUS_MODEL_PATH marks a checkpoint-path run as
# native. Explicit FOREIGN_MODEL / --enable-image-token-cache always win.
USER_FOREIGN_MODEL="${FOREIGN_MODEL:-}"
classify_foreign() {
  [[ -n "${USER_FOREIGN_MODEL}" ]] && { echo "${USER_FOREIGN_MODEL}"; return; }
  [[ -n "${APERTUS_MODEL_PATH:-}" ]] && { echo 0; return; }
  case "$1" in [Aa]pertus*) echo 0 ;; *) echo 1 ;; esac
}

# Judge datasets score via an OpenAI-compatible judge; without a key VLMEvalKit
# silently falls back to regex parsing and produces wrong-looking-real numbers.
# Fail loud instead (ALLOW_NO_JUDGE=1 to override).
JUDGE_SUITE="${REPO_ROOT}/task_suites/VLMEvalKit/llm_judge.txt"
if [[ -f "${JUDGE_SUITE}" && "${ALLOW_NO_JUDGE:-0}" != "1" ]]; then
  JUDGE_HITS="$(comm -12 <(echo "${DATASETS}" | sort -u) <(resolve_list "@${JUDGE_SUITE}" | sort -u) || true)"
  if [[ -n "${JUDGE_HITS}" && -z "${OPENAI_API_KEY:-}" ]] && ! grep -qs '^OPENAI_API_KEY=' "${REPO_DIR}/.env"; then
    echo "ERROR: judge dataset(s) [$(echo "${JUDGE_HITS}" | tr '\n' ' ')] need OPENAI_API_KEY" >&2
    echo "       (env var or ${REPO_DIR}/.env). Set ALLOW_NO_JUDGE=1 to run anyway" >&2
    echo "       with regex-fallback scoring." >&2
    exit 1
  fi
fi
[[ -n "${MODELS}" ]] || { echo "no models resolved" >&2; exit 1; }
[[ -f "${SLURM_TEMPLATE}" ]] || { echo "slurm template not found: ${SLURM_TEMPLATE}" >&2; exit 1; }

# sbatch from inside an existing Pyxis container can inherit SPANK variables that
# conflict with a new --environment. Match the lmms-eval submission wrapper.
while IFS='=' read -r key _; do
  [[ "${key}" == SLURM_SPANK* ]] && unset "${key}"
done < <(env)
export LD_LIBRARY_PATH="/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_DIR}" "${RESPONSE_CACHE}" "${LMU_DATA}" "${WORK_BASE}" "${RUNTIME_CACHE}"
cd "${REPO_ROOT}"

echo "========================================"
echo "Apertus VLMEvalKit submit"
echo "  repo:           ${REPO_DIR}"
echo "  slurm template: ${SLURM_TEMPLATE}"
echo "  models:         $(echo "${MODELS}" | tr '\n' ',' | sed 's/,$//')"
echo "  datasets:       $(echo "${DATASETS}" | tr '\n' ',' | sed 's/,$//')"
echo "  mode:           ${MODE}"
echo "  nodes:          ${NODES}"
echo "  dp workers:     ${NUM_PROCESSES} per node (world_size = ${NODES} * ${NUM_PROCESSES})"
echo "  batch size:     ${BATCH_SIZE}"
echo "  response cache: ${RESPONSE_CACHE}"
echo "  image cache:    ${ENABLE_IMAGE_TOKEN_CACHE:-<per-model>} ${IMAGE_TOKEN_CACHE_MODE} (${IMAGE_TOKEN_CACHE_BASE})"
echo "  foreign model:  ${USER_FOREIGN_MODEL:-per-model}"
echo "  LMUData:        ${LMU_DATA}"
echo "  work base:      ${WORK_BASE}"
echo "  run id:         ${RUN_ID}"
echo "  logs:           ${LOG_DIR}"
echo "========================================"

while IFS= read -r DATASET; do
  [[ -z "${DATASET}" ]] && continue
  DATA_SLUG="$(safe_name "${DATASET}")"

  while IFS= read -r MODEL; do
    [[ -z "${MODEL}" ]] && continue
    MODEL_SLUG="$(safe_name "$(basename "${MODEL}")")"
    MODEL_FOREIGN="$(classify_foreign "${MODEL}")"
    if [[ "${MODEL_FOREIGN}" == "1" ]]; then
      MODEL_IMAGE_TOKEN_CACHE="${ENABLE_IMAGE_TOKEN_CACHE:-false}"
    else
      MODEL_IMAGE_TOKEN_CACHE="${ENABLE_IMAGE_TOKEN_CACHE:-true}"
    fi
    # Preflight (suite/preflight.py): refuse to submit what cannot succeed.
    TASK_MAX_MODEL_LEN="$(PYTHONPATH="${REPO_ROOT}" python3 -m suite.tasks --framework VLMEvalKit --max-model-len "${DATASET}")"
    if [[ "${TASK_MAX_MODEL_LEN}" -gt "${USER_APERTUS_MAX_MODEL_LEN:-131072}" ]]; then
      echo "    context: ${DATASET} needs max_model_len ${TASK_MAX_MODEL_LEN}; overriding"
      export APERTUS_MAX_MODEL_LEN="${TASK_MAX_MODEL_LEN}"
    elif [[ -n "${USER_APERTUS_MAX_MODEL_LEN+x}" ]]; then
      export APERTUS_MAX_MODEL_LEN="${USER_APERTUS_MAX_MODEL_LEN}"
    else
      unset APERTUS_MAX_MODEL_LEN
    fi
    PREFLIGHT_ARGS=(--tasks "${DATASET}" --container-image "${SUITE_CONTAINER_IMAGE:-}"
                    --max-model-len "${APERTUS_MAX_MODEL_LEN:-131072}")
    if [[ "${MODEL_FOREIGN}" == "1" ]]; then
      PREFLIGHT_ARGS+=(--model "${MODEL}" --skip-model)
    else
      PREFLIGHT_ARGS+=(--model "${APERTUS_MODEL_PATH:-${MODEL}}" --tokenizer "${APERTUS_TOKENIZER_PATH:-${DEFAULT_APERTUS_TOKENIZER}}"
                       --vision-tokenizer "${RUNTIME_CACHE}/models/BAAI/Emu3.5-VisionTokenizer")
      [[ -n "${APERTUS_ENABLE_THINKING:-}" ]] && PREFLIGHT_ARGS+=(--thinking)
    fi
    preflight_or_die VLMEvalKit "${MODEL}:${DATASET}" "${PREFLIGHT_ARGS[@]}"
    JOB_NAME="vlmeval-${DATA_SLUG}"
    WORK_DIR="${WORK_BASE}/${MODEL_SLUG}/${DATA_SLUG}"
    if [[ "${SUBMIT_MODE}" == "interactive" ]]; then
      JOB_OUTPUT="${LOG_DIR}/${JOB_NAME}_${MODEL_SLUG}_interactive.out"
      JOB_ERROR="${LOG_DIR}/${JOB_NAME}_${MODEL_SLUG}_interactive.err"
    else
      JOB_OUTPUT="${LOG_DIR}/${JOB_NAME}_${MODEL_SLUG}_%j.out"
      JOB_ERROR="${LOG_DIR}/${JOB_NAME}_${MODEL_SLUG}_%j.err"
    fi

    JOB_ARGS=(
      --repo-dir "${REPO_DIR}"
      --model "${MODEL}"
      --data "${DATASET}"
      --mode "${MODE}"
      --work-dir "${WORK_DIR}"
      --response-cache "${RESPONSE_CACHE}"
      --enable-image-token-cache "${MODEL_IMAGE_TOKEN_CACHE}"
      --image-token-cache-mode "${IMAGE_TOKEN_CACHE_MODE}"
      --image-token-cache-base "${IMAGE_TOKEN_CACHE_BASE}"
      --lmu-data "${LMU_DATA}"
      --runtime-cache "${RUNTIME_CACHE}"
      --num-processes "${NUM_PROCESSES}"
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --batch-size "${BATCH_SIZE}"
      --main-process-port "${MAIN_PROCESS_PORT}"
    )

    if [[ "${SUBMIT_MODE}" == "interactive" ]]; then
      CMD=(bash "${SLURM_TEMPLATE}" "${JOB_ARGS[@]}")
    else
      CMD=(
        sbatch
        "${SBATCH_OVERRIDES[@]}"
        --job-name "${JOB_NAME}"
        --output "${JOB_OUTPUT}"
        --error "${JOB_ERROR}"
        --time "${SBATCH_TIME}"
        --nodes "${NODES}"
        "${SLURM_TEMPLATE}"
        "${JOB_ARGS[@]}"
      )
    fi

    export FOREIGN_MODEL="${MODEL_FOREIGN}"
    export SUITE_JOB_OUTPUT="${JOB_OUTPUT}" SUITE_JOB_ERROR="${JOB_ERROR}"
    echo "--- submit: data=${DATASET} model=${MODEL} foreign=${MODEL_FOREIGN} work=${WORK_DIR} ---"
    echo "    logs:   ${JOB_OUTPUT} / ${JOB_ERROR}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf ' %q' "${CMD[@]}"
      if [[ "${SUBMIT_MODE}" == "interactive" ]]; then
        printf ' >%q 2>%q\n' "${JOB_OUTPUT}" "${JOB_ERROR}"
      else
        printf '\n'
      fi
    else
      if [[ "${SUBMIT_MODE}" == "interactive" ]]; then
        "${CMD[@]}" >"${JOB_OUTPUT}" 2>"${JOB_ERROR}"
      else
        "${CMD[@]}"
      fi
    fi
  done <<< "${MODELS}"
done <<< "${DATASETS}"

echo "========================================"
echo "Submissions complete. Logs: ${LOG_DIR}"
echo "========================================"
