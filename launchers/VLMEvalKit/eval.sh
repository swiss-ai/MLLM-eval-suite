#!/usr/bin/env bash
# Submit Apertus VLMEvalKit jobs with the dedicated Apertus vLLM runtime.
#
# Examples:
#   bash launchers/VLMEvalKit/eval.sh
#   bash launchers/VLMEvalKit/eval.sh --data 3DSRBench --model Apertus-1p5-8B
#   bash launchers/VLMEvalKit/eval.sh --data 3DSRBench --mode infer
#   bash launchers/VLMEvalKit/eval.sh --data @datasets.txt --model @models.txt
#   bash launchers/VLMEvalKit/eval.sh --data 3DSRBench --model Apertus-1p5-8B --submit-mode interactive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${ORCH_REPO_ROOT:-}" ]]; then
  echo "ORCH_REPO_ROOT must be set by the orchestration launcher." >&2
  exit 2
fi
REPO_ROOT="${ORCH_REPO_ROOT}"
REPO_DIR="${REPO_DIR:-${REPO_ROOT}/third_party/VLMEvalKit}"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-${REPO_ROOT}/slurm/VLMEvalKit/eval_job.slurm}"

RESPONSE_CACHE="${VLMEVAL_RESPONSE_CACHE:-${REPO_ROOT}/cache/VLMEvalKit}"
IMAGE_TOKEN_CACHE_BASE="${IMAGE_TOKEN_CACHE_BASE:-${RESPONSE_CACHE}/image_token_cache}"
LMU_DATA="${LMUData:-${REPO_ROOT}/cache/VLMEvalKit/LMUData}"
WORK_BASE="${WORK_BASE:-${REPO_ROOT}/results/VLMEvalKit}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
WORK_BASE="${WORK_BASE}/${RUN_ID}"
RUNTIME_CACHE="${RUNTIME_CACHE:-${REPO_ROOT}/cache}"
LOG_BASE="${LOG_DIR:-${REPO_ROOT}/logs/VLMEvalKit}"
LOG_DIR="${LOG_BASE}/${RUN_ID}"

DEFAULT_MODEL="Apertus-1p5-8B"

MODELS_RAW="${DEFAULT_MODEL}"
DATA_RAW=""
SUITE="smoke"
MODE="all"
SUBMIT_MODE="batch"
NODES="${NODES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ENABLE_IMAGE_TOKEN_CACHE="${ENABLE_IMAGE_TOKEN_CACHE:-true}"
IMAGE_TOKEN_CACHE_MODE="${IMAGE_TOKEN_CACHE_MODE:-fill}"
IMAGE_TOKEN_CACHE_LOCAL_COPY="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_LOCAL_COPY:-0}"
IMAGE_TOKEN_CACHE_PRELOAD="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_PRELOAD:-1}"
IMAGE_TOKEN_CACHE_READONLY="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY:-0}"
IMAGE_TOKEN_CACHE_WRITE_MISSES="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES:-1}"
IMAGE_TOKEN_CACHE_COLLISION_GUARD="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD:-0}"
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
  --batch-size <int>                Batch size value passed through/logged for the framework. Default: 512.
  --work-base <path>                Root for VLMEvalKit outputs.
  --response-cache <path>           SQLite response cache root.
  --image-token-cache-base <path>   Apertus image-token cache base. Default: response-cache/image_token_cache.
  --enable-image-token-cache <true|false>
                                    Export Apertus vLLM image-token cache env. Default: true.
  --image-token-cache-mode <fill|readonly>
                                    Accepted for compatibility. Both modes preload the
                                    shared cache and write missing image tokens.
                                    Default: fill.
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

case "${MODE}" in
  all|infer|eval) ;;
  *) echo "--mode must be all, infer, or eval (got: ${MODE})" >&2; exit 1 ;;
esac

case "${SUBMIT_MODE}" in
  batch|interactive) ;;
  *) echo "--submit-mode must be batch or interactive (got: ${SUBMIT_MODE})" >&2; exit 1 ;;
esac

# Validate; the mode -> {preload, readonly, write-misses} mapping lives in
# eval_job.slurm (single source of truth).
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
echo "  image cache:    ${ENABLE_IMAGE_TOKEN_CACHE} ${IMAGE_TOKEN_CACHE_MODE} (${IMAGE_TOKEN_CACHE_BASE})"
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
    MODEL_SLUG="$(safe_name "${MODEL}")"
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
      --enable-image-token-cache "${ENABLE_IMAGE_TOKEN_CACHE}"
      --image-token-cache-mode "${IMAGE_TOKEN_CACHE_MODE}"
      --image-token-cache-base "${IMAGE_TOKEN_CACHE_BASE}"
      --image-token-cache-collision-guard "${IMAGE_TOKEN_CACHE_COLLISION_GUARD}"
      --image-token-cache-local-copy "${IMAGE_TOKEN_CACHE_LOCAL_COPY}"
      --image-token-cache-preload "${IMAGE_TOKEN_CACHE_PRELOAD}"
      --image-token-cache-readonly "${IMAGE_TOKEN_CACHE_READONLY}"
      --image-token-cache-write-misses "${IMAGE_TOKEN_CACHE_WRITE_MISSES}"
      --lmu-data "${LMU_DATA}"
      --runtime-cache "${RUNTIME_CACHE}"
      --num-processes "${NUM_PROCESSES}"
      --batch-size "${BATCH_SIZE}"
      --main-process-port "${MAIN_PROCESS_PORT}"
    )

    if [[ "${SUBMIT_MODE}" == "interactive" ]]; then
      CMD=(bash "${SLURM_TEMPLATE}" "${JOB_ARGS[@]}")
    else
      CMD=(
        sbatch
        --job-name "${JOB_NAME}"
        --output "${JOB_OUTPUT}"
        --error "${JOB_ERROR}"
        --time "${SBATCH_TIME}"
        --nodes "${NODES}"
        "${SLURM_TEMPLATE}"
        "${JOB_ARGS[@]}"
      )
    fi

    echo "--- submit: data=${DATASET} model=${MODEL} work=${WORK_DIR} ---"
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
