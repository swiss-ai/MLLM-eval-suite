#!/usr/bin/env bash
# Submit NeMo Evaluator text benchmarks with Apertus served by managed vLLM.
#
# This is the outer submitter. It resolves suites, writes one Evaluator
# template/env pair per task, and launches slurm/Evaluator/eval_job.slurm. The
# Slurm script is the final entrypoint that runs `nel eval run` inside the
# selected image.

set -euo pipefail

if [[ -z "${ORCH_REPO_ROOT:-}" ]]; then
  echo "ORCH_REPO_ROOT must be set by the orchestration launcher." >&2
  exit 2
fi

REPO_ROOT="${ORCH_REPO_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=launchers/Evaluator/config_env.sh
source "${SCRIPT_DIR}/config_env.sh"
REPO_DIR="${EVALUATOR_REPO_DIR:-${REPO_ROOT}/third_party/Evaluator}"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-${REPO_ROOT}/slurm/Evaluator/eval_job.slurm}"
CONFIG_TEMPLATE="${EVALUATOR_CONFIG_TEMPLATE:-${REPO_ROOT}/configs/Evaluator/apertus_text_vllm_template.yaml}"
CONFIG_TEMPLATE_EXPLICIT=0
CONFIG_TEMPLATE_DIR="${EVALUATOR_CONFIG_TEMPLATE_DIR:-${REPO_ROOT}/configs/Evaluator/templates}"
SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/task_suites/Evaluator}"
OUTPUT_BASE="${OUTPUT_PATH:-${REPO_ROOT}/results/Evaluator}"
LOG_BASE="${LOG_DIR:-${REPO_ROOT}/logs/Evaluator}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
RUN_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_ID}"
RUN_LOG_DIR="${LOG_BASE}/${RUN_ID}"

DEFAULT_MODEL="/capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus-1p5-8B-sft-capfilter-lr6e-5-constant-innovator-fix-it23409"
DEFAULT_TOKENIZER="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"

MODEL_PATH="${NEL_APERTUS_MODEL:-${DEFAULT_MODEL}}"
TOKENIZER_PATH="${NEL_APERTUS_TOKENIZER:-${TOKENIZER_PATH:-${DEFAULT_TOKENIZER}}}"
CHAT_TEMPLATE="${NEL_APERTUS_CHAT_TEMPLATE:-${CHAT_TEMPLATE:-}}"
TASKS_RAW=""
SUITE="smoke"
SUBMIT_MODE="batch"
RUNNER_DEFAULT="evaluator"
PORT="${NEL_APERTUS_PORT:-8000}"
SERVED_MODEL_NAME="${NEL_APERTUS_SERVED_MODEL_NAME:-Apertus-1p5-8B}"
SERVICE_TYPE="${NEL_SERVICE_TYPE:-vllm}"
SERVICE_PROTOCOL="${NEL_APERTUS_PROTOCOL:-chat_completions}"
TP_SIZE="${NEL_APERTUS_TP:-1}"
PP_SIZE="${NEL_APERTUS_PP:-}"
DP_SIZE="${NEL_APERTUS_DP:-}"
NUM_NODES="${NEL_APERTUS_NUM_NODES:-1}"
STARTUP_TIMEOUT="${NEL_APERTUS_STARTUP_TIMEOUT:-1800}"
MAX_PROBLEMS="${NEL_EVALUATOR_MAX_PROBLEMS:-5}"
MAX_CONCURRENT="${NEL_EVALUATOR_MAX_CONCURRENT:-4}"
REPEATS="${NEL_BENCHMARK_REPEATS:-1}"
BENCHMARK_TIMEOUT="${NEL_BENCHMARK_TIMEOUT:-1800}"
BENCHMARK_FEWSHOT="${NEL_BENCHMARK_FEWSHOT:-}"
BENCHMARK_CONTEXT_WINDOW="${NEL_BENCHMARK_CONTEXT_WINDOW:-}"
BENCHMARK_SKIP_FAILED="${NEL_BENCHMARK_SKIP_FAILED:-false}"
BENCHMARK_MAX_SYSTEM_RETRIES="${NEL_BENCHMARK_MAX_SYSTEM_RETRIES:-3}"
BENCHMARK_INSTRUCTION_TEMPLATE="${NEL_BENCHMARK_INSTRUCTION_TEMPLATE:-}"
BENCHMARK_VERIFIER="${NEL_BENCHMARK_VERIFIER:-}"
BENCHMARK_SHUFFLE_SEED="${NEL_BENCHMARK_SHUFFLE_SEED:-42}"
BENCHMARK_SANDBOX="${NEL_BENCHMARK_SANDBOX:-}"
BENCHMARK_PARAMS="${NEL_BENCHMARK_PARAMS:-{}}"
PROGRESS_INTERVAL="${NEL_PROGRESS_INTERVAL:-60}"
MAX_TOKENS="${NEL_APERTUS_MAX_TOKENS:-256}"
TEMPERATURE="${NEL_APERTUS_TEMPERATURE:-0}"
TOP_P="${NEL_APERTUS_TOP_P:-1}"
SEED="${NEL_APERTUS_SEED:-1234}"
STOP="${NEL_APERTUS_STOP:-}"
FREQUENCY_PENALTY="${NEL_APERTUS_FREQUENCY_PENALTY:-}"
PRESENCE_PENALTY="${NEL_APERTUS_PRESENCE_PENALTY:-}"
GPU_MEMORY_UTILIZATION="${NEL_APERTUS_GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${NEL_APERTUS_MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${NEL_APERTUS_MAX_NUM_BATCHED_TOKENS:-49152}"
HF_OVERRIDES="${NEL_APERTUS_HF_OVERRIDES:-{\"max_position_embeddings\":131072}}"
MODELS_CACHE="${NEL_APERTUS_MODELS_CACHE:-${REPO_ROOT}/cache/models}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_RESERVATION="${SBATCH_RESERVATION:-}"
SBATCH_ENVIRONMENT="${SBATCH_ENVIRONMENT:-}"
SBATCH_NODES="${SBATCH_NODES:-}"
SBATCH_NTASKS_PER_NODE="${SBATCH_NTASKS_PER_NODE:-}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-}"
SBATCH_GPUS_PER_NODE="${SBATCH_GPUS_PER_NODE:-}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"
SOLVER_TYPE="${NEL_SOLVER_TYPE:-simple}"
SOLVER_SERVICE="${NEL_SOLVER_SERVICE:-apertus}"
SOLVER_SYSTEM_PROMPT="${NEL_SOLVER_SYSTEM_PROMPT:-}"
SOLVER_TEMPERATURE="${NEL_SOLVER_TEMPERATURE:-}"
SOLVER_TOP_P="${NEL_SOLVER_TOP_P:-}"
SOLVER_MAX_TOKENS="${NEL_SOLVER_MAX_TOKENS:-}"
SOLVER_SEED="${NEL_SOLVER_SEED:-}"
SOLVER_STOP="${NEL_SOLVER_STOP:-}"
SOLVER_FREQUENCY_PENALTY="${NEL_SOLVER_FREQUENCY_PENALTY:-}"
SOLVER_PRESENCE_PENALTY="${NEL_SOLVER_PRESENCE_PENALTY:-}"
SCORING_INCLUDE_DEFAULTS="${NEL_SCORING_INCLUDE_DEFAULTS:-true}"
SCORING_METRICS="${NEL_SCORING_METRICS:-[]}"
SCORING_PRIMARY="${NEL_SCORING_PRIMARY:-}"
OUTPUT_TIMESTAMPED="${NEL_OUTPUT_TIMESTAMPED:-false}"
OUTPUT_EXPORT="${NEL_OUTPUT_EXPORT:-[]}"
OUTPUT_EXPORT_CONFIG="${NEL_OUTPUT_EXPORT_CONFIG:-{}}"
HARBOR_AGENT="${NEL_HARBOR_AGENT:-}"
HARBOR_AGENT_KWARGS="${NEL_HARBOR_AGENT_KWARGS:-{}}"
HARBOR_CONTAINER_ENV="${NEL_HARBOR_CONTAINER_ENV:-{}}"
HARBOR_RUN_TIMEOUT="${NEL_HARBOR_RUN_TIMEOUT:-}"
HARBOR_CMD_TIMEOUT="${NEL_HARBOR_CMD_TIMEOUT:-}"
HARBOR_TIMEOUT_STRATEGY="${NEL_HARBOR_TIMEOUT_STRATEGY:-}"
HARBOR_MAX_AGENT_TIMEOUT="${NEL_HARBOR_MAX_AGENT_TIMEOUT:-}"
HARBOR_SKILL="${NEL_HARBOR_SKILL:-}"
HARBOR_SKILL_DIR="${NEL_HARBOR_SKILL_DIR:-}"
OPENCLAW_THINKING="${NEL_OPENCLAW_THINKING:-}"
OPENCLAW_CONTEXT_WINDOW="${NEL_OPENCLAW_CONTEXT_WINDOW:-}"
OPENCLAW_MAX_CONCURRENT="${NEL_OPENCLAW_MAX_CONCURRENT:-}"
OPENCLAW_IDLE_TIMEOUT_SECONDS="${NEL_OPENCLAW_IDLE_TIMEOUT_SECONDS:-}"
OPENCLAW_RUN_TIMEOUT="${NEL_OPENCLAW_RUN_TIMEOUT:-}"
OPENCLAW_TIMEOUT_STRATEGY="${NEL_OPENCLAW_TIMEOUT_STRATEGY:-}"
OPENCLAW_MAX_AGENT_TIMEOUT="${NEL_OPENCLAW_MAX_AGENT_TIMEOUT:-}"
OPENCLAW_WEB_SEARCH_PROVIDER="${NEL_OPENCLAW_WEB_SEARCH_PROVIDER:-}"
OPENCLAW_CONFIG_PATH="${NEL_OPENCLAW_CONFIG_PATH:-}"
OPENCLAW_SKIP_PREFLIGHT="${NEL_OPENCLAW_SKIP_PREFLIGHT:-}"
OPENCLAW_BIN="${NEL_OPENCLAW_BIN:-}"
CONFIG_JSON="${NEL_CONFIG_JSON:-}"
CONFIG_JSON_FILE="${NEL_CONFIG_JSON_FILE:-}"
DRY_RUN=0
NEL_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  launchers/Evaluator/eval.sh [options] [-- nel eval run args...]

Options:
  --model <path>                 Apertus model path. Defaults to the shared Apertus checkpoint.
  --tasks <csv|@file|file>       Benchmark names accepted by NeMo Evaluator.
  --suite <name>                 Suite in task_suites/Evaluator/. Default: smoke.
                                 Hyphens map to underscores.
  --submit-mode <batch|interactive>
                                 batch submits one Slurm job per task; interactive
                                 runs the Slurm script with bash in this allocation.
  --output-dir <path>            Shared run output root. Default: results/Evaluator/$RUN_ID.
  --config-template <path>       Use one explicit Evaluator YAML template for every task.
  --template-dir <path>          Directory of per-benchmark templates. Default: configs/Evaluator/templates.
  --tokenizer-path <path>        Apertus tokenizer path.
  --chat-template <path>         Chat template path.
  --service-type <name>          Evaluator service type. Default: vllm.
  --service-protocol <name>      Evaluator protocol. Default: chat_completions.
  --served-model-name <name>     Served model name. Default: Apertus-1p5-8B.
  --port <int>                   vLLM OpenAI server port. Default: 8000.
  --tp-size <int>                vLLM tensor parallel size. Default: 1.
  --pp-size <int>                vLLM pipeline parallel size.
  --dp-size <int>                vLLM data parallel size.
  --num-nodes <int>              Managed service node count. Default: 1.
  --time <hh:mm:ss>              Slurm walltime for batch mode. Default: 04:00:00.
  --account <name>               Slurm account override for sbatch.
  --partition <name>             Slurm partition override for sbatch.
  --reservation <name>           Slurm reservation override for sbatch.
  --environment <path>           Slurm/Pyxis environment TOML override for sbatch.
  --nodes <int>                  Slurm node count override for sbatch.
  --ntasks-per-node <int>        Slurm tasks-per-node override for sbatch.
  --cpus-per-task <int>          Slurm CPUs-per-task override for sbatch.
  --gpus-per-node <value>        Slurm GPUs-per-node override for sbatch.
  --constraint <value>           Slurm constraint override for sbatch.
  --startup-timeout <seconds>    vLLM health wait timeout. Default: 1800.
  --max-problems <int>           Per-benchmark smoke limit. Default: 5.
  --max-concurrent <int>         Evaluator concurrency per benchmark. Default: 4.
  --repeats <int>                Evaluator repeats. Default: 1.
  --benchmark-timeout <seconds>  Evaluator benchmark timeout. Default: 1800.
  --fewshot <int>                Benchmark few-shot setting.
  --context-window <int>         Benchmark context window.
  --skip-failed <bool>           Continue past failed samples. Default: false.
  --max-system-retries <int>     Per-sample system retries. Default: 3.
  --instruction-template <text>  Benchmark instruction template.
  --verifier <name>              Benchmark verifier name.
  --shuffle-seed <int|null>      Benchmark shuffle seed. Default: 42.
  --benchmark-sandbox <yaml>     Inline sandbox mapping, e.g. '{type: docker, image: python:3.12-slim}'.
  --benchmark-params <yaml>      Inline params mapping forwarded to the benchmark env.
  --progress-interval <seconds>  Evaluator progress log interval. Default: 60.
  --max-tokens <int>             Generation max tokens. Default: 256.
  --temperature <float>          Generation temperature. Default: 0.
  --top-p <float>                Generation top_p. Default: 1.
  --seed <int>                   Generation seed. Default: 1234.
  --stop <yaml>                  Service stop sequence list, e.g. '["\\n\\n"]'.
  --frequency-penalty <float>    Service frequency penalty.
  --presence-penalty <float>     Service presence penalty.
  --solver-type <name>           Evaluator solver type. Default: simple.
  --solver-service <name>        Evaluator solver service. Default: apertus.
  --solver-system-prompt <text>  Optional system prompt for simple solver.
  --solver-temperature <float>   Solver-level generation temperature override.
  --solver-top-p <float>         Solver-level top_p override.
  --solver-max-tokens <int>      Solver-level max token override.
  --solver-seed <int>            Solver-level seed override.
  --scoring-include-defaults <bool>
                                 Whether to include benchmark default scoring.
  --scoring-metrics <yaml>       Inline scoring metric list.
  --scoring-primary <name>       Primary metric when multiple metrics are configured.
  --gpu-memory-utilization <f>   vLLM GPU memory utilization. Default: 0.6.
  --max-model-len <int>          vLLM max model length. Default: 131072.
  --max-num-batched-tokens <int> vLLM max batched tokens. Default: 49152.
  --harbor-agent <name>          Agent name for Harbor templates.
  --harbor-agent-kwargs <yaml>   Inline Harbor agent_kwargs mapping.
  --openclaw-thinking <level>    OpenClaw thinking setting for PinchBench.
  --config-json <json>           JSON overrides for template variables.
  --config-file <path>           JSON override file. Supports shared values and
                                 per-benchmark values under "benchmarks".
  --extra-framework-config <arg> Extra nel eval run argv token. Repeat for each token.
  --dry-run                      Write configs/print commands without executing.
  -h, --help                     Show this help.
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

shell_quote() {
  printf "%q" "$1"
}

task_payload() {
  local task="$1"
  printf '%s' "${task}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --tasks|--data) TASKS_RAW="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --mode) shift 2 ;; # Reserved for dispatcher compatibility.
    --submit-mode) SUBMIT_MODE="$2"; shift 2 ;;
    --output-dir|--output-path) RUN_OUTPUT_DIR="$2"; shift 2 ;;
    --log-dir) RUN_LOG_DIR="$2"; shift 2 ;;
    --config-template) CONFIG_TEMPLATE="$2"; CONFIG_TEMPLATE_EXPLICIT=1; shift 2 ;;
    --template-dir|--config-template-dir) CONFIG_TEMPLATE_DIR="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --chat-template) CHAT_TEMPLATE="$2"; shift 2 ;;
    --service-type) SERVICE_TYPE="$2"; shift 2 ;;
    --service-protocol|--protocol) SERVICE_PROTOCOL="$2"; shift 2 ;;
    --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tp-size|--tensor-parallel-size) TP_SIZE="$2"; shift 2 ;;
    --pp-size|--pipeline-parallel-size) PP_SIZE="$2"; shift 2 ;;
    --dp-size|--data-parallel-size) DP_SIZE="$2"; shift 2 ;;
    --num-nodes) NUM_NODES="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --account|--sbatch-account) SBATCH_ACCOUNT="$2"; shift 2 ;;
    --partition|--sbatch-partition) SBATCH_PARTITION="$2"; shift 2 ;;
    --reservation|--sbatch-reservation) SBATCH_RESERVATION="$2"; shift 2 ;;
    --environment|--sbatch-environment) SBATCH_ENVIRONMENT="$2"; shift 2 ;;
    --nodes|--sbatch-nodes) SBATCH_NODES="$2"; shift 2 ;;
    --ntasks-per-node|--sbatch-ntasks-per-node) SBATCH_NTASKS_PER_NODE="$2"; shift 2 ;;
    --cpus-per-task|--sbatch-cpus-per-task) SBATCH_CPUS_PER_TASK="$2"; shift 2 ;;
    --gpus-per-node|--sbatch-gpus-per-node) SBATCH_GPUS_PER_NODE="$2"; shift 2 ;;
    --constraint|--sbatch-constraint) SBATCH_CONSTRAINT="$2"; shift 2 ;;
    --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
    --max-problems) MAX_PROBLEMS="$2"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --benchmark-timeout) BENCHMARK_TIMEOUT="$2"; shift 2 ;;
    --fewshot) BENCHMARK_FEWSHOT="$2"; shift 2 ;;
    --context-window) BENCHMARK_CONTEXT_WINDOW="$2"; shift 2 ;;
    --skip-failed) BENCHMARK_SKIP_FAILED="$2"; shift 2 ;;
    --max-system-retries) BENCHMARK_MAX_SYSTEM_RETRIES="$2"; shift 2 ;;
    --instruction-template) BENCHMARK_INSTRUCTION_TEMPLATE="$2"; shift 2 ;;
    --verifier) BENCHMARK_VERIFIER="$2"; shift 2 ;;
    --shuffle-seed) BENCHMARK_SHUFFLE_SEED="$2"; shift 2 ;;
    --benchmark-sandbox|--sandbox) BENCHMARK_SANDBOX="$2"; shift 2 ;;
    --benchmark-params|--params) BENCHMARK_PARAMS="$2"; shift 2 ;;
    --progress-interval) PROGRESS_INTERVAL="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --stop) STOP="$2"; shift 2 ;;
    --frequency-penalty) FREQUENCY_PENALTY="$2"; shift 2 ;;
    --presence-penalty) PRESENCE_PENALTY="$2"; shift 2 ;;
    --solver-type) SOLVER_TYPE="$2"; shift 2 ;;
    --solver-service) SOLVER_SERVICE="$2"; shift 2 ;;
    --solver-system-prompt) SOLVER_SYSTEM_PROMPT="$2"; shift 2 ;;
    --solver-temperature) SOLVER_TEMPERATURE="$2"; shift 2 ;;
    --solver-top-p) SOLVER_TOP_P="$2"; shift 2 ;;
    --solver-max-tokens) SOLVER_MAX_TOKENS="$2"; shift 2 ;;
    --solver-seed) SOLVER_SEED="$2"; shift 2 ;;
    --solver-stop) SOLVER_STOP="$2"; shift 2 ;;
    --solver-frequency-penalty) SOLVER_FREQUENCY_PENALTY="$2"; shift 2 ;;
    --solver-presence-penalty) SOLVER_PRESENCE_PENALTY="$2"; shift 2 ;;
    --scoring-include-defaults) SCORING_INCLUDE_DEFAULTS="$2"; shift 2 ;;
    --scoring-metrics) SCORING_METRICS="$2"; shift 2 ;;
    --scoring-primary) SCORING_PRIMARY="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --hf-overrides) HF_OVERRIDES="$2"; shift 2 ;;
    --models-cache) MODELS_CACHE="$2"; shift 2 ;;
    --output-timestamped) OUTPUT_TIMESTAMPED="$2"; shift 2 ;;
    --output-export) OUTPUT_EXPORT="$2"; shift 2 ;;
    --output-export-config) OUTPUT_EXPORT_CONFIG="$2"; shift 2 ;;
    --harbor-agent) HARBOR_AGENT="$2"; shift 2 ;;
    --harbor-agent-kwargs) HARBOR_AGENT_KWARGS="$2"; shift 2 ;;
    --harbor-container-env) HARBOR_CONTAINER_ENV="$2"; shift 2 ;;
    --harbor-run-timeout) HARBOR_RUN_TIMEOUT="$2"; shift 2 ;;
    --harbor-cmd-timeout) HARBOR_CMD_TIMEOUT="$2"; shift 2 ;;
    --harbor-timeout-strategy) HARBOR_TIMEOUT_STRATEGY="$2"; shift 2 ;;
    --harbor-max-agent-timeout) HARBOR_MAX_AGENT_TIMEOUT="$2"; shift 2 ;;
    --harbor-skill) HARBOR_SKILL="$2"; shift 2 ;;
    --harbor-skill-dir) HARBOR_SKILL_DIR="$2"; shift 2 ;;
    --openclaw-thinking) OPENCLAW_THINKING="$2"; shift 2 ;;
    --openclaw-context-window) OPENCLAW_CONTEXT_WINDOW="$2"; shift 2 ;;
    --openclaw-max-concurrent) OPENCLAW_MAX_CONCURRENT="$2"; shift 2 ;;
    --openclaw-idle-timeout-seconds) OPENCLAW_IDLE_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --openclaw-run-timeout) OPENCLAW_RUN_TIMEOUT="$2"; shift 2 ;;
    --openclaw-timeout-strategy) OPENCLAW_TIMEOUT_STRATEGY="$2"; shift 2 ;;
    --openclaw-max-agent-timeout) OPENCLAW_MAX_AGENT_TIMEOUT="$2"; shift 2 ;;
    --openclaw-web-search-provider) OPENCLAW_WEB_SEARCH_PROVIDER="$2"; shift 2 ;;
    --openclaw-config-path) OPENCLAW_CONFIG_PATH="$2"; shift 2 ;;
    --openclaw-skip-preflight) OPENCLAW_SKIP_PREFLIGHT="$2"; shift 2 ;;
    --openclaw-bin) OPENCLAW_BIN="$2"; shift 2 ;;
    --config-json) CONFIG_JSON="$2"; shift 2 ;;
    --config-file|--config-json-file) CONFIG_JSON_FILE="$2"; shift 2 ;;
    --extra-framework-config|--extra-framework-arg) NEL_ARGS+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; NEL_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
      *) NEL_ARGS+=("$1"); shift ;;
    esac
  done
}

require_submit_mode() {
  case "${SUBMIT_MODE}" in
    batch|interactive) ;;
    *) echo "--submit-mode must be batch or interactive for Evaluator (got: ${SUBMIT_MODE})" >&2; exit 2 ;;
  esac
}

abs_path_from_repo() {
  local path="$1"
  case "${path}" in
    /*) printf '%s\n' "${path}" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "${path}" ;;
  esac
}

normalize_paths() {
  if [[ -z "${CHAT_TEMPLATE}" ]]; then
    CHAT_TEMPLATE="${TOKENIZER_PATH}/chat_template.jinja"
  fi

  RUN_OUTPUT_DIR="$(abs_path_from_repo "${RUN_OUTPUT_DIR}")"
  RUN_LOG_DIR="$(abs_path_from_repo "${RUN_LOG_DIR}")"
  CONFIG_TEMPLATE="$(abs_path_from_repo "${CONFIG_TEMPLATE}")"
  CONFIG_TEMPLATE_DIR="$(abs_path_from_repo "${CONFIG_TEMPLATE_DIR}")"
  if [[ -n "${CONFIG_JSON_FILE}" ]]; then
    CONFIG_JSON_FILE="$(abs_path_from_repo "${CONFIG_JSON_FILE}")"
  fi
}

resolve_tasks() {
  local suite_file available
  if [[ -n "${TASKS_RAW}" ]]; then
    TASKS="$(resolve_list "${TASKS_RAW}")"
    return
  fi

  [[ "${SUITE}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid suite name: ${SUITE}" >&2; exit 1; }
  suite_file="${SUITE_DIR}/${SUITE//-/_}.txt"
  [[ -f "${suite_file}" ]] || {
    available="$(cd "${SUITE_DIR}" && ls *.txt 2>/dev/null | sed 's/\.txt$//' | tr '_' '-' | paste -sd, -)"
    echo "--suite must be one of: ${available} (got: ${SUITE})" >&2
    exit 1
  }
  TASKS="$(resolve_list "@${suite_file}")"
}

validate_inputs() {
  mkdir -p "${RUN_OUTPUT_DIR}" "${RUN_LOG_DIR}" "${MODELS_CACHE}"
  [[ -f "${CONFIG_TEMPLATE}" ]] || { echo "config template not found: ${CONFIG_TEMPLATE}" >&2; exit 1; }
  [[ -d "${CONFIG_TEMPLATE_DIR}" ]] || { echo "config template dir not found: ${CONFIG_TEMPLATE_DIR}" >&2; exit 1; }
  [[ -z "${CONFIG_JSON_FILE}" || -f "${CONFIG_JSON_FILE}" ]] || { echo "config JSON file not found: ${CONFIG_JSON_FILE}" >&2; exit 1; }
  [[ -z "${CONFIG_JSON}" || -z "${CONFIG_JSON_FILE}" ]] || { echo "use either --config-json or --config-file, not both" >&2; exit 2; }
  [[ -n "${TASKS}" ]] || { echo "no Evaluator tasks resolved" >&2; exit 1; }
  [[ -d "${REPO_DIR}" ]] || { echo "Evaluator repo not found: ${REPO_DIR}" >&2; exit 1; }
  [[ -f "${SLURM_TEMPLATE}" ]] || { echo "slurm template not found: ${SLURM_TEMPLATE}" >&2; exit 1; }
}

prepare_runtime_env() {
  local key
  while IFS='=' read -r key _; do
    [[ "${key}" == SLURM_SPANK* ]] && unset "${key}"
  done < <(env)
  export LD_LIBRARY_PATH="/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}"
}

print_summary() {
  echo "========================================"
  echo "Apertus Evaluator submit"
  echo "  repo:        ${REPO_DIR}"
  echo "  slurm:       ${SLURM_TEMPLATE}"
  echo "  templates:   ${CONFIG_TEMPLATE_DIR}"
  echo "  mode:        ${SUBMIT_MODE}"
  echo "  tasks:       $(echo "${TASKS}" | tr '\n' ',' | sed 's/,$//')"
  echo "  run output:  ${RUN_OUTPUT_DIR}"
  echo "  run logs:    ${RUN_LOG_DIR}"
  if [[ -n "${CONFIG_JSON_FILE}" ]]; then
    echo "  config json: ${CONFIG_JSON_FILE}"
  elif [[ -n "${CONFIG_JSON}" ]]; then
    echo "  config json: <inline>"
  fi
  echo "========================================"
}

select_config_template() {
  local task_slug="$1"
  local source="${CONFIG_TEMPLATE}"
  if [[ "${CONFIG_TEMPLATE_EXPLICIT}" -eq 0 && -f "${CONFIG_TEMPLATE_DIR}/${task_slug}.yaml" ]]; then
    source="${CONFIG_TEMPLATE_DIR}/${task_slug}.yaml"
  fi
  printf '%s\n' "${source}"
}

build_job_args() {
  JOB_ARGS=(
    --runner "${RUNNER}"
    --task "${TASK_NAME}"
    --output-dir "${TASK_OUTPUT_DIR}"
    --repo-dir "${REPO_DIR}"
    --log-dir "${RUN_LOG_DIR}"
    --config "${TASK_CONFIG_TEMPLATE}"
    --env-file "${TASK_ENV}"
  )
  if [[ -n "${TASK_CONFIG_JSON}" ]]; then
    JOB_ARGS+=(--config-json-file "${TASK_CONFIG_JSON}")
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    JOB_ARGS+=(--dry-run)
  fi
  if [[ ${#NEL_ARGS[@]} -gt 0 ]]; then
    local arg
    for arg in "${NEL_ARGS[@]}"; do
      JOB_ARGS+=(--extra-framework-config "${arg}")
    done
  fi
}

build_sbatch_args() {
  SBATCH_ARGS=(
    --job-name "evaluator-${TASK_SLUG}"
    --output "${JOB_OUTPUT}"
    --error "${JOB_ERROR}"
    --time "${SBATCH_TIME}"
  )
  [[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account "${SBATCH_ACCOUNT}")
  [[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition "${SBATCH_PARTITION}")
  [[ -n "${SBATCH_RESERVATION}" ]] && SBATCH_ARGS+=(--reservation "${SBATCH_RESERVATION}")
  [[ -n "${SBATCH_ENVIRONMENT}" ]] && SBATCH_ARGS+=(--environment "${SBATCH_ENVIRONMENT}")
  [[ -n "${SBATCH_NODES}" ]] && SBATCH_ARGS+=(--nodes "${SBATCH_NODES}")
  [[ -n "${SBATCH_NTASKS_PER_NODE}" ]] && SBATCH_ARGS+=(--ntasks-per-node "${SBATCH_NTASKS_PER_NODE}")
  [[ -n "${SBATCH_CPUS_PER_TASK}" ]] && SBATCH_ARGS+=(--cpus-per-task "${SBATCH_CPUS_PER_TASK}")
  [[ -n "${SBATCH_GPUS_PER_NODE}" ]] && SBATCH_ARGS+=(--gpus-per-node "${SBATCH_GPUS_PER_NODE}")
  [[ -n "${SBATCH_CONSTRAINT}" ]] && SBATCH_ARGS+=(--constraint "${SBATCH_CONSTRAINT}")
  true
}

render_task_files() {
  mkdir -p "${TASK_OUTPUT_DIR}"
  cp "${SOURCE_CONFIG_TEMPLATE}" "${TASK_CONFIG_TEMPLATE}"
  evaluator_write_env_file "${TASK_NAME}" "${TASK_OUTPUT_DIR}" "${TASK_ENV}"
  write_config_json_file
}

write_config_json_file() {
  TASK_CONFIG_JSON=""
  if [[ -z "${CONFIG_JSON}" && -z "${CONFIG_JSON_FILE}" ]]; then
    return
  fi

  TASK_CONFIG_JSON="${TASK_OUTPUT_DIR}/config.overrides.json"
  if [[ -n "${CONFIG_JSON_FILE}" ]]; then
    CONFIG_JSON_FILE="${CONFIG_JSON_FILE}" TASK_CONFIG_JSON="${TASK_CONFIG_JSON}" python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["CONFIG_JSON_FILE"])
dst = Path(os.environ["TASK_CONFIG_JSON"])
data = json.loads(src.read_text())
dst.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  else
    CONFIG_JSON="${CONFIG_JSON}" TASK_CONFIG_JSON="${TASK_CONFIG_JSON}" python - <<'PY'
import json
import os
from pathlib import Path

data = json.loads(os.environ["CONFIG_JSON"])
Path(os.environ["TASK_CONFIG_JSON"]).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  fi
}

print_task_summary() {
  echo "--- submit: runner=${RUNNER} task=${TASK_NAME} output=${TASK_OUTPUT_DIR} ---"
  echo "    template source: ${SOURCE_CONFIG_TEMPLATE}"
  echo "    config:          ${TASK_CONFIG_TEMPLATE}"
  echo "    env:    ${TASK_ENV}"
  if [[ -n "${TASK_CONFIG_JSON}" ]]; then
    echo "    json:   ${TASK_CONFIG_JSON}"
  fi
  echo "    logs:   ${JOB_OUTPUT} / ${JOB_ERROR}"
}

submit_interactive() {
  CMD=(bash "${SLURM_TEMPLATE}" "${JOB_ARGS[@]}")
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf ' %q' "${CMD[@]}"
    printf ' >%q 2>%q\n' "${JOB_OUTPUT}" "${JOB_ERROR}"
  else
    "${CMD[@]}" >"${JOB_OUTPUT}" 2>"${JOB_ERROR}"
  fi
}

submit_batch() {
  build_sbatch_args
  CMD=(sbatch "${SBATCH_ARGS[@]}" "${SLURM_TEMPLATE}" "${JOB_ARGS[@]}")
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf ' %q' "${CMD[@]}"
    printf '\n'
  else
    "${CMD[@]}"
  fi
}

submit_task() {
  case "${SUBMIT_MODE}" in
    interactive) submit_interactive ;;
    batch) submit_batch ;;
  esac
}

prepare_task() {
  local task="$1"
  RUNNER="${RUNNER_DEFAULT}"
  TASK_NAME="$(task_payload "${task}")"
  TASK_SLUG="$(safe_name "${task}")"
  TASK_OUTPUT_DIR="${RUN_OUTPUT_DIR}/${TASK_SLUG}"
  TASK_CONFIG_TEMPLATE="${TASK_OUTPUT_DIR}/config.template.yaml"
  TASK_ENV="${TASK_OUTPUT_DIR}/config.env"
  TASK_CONFIG_JSON=""
  JOB_OUTPUT="${RUN_LOG_DIR}/evaluator-${TASK_SLUG}_${SUBMIT_MODE}.out"
  JOB_ERROR="${RUN_LOG_DIR}/evaluator-${TASK_SLUG}_${SUBMIT_MODE}.err"
  SOURCE_CONFIG_TEMPLATE="$(select_config_template "${TASK_SLUG}")"
}

run_task() {
  local task="$1"
  [[ -z "${task}" ]] && return

  prepare_task "${task}"
  render_task_files
  build_job_args
  print_task_summary
  submit_task
}

run_tasks() {
  local task
  while IFS= read -r task; do
    run_task "${task}"
  done <<< "${TASKS}"
}

main() {
  parse_args "$@"
  require_submit_mode
  normalize_paths
  resolve_tasks
  validate_inputs
  prepare_runtime_env
  print_summary
  run_tasks

  echo "========================================"
  echo "Evaluator submissions complete. Logs: ${RUN_LOG_DIR}"
  echo "========================================"
}

main "$@"
