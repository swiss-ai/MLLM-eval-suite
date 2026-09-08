#!/bin/bash
# account / environment / reservation come from the launcher via
# slurm/shared/sbatch_overrides.sh (EVAL_ACCOUNT / EVAL_ENVIRONMENT /
# EVAL_RESERVATION). For direct submission: source that snippet and pass
# "${SBATCH_OVERRIDES[@]}" to sbatch.
#SBATCH --job-name=lmms-eval-apertus
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=logs/lmms-eval/apertus_eval_%j.out
#SBATCH --error=logs/lmms-eval/apertus_eval_%j.err

# Canonical eval job template — called by eval.sh.
# Supports both fill and readonly modes via --image-token-cache-* flags
# passed by the orchestrator. Defaults are tuned for readonly (production eval).
#
# Direct invocation is supported for advanced use; the more common entrypoint
# is the orchestrator:
#   bash eval.sh <model> [--tasks ...] [--suite ...] [--mode fill|readonly]
#
# Direct example:
#   sbatch eval_job.slurm \
#     --model-path /path/to/ckpt \
#     --tokenizer-path /capstor/.../tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed \
#     --chat-template /capstor/.../tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed/chat_template.jinja \
#     --tasks mmmu_val \
#     --image-token-cache-dir /capstor/.../benchmark/image_token_cache/mmmu_val \
#     --image-token-cache-local-copy 0
set -euo pipefail
# vLLM worker aborts otherwise dump multi-GB cores into the submodule cwd.
ulimit -c 0

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ORCH_REPO_ROOT="${ORCH_REPO_ROOT:-${SUBMIT_DIR}}"
CACHE_ROOT="${CACHE_ROOT:-${ORCH_REPO_ROOT}/cache/lmms-eval}"

########################################
# Required runtime configuration
########################################
DEFAULT_TOKENIZER_PATH="${DEFAULT_APERTUS_TOKENIZER:?exported by slurm/shared/sbatch_overrides.sh}"
MODEL_PATH="/path/to/apertus/model-or-hf-id"
TOKENIZER_PATH="${DEFAULT_TOKENIZER_PATH}"
MODEL_BACKEND="${MODEL_BACKEND:-apertus_1p5_vllm}"
OUTPUT_PATH="${ORCH_REPO_ROOT}/results/lmms-eval"
LOG_DIR="${ORCH_REPO_ROOT}/logs/lmms-eval"
HF_HOME_PATH="${HF_HOME:-${ORCH_REPO_ROOT}/cache/hf}"
NLTK_DATA_PATH="${NLTK_DATA:-${ORCH_REPO_ROOT}/cache/nltk_data}"
IMAGE_TOKEN_CACHE_DIR="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR:-${CACHE_ROOT}/image_token_cache}"
XDG_CACHE_HOME_PATH="${XDG_CACHE_HOME:-${ORCH_REPO_ROOT}/cache/xdg}"
VLLM_CACHE_ROOT_PATH="${VLLM_CACHE_ROOT:-${ORCH_REPO_ROOT}/cache/vllm}"
LMMS_EVAL_MODELS_CACHE_PATH="${LMMS_EVAL_MODELS_CACHE:-${ORCH_REPO_ROOT}/cache/models}"

########################################
# Evaluation configuration
########################################
TASKS="pope,gqa,vqav2_val,mmmu_test_group_img,mme,ai2d,ocrbench_v2,chartqa,docvqa,infovqa"
BATCH_SIZE=512
NUM_PROCESSES=4
MAIN_PROCESS_PORT=29541
GPU_MEMORY_UTILIZATION=0.6
MAX_MODEL_LEN=131072
MAX_NUM_BATCHED_TOKENS=49152
HF_OVERRIDES='{"max_position_embeddings":131072}'
TRUST_REMOTE_CODE=False
ENABLE_PREFIX_CACHING=true
ENFORCE_EAGER=false


OFFLINE_DATASETS=false
LIMIT=""
GEN_KWARGS=""
CHAT_TEMPLATE=""
EXTRA_MODEL_ARGS=""
TOKENIZE_ONLY=0
SEED=1234
EVAL_SEED="${SEED}"
ENABLE_IMAGE_TOKEN_CACHE=true
IMAGE_TOKEN_CACHE_MODE="${IMAGE_TOKEN_CACHE_MODE:-fill}"
PREFETCH_EMU35_VISION_TOKENIZER=true
ENABLE_WANDB=false
WANDB_PROJECT="${WANDB_PROJECT:-lmms-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-rkreft-eth-z-rich}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-eval}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_LOG_SAMPLES="${WANDB_LOG_SAMPLES:-false}"
WANDB_ARGS="${WANDB_ARGS:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

usage() {
  cat <<'EOF'
Usage:
  sbatch eval_job.slurm -- [options]

Options:
  --model-path <path_or_hf_id>           Required model path or HF id.
  --tokenizer-path <path>                Tokenizer path.
  --model-backend <name>                 lmms-eval model backend.
  --tasks <csv>                          lmms-eval task list.
  --output-path <path>                   Output directory.
  --log-dir <path>                       Log directory.
  --hf-home <path>                       HF_HOME path.
  --nltk-data <path>                     NLTK_DATA path.
  --xdg-cache-home <path>                XDG_CACHE_HOME path.
  --vllm-cache-root <path>               VLLM_CACHE_ROOT path.
  --models-cache <path>                  LMMS_EVAL_MODELS_CACHE path.
  --batch-size <int>                     Per-rank batch size.
  --num-processes <int>                  Accelerate process count.
  --main-process-port <int>              Accelerate main process port.
  --gpu-memory-utilization <float>       vLLM gpu_memory_utilization.
  --max-model-len <int>                  Max model length (>8192).
  --max-num-batched-tokens <int>         vLLM max_num_batched_tokens.
  --trust-remote-code <True|False>       vLLM trust_remote_code.
  --enable-prefix-caching <true|false>   vLLM prefix caching.
  --enforce-eager <true|false>           vLLM enforce_eager.
  --hf-overrides <json>                  vLLM hf_overrides JSON string.
  --gen-kwargs <kv_csv>                  lmms-eval --gen_kwargs string.
  --seed <seed_spec>                     lmms-eval deterministic seed tuple
                                         (e.g. 0,1234,1234,1234 or single int).
  --limit <int_or_float>                 lmms-eval --limit.
  --chat-template <path_or_template>     chat template for vLLM.
  --extra-model-args <kv_csv>            Extra --model_args suffix.
  --offline-datasets <true|false>        Default: false.
  --enable-image-token-cache <true|false>
                                         Enable Apertus image tokenization cache.
  --image-token-cache-dir <path>         Existing directory for SQLite cache.
  --image-token-cache-mode <fill|readonly>
                                         fill builds the cache; readonly uses a sealed one. Default: fill.
  --prefetch-emu35-vision-tokenizer <true|false>
                                         Prefetch BAAI/Emu3.5-VisionTokenizer once before launch.
  --enable-wandb <true|false>            Enable W&B metric logging. Default: false (forced off in this script).
  --wandb-project <name>                 W&B project. Default: lmms-eval.
  --wandb-entity <name>                  W&B entity/team. Default: rkreft-eth-z-rich.
  --wandb-job-type <name>                W&B job_type. Default: eval.
  --wandb-group <name>                   W&B run group.
  --wandb-run-name <name>                W&B run name. Default: model name.
  --wandb-run-id <id>                    W&B run id. Default: deterministic id from model name.
  --wandb-resume <mode>                  W&B resume mode (allow|must|never|auto). Default: allow.
  --wandb-log-samples <true|false>       Also upload eval samples to W&B. Default: false.
  --wandb-args <kv_csv>                  Extra lmms-eval --wandb_args (comma-separated key=value).
  --wandb-api-key <key>                  W&B API key. If omitted, uses WANDB_API_KEY env.
  --debug-mode                           Enable debug mode (sets VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG=1).
  --help                                 Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --model-backend) MODEL_BACKEND="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --hf-home) HF_HOME_PATH="$2"; shift 2 ;;
    --nltk-data) NLTK_DATA_PATH="$2"; shift 2 ;;
    --xdg-cache-home) XDG_CACHE_HOME_PATH="$2"; shift 2 ;;
    --vllm-cache-root) VLLM_CACHE_ROOT_PATH="$2"; shift 2 ;;
    --models-cache) LMMS_EVAL_MODELS_CACHE_PATH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    --main-process-port) MAIN_PROCESS_PORT="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --trust-remote-code) TRUST_REMOTE_CODE="$2"; shift 2 ;;
    --enable-prefix-caching) ENABLE_PREFIX_CACHING="$2"; shift 2 ;;
    --enforce-eager) ENFORCE_EAGER="$2"; shift 2 ;;
    --hf-overrides) HF_OVERRIDES="$2"; shift 2 ;;
    --gen-kwargs) GEN_KWARGS="$2"; shift 2 ;;
    --seed) EVAL_SEED="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --chat-template) CHAT_TEMPLATE="$2"; shift 2 ;;
    --extra-model-args) EXTRA_MODEL_ARGS="$2"; shift 2 ;;
    --tokenize-only) TOKENIZE_ONLY=1; shift ;;
    --offline-datasets) OFFLINE_DATASETS="$2"; shift 2 ;;
    --enable-image-token-cache) ENABLE_IMAGE_TOKEN_CACHE="$2"; shift 2 ;;
    --image-token-cache-dir) IMAGE_TOKEN_CACHE_DIR="$2"; shift 2 ;;
    --image-token-cache-mode) IMAGE_TOKEN_CACHE_MODE="$2"; shift 2 ;;
    --prefetch-emu35-vision-tokenizer) PREFETCH_EMU35_VISION_TOKENIZER="$2"; shift 2 ;;
    --enable-wandb) echo "Info: W&B is disabled in this script; ignoring --enable-wandb=$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-job-type) WANDB_JOB_TYPE="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --wandb-run-id) WANDB_RUN_ID="$2"; shift 2 ;;
    --wandb-resume) WANDB_RESUME="$2"; shift 2 ;;
    --wandb-log-samples) WANDB_LOG_SAMPLES="$2"; shift 2 ;;
    --wandb-args) WANDB_ARGS="$2"; shift 2 ;;
    --wandb-api-key) WANDB_API_KEY="$2"; shift 2 ;;
    --debug-mode) export VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG=1; shift ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

is_true() {
  local value
  value="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

# Default generation cap: 16384 for non-thinking runs (Artificial Analysis
# standard). Thinking recipes pass their own 32k cap explicitly.
if [[ -z "${GEN_KWARGS}" ]]; then
  GEN_KWARGS="max_new_tokens=16384"
elif [[ ! "${GEN_KWARGS}" =~ (^|,)max_new_tokens= ]]; then
  GEN_KWARGS="${GEN_KWARGS},max_new_tokens=16384"
fi

########################################
# Runtime environment
########################################
# Dummy judge by default so fleet jobs never spend API budget by accident;
# judge-intended submissions opt in with API_TYPE=openai.
export API_TYPE="${API_TYPE:-dummy}"
source "${ORCH_REPO_ROOT}/slurm/shared/job_env.sh"
resolve_judge_key
export PATH="/opt/venv/bin:${PATH}"
# The submodule checkout is the only code source; the image carries engine and
# dependencies but no harness code.
LMMS_EVAL_DEV_PATH="${LMMS_EVAL_DEV_PATH:-${ORCH_REPO_ROOT}/third_party/lmms-eval}"
suite_pythonpath "${LMMS_EVAL_DEV_PATH}"
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME_PATH}"
resolve_hf_token
export NLTK_DATA="${NLTK_DATA_PATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME_PATH}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT_PATH}"
export LMMS_EVAL_MODELS_CACHE="${LMMS_EVAL_MODELS_CACHE_PATH}"
# Prepared arrow + extracted archives on shared storage; unset, lmms-eval falls
# back to node-local scratch and every fresh node re-extracts the datasets.
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-/capstor/store/cscs/swissai/infra01/vision-datasets/benchmark/lmms_eval_datasets_cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
# Remote-sensing task image trees (geospatial suite), staged under cache/rs_datasets.
RS_DATASETS_ROOT="${RS_DATASETS_ROOT:-${ORCH_REPO_ROOT}/cache/rs_datasets}"
export VRSBENCH_DIR="${VRSBENCH_DIR:-${RS_DATASETS_ROOT}/vrsbench}"
export GEOBENCH_DIR="${GEOBENCH_DIR:-${RS_DATASETS_ROOT}/geobench}"
export FRIEDA_DIR="${FRIEDA_DIR:-${RS_DATASETS_ROOT}/frieda}"
export BIGEARTH_S2_DIR="${BIGEARTH_S2_DIR:-${RS_DATASETS_ROOT}/bigearth/BigEarthNet-S2}"
if is_true "${ENABLE_WANDB}"; then
  if [[ -z "${WANDB_API_KEY}" && -t 0 ]]; then
    read -r -s -p "Enter WANDB_API_KEY: " WANDB_API_KEY
    echo
  fi
  if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "ERROR: W&B enabled but API key is missing. Pass --wandb-api-key or set WANDB_API_KEY."
    exit 1
  fi
  export WANDB_API_KEY
  export WANDB_ENTITY="${WANDB_ENTITY}"
  export WANDB_PROJECT="${WANDB_PROJECT}"
fi
# Keep Python hash() stable across processes/runs (affects some internal naming).
HASH_SEED="${EVAL_SEED%%,*}"
if ! [[ "${HASH_SEED}" =~ ^[0-9]+$ ]]; then
  HASH_SEED="${SEED}"
fi
export PYTHONHASHSEED="${PYTHONHASHSEED:-${HASH_SEED}}"
export VLLM_SEED="${VLLM_SEED:-${SEED}}"

if is_true "${OFFLINE_DATASETS}"; then
  export HF_DATASETS_OFFLINE=1
else
  unset HF_DATASETS_OFFLINE || true
fi

if is_true "${ENABLE_IMAGE_TOKEN_CACHE}"; then
  source "${ORCH_REPO_ROOT}/slurm/shared/image_token_cache_env.sh"
else
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR || true
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must be set."
  exit 1
fi

if [[ "${MAX_MODEL_LEN}" -le 8192 ]]; then
  echo "MAX_MODEL_LEN must be > 8192 (got ${MAX_MODEL_LEN})."
  exit 1
fi

if [[ -z "${CHAT_TEMPLATE}" && "${TOKENIZER_PATH}" == "${DEFAULT_TOKENIZER_PATH}" ]]; then
  CHAT_TEMPLATE="${TOKENIZER_PATH}/chat_template.jinja"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
  set +e
  NUM_GPUS_RAW="$(nvidia-smi -L 2>/dev/null | wc -l)"
  NVS_STATUS=$?
  set -e
  if [[ ${NVS_STATUS} -eq 0 && "${NUM_GPUS_RAW}" -gt 0 ]]; then
    NUM_GPUS="${NUM_GPUS_RAW}"
  else
    set +e
    TORCH_GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
    TORCH_STATUS=$?
    set -e
    if [[ ${TORCH_STATUS} -eq 0 && "${TORCH_GPU_COUNT}" -gt 0 ]]; then
      NUM_GPUS="${TORCH_GPU_COUNT}"
      echo "Warning: nvidia-smi probe failed (status=${NVS_STATUS}); using torch.cuda.device_count=${NUM_GPUS}."
    else
      NUM_GPUS=0
      echo "Warning: unable to detect GPUs via nvidia-smi or torch; assuming 0 visible GPUs."
    fi
  fi
fi

if [[ "${NUM_GPUS}" -lt "${NUM_PROCESSES}" ]]; then
  echo "Need at least ${NUM_PROCESSES} visible GPUs, found ${NUM_GPUS}."
  exit 1
fi

mkdir -p "${OUTPUT_PATH}" "${NLTK_DATA}" "${LOG_DIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${LMMS_EVAL_MODELS_CACHE}" "${LMMS_EVAL_DATASETS_CACHE}"

# Ensure ANTLR runtime version expected by OmegaConf/Emu3.5 tokenizer path.
# This keeps jobs robust even if the base image has a newer incompatible ANTLR.
if [[ "${PIN_ANTLR_RUNTIME:-true}" == "true" ]]; then
  /opt/venv/bin/pip install --disable-pip-version-check --no-cache-dir --quiet --force-reinstall \
    "antlr4-python3-runtime==4.9.3"
fi

source "${ORCH_REPO_ROOT}/slurm/shared/apertus_env.sh"

# Fail fast on known ANTLR/OmegaConf mismatch that otherwise appears late.
python - <<'PY'
import importlib.util
import os
import sys

try:
    import omegaconf  # noqa: F401
except Exception as exc:
    print("Preflight failed: cannot import omegaconf.", file=sys.stderr)
    print(f"Root cause: {exc}", file=sys.stderr)
    sys.exit(2)

emu35_root = os.environ["VLLM_APERTUS_EMU35_CODEBASE"]
module_path = os.path.join(emu35_root, "src", "vision_tokenizer", "__init__.py")
if not os.path.isfile(module_path):
    print(f"Preflight failed: missing Emu3.5 vision tokenizer module: {module_path}", file=sys.stderr)
    sys.exit(2)

spec = importlib.util.spec_from_file_location(
    "emu35_preflight_module",
    module_path,
    submodule_search_locations=[os.path.dirname(module_path)],
)
if spec is None or spec.loader is None:
    print(f"Preflight failed: unable to build import spec for {module_path}", file=sys.stderr)
    sys.exit(2)

module = importlib.util.module_from_spec(spec)
sys.modules["emu35_preflight_module"] = module
try:
    spec.loader.exec_module(module)
except Exception as exc:
    print("Preflight failed: cannot import Emu3.5 vision tokenizer module.", file=sys.stderr)
    print(f"Root cause: {exc}", file=sys.stderr)
    sys.exit(2)
PY

if is_true "${PREFETCH_EMU35_VISION_TOKENIZER}"; then
  PREFETCH_DIR="${LMMS_EVAL_MODELS_CACHE}/BAAI/Emu3.5-VisionTokenizer"
  if [[ ! -f "${PREFETCH_DIR}/config.yaml" || ! -f "${PREFETCH_DIR}/model.ckpt" ]]; then
    echo "Prefetching BAAI/Emu3.5-VisionTokenizer into ${PREFETCH_DIR}"
    PREFETCH_DIR_ENV="${PREFETCH_DIR}" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/Emu3.5-VisionTokenizer",
    local_dir=os.environ["PREFETCH_DIR_ENV"],
    allow_patterns=["config.yaml", "model.ckpt"],
)
PY
  fi
fi

if [[ "${MODEL_BACKEND}" != apertus* ]]; then
  # Foreign backends (emu3, llava_hf, qwen3_vl, ...) own their own arg schema;
  # the vLLM/Apertus-shaped string below would be rejected. Pass through only
  # what the caller supplies (wrapper defaults cover the rest, e.g. emu3 ->
  # BAAI/Emu3-Chat).
  MODEL_ARGS="${EXTRA_MODEL_ARGS}"
  EXTRA_MODEL_ARGS=""
  # Without this the wrapper falls back to its own default checkpoint and the run
  # silently evaluates a different model than --model-path names. vLLM-style
  # and emu3 wrappers take model=; HF-style wrappers take pretrained=.
  if [[ "${MODEL_ARGS}" != *model=* && "${MODEL_ARGS}" != *pretrained=* ]]; then
    case "${MODEL_BACKEND}" in
      emu3*|*vllm*) MODEL_ARGS="model=${MODEL_PATH}${MODEL_ARGS:+,${MODEL_ARGS}}" ;;
      *)            MODEL_ARGS="pretrained=${MODEL_PATH}${MODEL_ARGS:+,${MODEL_ARGS}}" ;;
    esac
  fi
  # Gemma 3's encoder is fixed at 896x896, so high-resolution pages are squashed
  # until pan-and-scan tiles them; the shipped preprocessor leaves it off while
  # the published scores assume it on (InfoVQA 52.6 -> 68.7 here).
  if [[ "${MODEL_PATH}" == *gemma-3* && "${MODEL_ARGS}" != *do_pan_and_scan* ]]; then
    MODEL_ARGS="${MODEL_ARGS:+${MODEL_ARGS},}mm_processor_kwargs={\"do_pan_and_scan\":true}"
  fi
else
MODEL_ARGS="model=${MODEL_PATH},tokenizer=${TOKENIZER_PATH},tensor_parallel_size=1,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},trust_remote_code=${TRUST_REMOTE_CODE},max_model_len=${MAX_MODEL_LEN},hf_overrides=${HF_OVERRIDES}"
MODEL_ARGS="${MODEL_ARGS},max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS},enable_prefix_caching=${ENABLE_PREFIX_CACHING},enforce_eager=${ENFORCE_EAGER}"
fi

if [[ -n "${CHAT_TEMPLATE}" && "${MODEL_BACKEND}" == apertus* ]]; then
  MODEL_ARGS="${MODEL_ARGS},chat_template=${CHAT_TEMPLATE}"
fi
if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

MODEL_LABEL="$(basename "${MODEL_PATH}")"
if [[ "${MODEL_LABEL}" == "HF" ]]; then
  MODEL_LABEL="$(basename "$(dirname "${MODEL_PATH}")")"
fi
MODEL_LABEL="${MODEL_LABEL// /_}"
if [[ -z "${WANDB_RUN_NAME}" ]]; then
  WANDB_RUN_NAME="${MODEL_LABEL}"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  WANDB_MODEL_HASH="$(printf "%s" "${MODEL_LABEL}" | sha1sum | awk '{print substr($1,1,12)}')"
  WANDB_RUN_ID="model_${WANDB_MODEL_HASH}"
fi

append_wandb_arg() {
  local key="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    return
  fi
  if [[ -n "${FINAL_WANDB_ARGS}" ]]; then
    FINAL_WANDB_ARGS+=","
  fi
  FINAL_WANDB_ARGS+="${key}=${value}"
}

wandb_arg_exists() {
  local key="$1"
  [[ ",${FINAL_WANDB_ARGS}," == *",${key}="* ]]
}

FINAL_WANDB_ARGS="${WANDB_ARGS}"
if ! wandb_arg_exists "project"; then
  append_wandb_arg "project" "${WANDB_PROJECT}"
fi
if ! wandb_arg_exists "entity"; then
  append_wandb_arg "entity" "${WANDB_ENTITY}"
fi
if ! wandb_arg_exists "job_type"; then
  append_wandb_arg "job_type" "${WANDB_JOB_TYPE}"
fi
if ! wandb_arg_exists "group"; then
  append_wandb_arg "group" "${WANDB_GROUP}"
fi
if ! wandb_arg_exists "name"; then
  append_wandb_arg "name" "${WANDB_RUN_NAME}"
fi
if ! wandb_arg_exists "id"; then
  append_wandb_arg "id" "${WANDB_RUN_ID}"
fi
if ! wandb_arg_exists "resume"; then
  append_wandb_arg "resume" "${WANDB_RESUME}"
fi

CMD=(
  accelerate launch
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --main_process_port "${MAIN_PROCESS_PORT}"
  -m lmms_eval
  --model "${MODEL_BACKEND}"
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --seed "${EVAL_SEED}"
  --output_path "${OUTPUT_PATH}"
  --log_samples
  --log_samples_suffix "vllm_prod"
)

if [[ -n "${GEN_KWARGS}" ]]; then
  CMD+=(--gen_kwargs "${GEN_KWARGS}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if is_true "${ENABLE_WANDB}" && [[ -n "${FINAL_WANDB_ARGS}" ]]; then
  CMD+=(--wandb_args "${FINAL_WANDB_ARGS}")
  if is_true "${WANDB_LOG_SAMPLES}"; then
    CMD+=(--wandb_log_samples)
  fi
fi

/opt/venv/bin/python -c "import nltk; nltk.download('wordnet')"

echo "========== lmms-eval prod launch =========="
echo "LMMS_EVAL_RESOLVED=$(python -c 'import lmms_eval,os;print(os.path.dirname(lmms_eval.__file__))' 2>/dev/null || echo '<import failed>')"
echo "LMMS_EVAL_COMMIT=$(git -C "${LMMS_EVAL_DEV_PATH}" rev-parse --short HEAD 2>/dev/null || echo '<unknown>')"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "CHAT_TEMPLATE=${CHAT_TEMPLATE:-<tokenizer default>}"
echo "MODEL_BACKEND=${MODEL_BACKEND}"
echo "TASKS=${TASKS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}"
echo "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "ENFORCE_EAGER=${ENFORCE_EAGER}"
echo "HF_OVERRIDES=${HF_OVERRIDES}"
echo "OFFLINE_DATASETS=${OFFLINE_DATASETS}"
echo "EVAL_SEED=${EVAL_SEED}"
echo "PYTHONHASHSEED=${PYTHONHASHSEED}"
echo "VLLM_SEED=${VLLM_SEED}"
echo "ENABLE_WANDB=${ENABLE_WANDB}"
if is_true "${ENABLE_WANDB}"; then
  echo "WANDB_ARGS=${FINAL_WANDB_ARGS}"
  echo "WANDB_LOG_SAMPLES=${WANDB_LOG_SAMPLES}"
fi
echo "ENABLE_IMAGE_TOKEN_CACHE=${ENABLE_IMAGE_TOKEN_CACHE}"
if is_true "${ENABLE_IMAGE_TOKEN_CACHE}"; then
  echo "IMAGE_TOKEN_CACHE_MODE=${IMAGE_TOKEN_CACHE_MODE}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_STRICT=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_STRICT}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE=${VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS=${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE=${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG}"
fi
echo "HF_HOME=${HF_HOME}"
echo "NLTK_DATA=${NLTK_DATA}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "LMMS_EVAL_MODELS_CACHE=${LMMS_EVAL_MODELS_CACHE}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "PREFETCH_EMU35_VISION_TOKENIZER=${PREFETCH_EMU35_VISION_TOKENIZER}"
if [[ -n "${GEN_KWARGS}" ]]; then
  echo "GEN_KWARGS=${GEN_KWARGS}"
fi
echo "==========================================="

# Preflight, run manifest, harness, finalize: suite_run in slurm/shared/job_env.sh.
SUITE_PY="${SUITE_PY:-/opt/venv/bin/python}"
THINKING_FLAG=()
case "${EXTRA_MODEL_ARGS}" in *enable_thinking=True*|*enable_thinking=true*) THINKING_FLAG=(--thinking) ;; esac
PREFLIGHT_ARGS=(--model "${MODEL_PATH}" "${THINKING_FLAG[@]}" --tokenizer "${TOKENIZER_PATH}"
                --vision-tokenizer "${LMMS_EVAL_MODELS_CACHE_PATH}/BAAI/Emu3.5-VisionTokenizer"
                --container-image "${SUITE_CONTAINER_IMAGE:-}" --max-model-len "${MAX_MODEL_LEN}")
if [[ "${TOKENIZE_ONLY}" == "1" ]]; then
  suite_preflight lmms-eval "${TASKS}" "${LMMS_EVAL_DEV_PATH}" "${PREFLIGHT_ARGS[@]}"
  # Tokenize-only pass (suite/tokenize_cache.py): fill the image-token cache
  # with the VQ encoder alone, one shard per GPU, and never load the LLM.
  echo "tokenize-only: tasks=${TASKS} shards=${NUM_PROCESSES} cache=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR:-<disabled>}"
  TOKENIZE_PIDS=()
  for ((g = 0; g < NUM_PROCESSES; g++)); do
    CUDA_VISIBLE_DEVICES="${g}" "${SUITE_PY}" -m suite.tokenize_cache --tasks "${TASKS}" --shard "${g}/${NUM_PROCESSES}" \
      --tokenizer "${TOKENIZER_PATH}" > "${OUTPUT_PATH}/tokenize_shard${g}.log" 2>&1 &
    TOKENIZE_PIDS+=($!)
  done
  TOKENIZE_RC=0
  for pid in "${TOKENIZE_PIDS[@]}"; do wait "${pid}" || TOKENIZE_RC=1; done
  grep -h "tokenize_cache:" "${OUTPUT_PATH}"/tokenize_shard*.log || true
  [[ "${TOKENIZE_RC}" != "0" ]] && { echo "tokenize-only pass failed; see ${OUTPUT_PATH}/tokenize_shard*.log" >&2; tail -n 20 "${OUTPUT_PATH}"/tokenize_shard*.log >&2; }
  exit "${TOKENIZE_RC}"
fi
MANIFEST_TP="$(sed -n 's/.*tensor_parallel_size=\([0-9]*\).*/\1/p' <<<"${MODEL_ARGS}" | tail -n1)"
suite_run "${OUTPUT_PATH}" lmms-eval "${TASKS}" "${LMMS_EVAL_DEV_PATH}" "$(basename "$(dirname "${OUTPUT_PATH}")")" \
  "${PREFLIGHT_ARGS[@]}" -- \
  --model "${MODEL_PATH}" --tokenizer "${TOKENIZER_PATH}" ${CHAT_TEMPLATE:+--chat-template "${CHAT_TEMPLATE}"} \
  --model-args "${MODEL_ARGS}" --gen-kwargs "${GEN_KWARGS}" "${THINKING_FLAG[@]}" \
  --tp "${MANIFEST_TP:-1}" --dp "${NUM_PROCESSES}" --batch-size "${BATCH_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --max-model-len "${MAX_MODEL_LEN}" --limit "${LIMIT}" -- \
  "${CMD[@]}"
