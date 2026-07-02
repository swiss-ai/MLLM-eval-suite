#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=vlmeval-apertus
#SBATCH --environment=/capstor/store/cscs/swissai/infra01/multimodal-eval/MLLM-eval-suite/toml/shared/apertus-vllm-vision-eval-prod.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=04:00:00
#SBATCH --output=logs/VLMEvalKit/vlmeval_%j.out
#SBATCH --error=logs/VLMEvalKit/vlmeval_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ORCH_REPO_ROOT="${ORCH_REPO_ROOT:-${SUBMIT_DIR}}"
REPO_DIR="${ORCH_REPO_ROOT}/third_party/VLMEvalKit"
MODEL="Apertus-1p5-8B"
DATASET="3DSRBench"
MODE="all"
WORK_BASE="${ORCH_REPO_ROOT}/results/VLMEvalKit"
WORK_DIR=""  # derived after arg parsing as ${WORK_BASE}/${MODEL}/${DATASET} if not provided
RESPONSE_CACHE="${ORCH_REPO_ROOT}/cache/VLMEvalKit"
IMAGE_TOKEN_CACHE_BASE="${RESPONSE_CACHE}/image_token_cache"
IMAGE_TOKEN_CACHE_TASK=""
IMAGE_TOKEN_CACHE_DIR=""
LMU_DATA="${ORCH_REPO_ROOT}/cache/VLMEvalKit/LMUData"
RUNTIME_CACHE="${ORCH_REPO_ROOT}/cache"
NUM_PROCESSES=4
BATCH_SIZE=512
ENABLE_IMAGE_TOKEN_CACHE=true
IMAGE_TOKEN_CACHE_MODE="${IMAGE_TOKEN_CACHE_MODE:-fill}"
MAIN_PROCESS_PORT=29541
SEED=1234
PYTHON_BIN="/opt/venv/bin/python"

usage() {
  cat <<'EOF'
Usage:
  sbatch eval_job.slurm -- [options]

Options:
  --repo-dir <path>              VLMEvalKit checkout to execute.
  --model <name>                 VLMEvalKit model name. Default: Apertus-1p5-8B.
  --data <name>                  VLMEvalKit dataset name. Default: 3DSRBench.
  --mode all|infer|eval          VLMEvalKit mode. Default: all.
  --work-dir <path>              Output directory passed to run.py.
  --response-cache <path>        SQLite response cache root passed to run.py.
  --enable-image-token-cache <true|false>
                                  Export Apertus image-token cache env. Default: true.
  --image-token-cache-mode <fill|readonly>
                                  fill builds the cache; readonly uses a sealed one. Default: fill.
  --image-token-cache-base <path> Base containing per-task SQLite caches.
  --image-token-cache-task <name> Cache task name under the base. Default: dataset-derived.
  --image-token-cache-dir <path>  Exact cache dir passed to vLLM; overrides base/task.
  --lmu-data <path>              LMUData root for datasets.
  --runtime-cache <path>         Root for HF/XDG/vLLM runtime caches.
  --num-processes <int>          torch.distributed workers. Default: 4.
  --batch-size <int>             Batch size value passed through/logged for the framework. Default: 512.
  --main-process-port <int>      torch.distributed master port.
  --seed <int>                   PYTHONHASHSEED and VLLM_SEED.
  --python <path>                Python executable. Default: /opt/venv/bin/python.
  --help                         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --data) DATASET="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --response-cache) RESPONSE_CACHE="$2"; shift 2 ;;
    --enable-image-token-cache) ENABLE_IMAGE_TOKEN_CACHE="$2"; shift 2 ;;
    --image-token-cache-mode) IMAGE_TOKEN_CACHE_MODE="$2"; shift 2 ;;
    --image-token-cache-base) IMAGE_TOKEN_CACHE_BASE="$2"; shift 2 ;;
    --image-token-cache-task) IMAGE_TOKEN_CACHE_TASK="$2"; shift 2 ;;
    --image-token-cache-dir) IMAGE_TOKEN_CACHE_DIR="$2"; shift 2 ;;
    --lmu-data) LMU_DATA="$2"; shift 2 ;;
    --runtime-cache) RUNTIME_CACHE="$2"; shift 2 ;;
    --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    --tensor-parallel-size) APERTUS_TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --gpu-memory-utilization) APERTUS_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --main-process-port) MAIN_PROCESS_PORT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${MODE}" in
  all|infer|eval) ;;
  *) echo "--mode must be all, infer, or eval (got: ${MODE})" >&2; exit 1 ;;
esac

is_true() {
  local value
  value="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

: "${WORK_DIR:=${WORK_BASE}/${MODEL}/${DATASET}}"

if [[ ! "${NUM_PROCESSES}" =~ ^[0-9]+$ || "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "--num-processes must be a positive integer (got: ${NUM_PROCESSES})" >&2
  exit 1
fi

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "VLMEvalKit repo not found: ${REPO_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"

VLLM_REPO="${VLLM_REPO:-/workspace/vllm}"
export PATH="/opt/venv/bin:${PATH}"
# DP workers race on the shared torch.compile cache during CUDA-graph capture; compile independently.
export VLLM_DISABLE_COMPILE_CACHE=1
if [[ -d "${VLLM_REPO}" ]]; then
  export PYTHONPATH="${REPO_DIR}:${VLLM_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi
# /capstor/.../MLLM/wheelhouse provides libjson-c.so.5 which the slurm
# switch/hpe_slingshot plugin dlopen()s when srun runs inside this container
# (the container itself lacks libjson-c5). Without this path, multi-node
# srun fails before launching anything with "Can't find plugin for
# switch/hpe_slingshot". Single-node runs work either way.
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}"

mkdir -p \
  "${WORK_DIR}" \
  "${RESPONSE_CACHE}" \
  "${LMU_DATA}" \
  "${RUNTIME_CACHE}/hf" \
  "${RUNTIME_CACHE}/xdg" \
  "${RUNTIME_CACHE}/vllm" \
  "${RUNTIME_CACHE}/models"

export LMUData="${LMU_DATA}"
export VLMEVAL_RESPONSE_CACHE="${RESPONSE_CACHE}"
export HF_HOME="${HF_HOME:-${RUNTIME_CACHE}/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_CACHE}/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${RUNTIME_CACHE}/vllm}"
export VLLM_APERTUS_MODELS_CACHE="${VLLM_APERTUS_MODELS_CACHE:-${RUNTIME_CACHE}/models}"
export LMMS_EVAL_MODELS_CACHE="${LMMS_EVAL_MODELS_CACHE:-${VLLM_APERTUS_MODELS_CACHE}}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export VLLM_LOOPBACK_IP="${VLLM_LOOPBACK_IP:-127.0.0.1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export VLLM_SEED="${VLLM_SEED:-${SEED}}"

# Path-driven Apertus identity for the config.py registry entry: APERTUS_RUN_NAME
# is the supported_VLM key (matches --model); APERTUS_MODEL_PATH is the checkpoint
# (defaults to --model when it is a path, else the wrapper default). Override knobs via env
# (e.g. APERTUS_ENABLE_THINKING=1 APERTUS_TEMPERATURE=0.6) or edit the defaults
# here; thinking needs APERTUS_TEMPERATURE > 0 and APERTUS_MAX_NEW_TOKENS ~16384.
export APERTUS_RUN_NAME="${MODEL}"
if [[ -z "${APERTUS_MODEL_PATH:-}" && "${MODEL}" == /* ]]; then
  [[ -d "${MODEL}" ]] || { echo "ERROR: --model '${MODEL}' is path-shaped but not a directory; refusing to silently load the default checkpoint." >&2; exit 1; }
  APERTUS_MODEL_PATH="${MODEL}"
fi
export APERTUS_MODEL_PATH="${APERTUS_MODEL_PATH:-}"
export APERTUS_ENABLE_THINKING="${APERTUS_ENABLE_THINKING:-}"
export APERTUS_TEMPERATURE="${APERTUS_TEMPERATURE:-}"
export APERTUS_TOP_P="${APERTUS_TOP_P:-}"
export APERTUS_REPETITION_PENALTY="${APERTUS_REPETITION_PENALTY:-}"
export APERTUS_MAX_NEW_TOKENS="${APERTUS_MAX_NEW_TOKENS:-}"
export APERTUS_TENSOR_PARALLEL_SIZE="${APERTUS_TENSOR_PARALLEL_SIZE:-}"
export APERTUS_GPU_MEMORY_UTILIZATION="${APERTUS_GPU_MEMORY_UTILIZATION:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if is_true "${ENABLE_IMAGE_TOKEN_CACHE}"; then
  if [[ -z "${IMAGE_TOKEN_CACHE_DIR}" ]]; then
    if [[ -z "${IMAGE_TOKEN_CACHE_TASK}" ]]; then
      IMAGE_TOKEN_CACHE_TASK="$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]')"
    fi
    IMAGE_TOKEN_CACHE_DIR="${IMAGE_TOKEN_CACHE_BASE}/${IMAGE_TOKEN_CACHE_TASK}"
  fi
  source "${ORCH_REPO_ROOT}/slurm/shared/image_token_cache_env.sh"
else
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR || true
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
else
  set +e
  NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
  NVS_STATUS=$?
  set -e
  if [[ "${NVS_STATUS}" -ne 0 || "${NUM_GPUS}" -eq 0 ]]; then
    NUM_GPUS="$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  fi
fi

NEED_GPUS=$(( NUM_PROCESSES * ${APERTUS_TENSOR_PARALLEL_SIZE:-1} ))
if [[ "${NUM_GPUS}" -lt "${NEED_GPUS}" ]]; then
  echo "Need at least ${NEED_GPUS} visible GPUs (${NUM_PROCESSES} workers x TP=${APERTUS_TENSOR_PARALLEL_SIZE:-1}), found ${NUM_GPUS}." >&2
  exit 1
fi

# TODO(2026-08-01): drop this whole if/fi (and the marker check in the
# Dockerfile) once /etc/apertus_image_version is baked into every cluster image
# we run against. Tracked as "legacy-image overlay" cleanup.
# Until then: pip install bs4/nltk and download wordnet at runtime because the
# older container image is missing them. Persists to RUNTIME_CACHE so a job
# that lands on a node with a warm cache skips the install + download.
if [[ ! -f /etc/apertus_image_version ]]; then
  PIP_OVERLAY="${RUNTIME_CACHE}/pip_pkgs"
  export NLTK_DATA="${RUNTIME_CACHE}/nltk_data"
  if [[ -d "${PIP_OVERLAY}/bs4" && -f "${NLTK_DATA}/corpora/wordnet.zip" ]]; then
    echo "[preflight] legacy image detected; reusing cached bs4 + wordnet overlay" >&2
    export PYTHONPATH="${PIP_OVERLAY}:${PYTHONPATH:-}"
  else
    echo "[preflight] legacy image detected; installing bs4 + wordnet overlay" >&2
    mkdir -p "${PIP_OVERLAY}" "${NLTK_DATA}"
    # Drop inherited PIP_CONSTRAINT from the base image (points at a non-existent
    # /etc/pip/constraint.txt) for this isolated --target install only.
    PIP_CONSTRAINT="" "${PYTHON_BIN}" -m pip install \
      --quiet --target="${PIP_OVERLAY}" beautifulsoup4 nltk >&2
    export PYTHONPATH="${PIP_OVERLAY}:${PYTHONPATH:-}"
    "${PYTHON_BIN}" -c "import nltk; nltk.download('wordnet', download_dir='${NLTK_DATA}', quiet=True)" >&2
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = "BAAI/Emu3.5-VisionTokenizer"
cache_dir = Path(os.environ["VLLM_APERTUS_MODELS_CACHE"]).expanduser()
local_dir = cache_dir / repo_id
required = [local_dir / "config.yaml", local_dir / "model.ckpt"]

if not all(path.is_file() for path in required):
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=["config.yaml", "model.ckpt"],
    )

missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"missing Apertus vision tokenizer files: {missing}")

print("preflight apertus vision tokenizer", local_dir)
PY

"${PYTHON_BIN}" - <<PY
import os
import torch
import vllm
from vlmeval.config import supported_VLM
from vlmeval.dataset import SUPPORTED_DATASETS

model = "${MODEL}"
dataset = "${DATASET}"
assert model in supported_VLM, f"unsupported model: {model}"
assert dataset in SUPPORTED_DATASETS, f"unsupported dataset: {dataset}"
print("preflight repo", os.getcwd())
print("preflight torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print("preflight vllm", getattr(vllm, "__version__", "unknown"))
print("preflight model", model)
print("preflight dataset", dataset)
print("preflight LMUData", os.environ["LMUData"])
PY

# NNODES > 1 means multi-node DP: slurm allocates 4 nodes, one srun task per
# node, each runs torchrun which fans out NUM_PROCESSES local ranks. The
# rendezvous endpoint is the first node from SLURM_JOB_NODELIST.
NNODES="${SLURM_NNODES:-1}"

# Build the python invocation (optional torchrun DP prefix + run.py + args),
# shared by the single- and multi-node paths below.
PY_CMD=("${PYTHON_BIN}")
if [[ "${NNODES}" -gt 1 ]]; then
  MASTER_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
  echo "MULTI_NODE=true NNODES=${NNODES} MASTER_NODE=${MASTER_NODE}"
  PY_CMD+=(
    -m torch.distributed.run
    --nnodes "${NNODES}"
    --nproc-per-node "${NUM_PROCESSES}"
    --rdzv-id "${SLURM_JOB_ID}"
    --rdzv-backend c10d
    --rdzv-endpoint "${MASTER_NODE}:${MAIN_PROCESS_PORT}"
  )
elif [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  PY_CMD+=(
    -m torch.distributed.run
    --nproc-per-node "${NUM_PROCESSES}"
    --master-port "${MAIN_PROCESS_PORT}"
  )
fi
PY_CMD+=(
  run.py
  --data "${DATASET}"
  --model "${MODEL}"
  --mode "${MODE}"
  --work-dir "${WORK_DIR}"
  --response-cache "${RESPONSE_CACHE}"
)

if [[ "${NNODES}" -gt 1 ]]; then
  # srun launches one worker container per node. The EDF .toml sets
  # workdir=/workspace, which overrides `srun --chdir`, so each worker starts at
  # /workspace and a bare `run.py` (or a leaked CWD) resolves to nothing / the
  # stale baked-in copy. cd into REPO_DIR *inside* each container — exactly what
  # the head-node batch script does at `cd ${REPO_DIR}` above — so both `run.py`
  # and `import vlmeval` resolve to the live /iopsstor checkout.
  RUN_CMD=(
    srun
    --ntasks="${NNODES}"
    --ntasks-per-node=1
    --kill-on-bad-exit=1
    bash -c 'cd "$1"; shift; exec "$@"' _ "${REPO_DIR}"
    "${PY_CMD[@]}"
  )
else
  RUN_CMD=("${PY_CMD[@]}")
fi

echo "========== VLMEvalKit launch =========="
echo "REPO_DIR=${REPO_DIR}"
echo "MODEL=${MODEL}"
echo "DATASET=${DATASET}"
echo "MODE=${MODE}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "WORK_DIR=${WORK_DIR}"
echo "RESPONSE_CACHE=${RESPONSE_CACHE}"
echo "ENABLE_IMAGE_TOKEN_CACHE=${ENABLE_IMAGE_TOKEN_CACHE}"
if is_true "${ENABLE_IMAGE_TOKEN_CACHE}"; then
  echo "IMAGE_TOKEN_CACHE_MODE=${IMAGE_TOKEN_CACHE_MODE}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_STRICT=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_STRICT}"
fi
echo "LMUData=${LMUData}"
echo "HF_HOME=${HF_HOME}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "VLLM_APERTUS_MODELS_CACHE=${VLLM_APERTUS_MODELS_CACHE}"
echo "LMMS_EVAL_MODELS_CACHE=${LMMS_EVAL_MODELS_CACHE}"
echo "VLLM_HOST_IP=${VLLM_HOST_IP}"
echo "VLLM_LOOPBACK_IP=${VLLM_LOOPBACK_IP}"
echo "PYTHONPATH=${PYTHONPATH}"
printf 'CMD:'
printf ' %q' "${RUN_CMD[@]}"
printf '\n'
echo "======================================="

"${RUN_CMD[@]}"
