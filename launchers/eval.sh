#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  launchers/eval.sh --eval-framework lmms-eval|VLMEvalKit [common args] [framework args...]

Common args:
  --eval-framework <name>   lmms-eval or VLMEvalKit
  --model <value>           Model path(s) for lmms-eval, model name(s) for VLMEvalKit
  --tasks <value>           Comma list, @file, or direct suite/task file path
  --suite <name>            Framework suite name
  --mode <value>            Framework run mode
  --submit-mode <value>     Submission mode for framework launchers
  --run-id <name>           Shared result directory name for this invocation

Use -- to separate framework-specific args if desired.

Examples:
  bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --suite smoke
  bash launchers/eval.sh --eval-framework lmms-eval /path/to/model --suite smoke
  bash launchers/eval.sh --eval-framework VLMEvalKit --suite smoke --model Apertus-1p5-8B
  bash launchers/eval.sh --eval-framework VLMEvalKit --model Apertus-1p5-8B -- --nodes 2
  bash launchers/eval.sh --eval-framework VLMEvalKit --model Apertus-1p5-8B --submit-mode interactive
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ORCH_REPO_ROOT
PREFETCH_EMU35_VISION_TOKENIZER="${PREFETCH_EMU35_VISION_TOKENIZER:-true}"
MODELS_CACHE_ROOT="${LMMS_EVAL_MODELS_CACHE:-${VLLM_APERTUS_MODELS_CACHE:-${ORCH_REPO_ROOT}/cache/models}}"

EVAL_FRAMEWORK=""
MODEL_ARG=""
TASKS_ARG=""
SUITE_ARG=""
MODE_ARG=""
SUBMIT_MODE_ARG=""
RUN_ID_ARG=""
PASSTHROUGH=()

is_true() {
  local value
  value="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

prefetch_emu35_vision_tokenizer() {
  if ! is_true "${PREFETCH_EMU35_VISION_TOKENIZER}"; then
    return 0
  fi

  local prefetch_dir="${MODELS_CACHE_ROOT}/BAAI/Emu3.5-VisionTokenizer"
  if [[ -f "${prefetch_dir}/config.yaml" && -f "${prefetch_dir}/model.ckpt" ]]; then
    return 0
  fi

  mkdir -p "${MODELS_CACHE_ROOT}"
  echo "Prefetching BAAI/Emu3.5-VisionTokenizer into ${prefetch_dir}"
  PREFETCH_DIR="${prefetch_dir}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/Emu3.5-VisionTokenizer",
    local_dir=os.environ["PREFETCH_DIR"],
    allow_patterns=["config.yaml", "model.ckpt"],
)
PY

  if [[ ! -f "${prefetch_dir}/config.yaml" || ! -f "${prefetch_dir}/model.ckpt" ]]; then
    echo "Failed to prefetch BAAI/Emu3.5-VisionTokenizer into ${prefetch_dir}" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-framework)
      EVAL_FRAMEWORK="${2:-}"
      shift 2
      ;;
    --model)
      MODEL_ARG="${2:-}"
      shift 2
      ;;
    --tasks|--data)
      TASKS_ARG="${2:-}"
      shift 2
      ;;
    --suite)
      SUITE_ARG="${2:-}"
      shift 2
      ;;
    --mode)
      MODE_ARG="${2:-}"
      shift 2
      ;;
    --submit-mode)
      SUBMIT_MODE_ARG="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID_ARG="${2:-}"
      shift 2
      ;;
    --)
      shift
      PASSTHROUGH+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${EVAL_FRAMEWORK}" ]]; then
  echo "Missing required --eval-framework." >&2
  usage >&2
  exit 2
fi

prefetch_emu35_vision_tokenizer

if [[ -n "${RUN_ID_ARG}" ]]; then
  export RUN_ID="${RUN_ID_ARG}"
fi

case "${EVAL_FRAMEWORK}" in
  lmms-eval)
    ARGS=()
    if [[ -n "${MODEL_ARG}" ]]; then
      ARGS+=("${MODEL_ARG}")
    fi
    if [[ -n "${TASKS_ARG}" ]]; then
      ARGS+=(--tasks "${TASKS_ARG}")
    fi
    if [[ -n "${SUITE_ARG}" ]]; then
      ARGS+=(--suite "${SUITE_ARG}")
    fi
    if [[ -n "${MODE_ARG}" ]]; then
      ARGS+=(--mode "${MODE_ARG}")
    fi
    if [[ -n "${SUBMIT_MODE_ARG}" ]]; then
      ARGS+=(--submit-mode "${SUBMIT_MODE_ARG}")
    fi
    ARGS+=("${PASSTHROUGH[@]}")
    exec "${ORCH_REPO_ROOT}/launchers/lmms-eval/eval.sh" "${ARGS[@]}"
    ;;
  VLMEvalKit|vlmevalkit)
    ARGS=()
    if [[ -n "${MODEL_ARG}" ]]; then
      ARGS+=(--model "${MODEL_ARG}")
    fi
    if [[ -n "${TASKS_ARG}" ]]; then
      ARGS+=(--tasks "${TASKS_ARG}")
    fi
    if [[ -n "${SUITE_ARG}" ]]; then
      ARGS+=(--suite "${SUITE_ARG}")
    fi
    if [[ -n "${MODE_ARG}" ]]; then
      ARGS+=(--mode "${MODE_ARG}")
    fi
    if [[ -n "${SUBMIT_MODE_ARG}" ]]; then
      ARGS+=(--submit-mode "${SUBMIT_MODE_ARG}")
    fi
    ARGS+=("${PASSTHROUGH[@]}")
    exec "${ORCH_REPO_ROOT}/launchers/VLMEvalKit/eval.sh" "${ARGS[@]}"
    ;;
  *)
    echo "Unsupported eval framework: ${EVAL_FRAMEWORK}" >&2
    echo "Supported values: lmms-eval, VLMEvalKit" >&2
    exit 2
    ;;
esac
