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
  --run-id <name>           Shared result directory name for this invocation

Use -- to separate framework-specific args if desired.

Examples:
  bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --suite smoke
  bash launchers/eval.sh --eval-framework lmms-eval /path/to/model --suite smoke
  bash launchers/eval.sh --eval-framework VLMEvalKit --suite smoke --model Apertus-1p5-8B
  bash launchers/eval.sh --eval-framework VLMEvalKit --model Apertus-1p5-8B -- --nodes 2
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ORCH_REPO_ROOT

EVAL_FRAMEWORK=""
MODEL_ARG=""
TASKS_ARG=""
SUITE_ARG=""
MODE_ARG=""
RUN_ID_ARG=""
PASSTHROUGH=()

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
    ARGS+=("${PASSTHROUGH[@]}")
    exec "${ORCH_REPO_ROOT}/launchers/VLMEvalKit/eval.sh" "${ARGS[@]}"
    ;;
  *)
    echo "Unsupported eval framework: ${EVAL_FRAMEWORK}" >&2
    echo "Supported values: lmms-eval, VLMEvalKit" >&2
    exit 2
    ;;
esac
