#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  launchers/eval.sh [--eval-framework all|lmms-eval|VLMEvalKit] [common args] [framework args...]

  Default framework is `all`: one checkpoint is run through BOTH harnesses in a
  single invocation (lmms-eval for its benchmarks, VLMEvalKit for the spatial /
  multi-image set), and the dashboard merges them by checkpoint identity.

Common args:
  --eval-framework <name>   all (default), lmms-eval, or VLMEvalKit
  --model <value>           Checkpoint PATH (recommended). In `all` mode the path
                            is given to lmms-eval directly and to VLMEvalKit as a
                            run name + APERTUS_MODEL_PATH so both columns merge.
  --tasks <value>           Comma list, @file, or direct suite/task file path
  --suite <name>            Framework suite name (resolved per harness)
  --mode <value>            Framework run mode
  --submit-mode <value>     Submission mode for framework launchers
  --run-id <name>           Shared result directory name for this invocation
  --dry-run                 Print the per-harness commands without executing

Use -- to separate framework-specific args if desired.

Examples:
  bash launchers/eval.sh --model /path/to/ckpt --suite full          # both harnesses
  bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --suite smoke
  bash launchers/eval.sh --eval-framework VLMEvalKit --suite smoke --model Apertus-1p5-8B
  bash launchers/eval.sh --eval-framework VLMEvalKit --model Apertus-1p5-8B --submit-mode interactive
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
SUBMIT_MODE_ARG=""
RUN_ID_ARG=""
THINKING=0
DRY_ALL=0
PASSTHROUGH=()
EXTRA_JOB_ARGS=()

# Thinking-mode config — single source for both harnesses' translation below.
THINK_GEN_KWARGS="max_new_tokens=32768,temperature=0.6,top_p=0.95"
THINK_SUFFIX="-thinking-32k"
set_thinking_env() { export APERTUS_ENABLE_THINKING=1 APERTUS_TEMPERATURE=0.6 APERTUS_TOP_P=0.95 APERTUS_MAX_NEW_TOKENS=32768; }
# Resolve VK_MODEL (the run NAME) and export APERTUS_MODEL_PATH (the checkpoint path) from a --model value.
derive_vk_identity() {
  VK_MODEL="${1}"
  if [[ -e "${1}" ]]; then
    if [[ -f "${1}/config.json" ]] && ! grep -q "Apertus" "${1}/config.json"; then
      echo "ERROR: --model ${1} is a checkpoint path but its config.json declares a non-Apertus architecture; refusing to route it through the Apertus wrapper. Pass the model's registry name instead." >&2
      exit 1
    fi
    export APERTUS_MODEL_PATH="${1}"
    VK_MODEL="$(basename "${1%/}")"
    [[ "${VK_MODEL}" == "HF" ]] && VK_MODEL="$(basename "$(dirname "${1%/}")")"
  fi
  if [[ "${THINKING}" -eq 1 ]]; then
    set_thinking_env
    [[ -n "${VK_MODEL}" ]] && VK_MODEL="${VK_MODEL}${THINK_SUFFIX}"
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
    --dry-run)
      DRY_ALL=1
      PASSTHROUGH+=(--dry-run)
      shift
      ;;
    --thinking)
      THINKING=1
      shift
      ;;
    --)
      shift
      EXTRA_JOB_ARGS+=("$@")
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
  EVAL_FRAMEWORK="all"
fi

# Post-`--` args go to the lmms-eval job script; the VLMEvalKit launcher has no
# passthrough channel, so reject rather than silently drop them.
if [[ ${#EXTRA_JOB_ARGS[@]} -gt 0 && "${EVAL_FRAMEWORK}" != "lmms-eval" ]]; then
  echo "-- job args are wired for --eval-framework lmms-eval only" >&2
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
    if [[ -n "${SUBMIT_MODE_ARG}" ]]; then
      ARGS+=(--submit-mode "${SUBMIT_MODE_ARG}")
    fi
    if [[ "${THINKING}" -eq 1 ]]; then
      ARGS+=(--enable-thinking --gen-kwargs "$THINK_GEN_KWARGS" --label-suffix "$THINK_SUFFIX")
    fi
    ARGS+=("${PASSTHROUGH[@]}")
    if [[ ${#EXTRA_JOB_ARGS[@]} -gt 0 ]]; then
      ARGS+=(-- "${EXTRA_JOB_ARGS[@]}")
    fi
    exec "${ORCH_REPO_ROOT}/launchers/lmms-eval/eval.sh" "${ARGS[@]}"
    ;;
  VLMEvalKit|vlmevalkit)
    derive_vk_identity "${MODEL_ARG}"
    ARGS=()
    [[ -n "${VK_MODEL}" ]] && ARGS+=(--model "${VK_MODEL}")
    [[ -n "${TASKS_ARG}" ]] && ARGS+=(--tasks "${TASKS_ARG}")
    [[ -n "${SUITE_ARG}" ]] && ARGS+=(--suite "${SUITE_ARG}")
    [[ -n "${MODE_ARG}" ]] && ARGS+=(--mode "${MODE_ARG}")
    [[ -n "${SUBMIT_MODE_ARG}" ]] && ARGS+=(--submit-mode "${SUBMIT_MODE_ARG}")
    ARGS+=("${PASSTHROUGH[@]}")
    exec "${ORCH_REPO_ROOT}/launchers/VLMEvalKit/eval.sh" "${ARGS[@]}"
    ;;
  all|both)
    if [[ -n "${MODE_ARG}" ]]; then
      echo "--mode is framework-specific (lmms-eval: fill|readonly, VLMEvalKit: all|infer|eval)" >&2
      echo "and cannot be forwarded to both harnesses; omit it (defaults work) or run the" >&2
      echo "frameworks separately with --eval-framework." >&2
      exit 2
    fi
    run_harness() {
      local label="$1"; shift
      echo "[all] -> ${label}: $*"
      if [[ "${DRY_ALL}" -eq 0 ]]; then "$@"; fi
    }

    # lmms-eval takes the checkpoint PATH as its positional model argument.
    LMMS_ARGS=(bash "${ORCH_REPO_ROOT}/launchers/lmms-eval/eval.sh")
    [[ -n "${MODEL_ARG}" ]] && LMMS_ARGS+=("${MODEL_ARG}")
    [[ -n "${TASKS_ARG}" ]] && LMMS_ARGS+=(--tasks "${TASKS_ARG}")
    [[ -n "${SUITE_ARG}" ]] && LMMS_ARGS+=(--suite "${SUITE_ARG}")
    [[ -n "${MODE_ARG}" ]] && LMMS_ARGS+=(--mode "${MODE_ARG}")
    [[ -n "${SUBMIT_MODE_ARG}" ]] && LMMS_ARGS+=(--submit-mode "${SUBMIT_MODE_ARG}")
    if [[ "${THINKING}" -eq 1 ]]; then
      LMMS_ARGS+=(--enable-thinking --gen-kwargs "$THINK_GEN_KWARGS" --label-suffix "$THINK_SUFFIX")
    fi
    LMMS_ARGS+=("${PASSTHROUGH[@]}")
    run_harness "lmms-eval" "${LMMS_ARGS[@]}"

    # VLMEvalKit takes a run NAME; forward the checkpoint path via APERTUS_MODEL_PATH
    # so the run identity matches lmms-eval and the dashboard merges the columns.
    derive_vk_identity "${MODEL_ARG}"
    VK_ARGS=(bash "${ORCH_REPO_ROOT}/launchers/VLMEvalKit/eval.sh")
    [[ -n "${VK_MODEL}" ]] && VK_ARGS+=(--model "${VK_MODEL}")
    [[ -n "${TASKS_ARG}" ]] && VK_ARGS+=(--tasks "${TASKS_ARG}")
    [[ -n "${SUITE_ARG}" ]] && VK_ARGS+=(--suite "${SUITE_ARG}")
    [[ -n "${MODE_ARG}" ]] && VK_ARGS+=(--mode "${MODE_ARG}")
    [[ -n "${SUBMIT_MODE_ARG}" ]] && VK_ARGS+=(--submit-mode "${SUBMIT_MODE_ARG}")
    VK_ARGS+=("${PASSTHROUGH[@]}")
    run_harness "VLMEvalKit" "${VK_ARGS[@]}"
    ;;
  *)
    echo "Unsupported eval framework: ${EVAL_FRAMEWORK}" >&2
    echo "Supported values: all, lmms-eval, VLMEvalKit" >&2
    exit 2
    ;;
esac
