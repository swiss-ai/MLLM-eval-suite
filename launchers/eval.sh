#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  launchers/eval.sh [--eval-framework all|lmms-eval|VLMEvalKit|Evaluator|lm-evaluation-harness] [common args] [framework args...]

  Default framework is `all`: one checkpoint is run through BOTH harnesses in a
  single invocation (lmms-eval for its benchmarks, VLMEvalKit for the spatial /
  multi-image set), and the dashboard merges them by checkpoint identity.

Common args:
  --eval-framework <name>   all (default), lmms-eval, VLMEvalKit, Evaluator, or lm-evaluation-harness
  --model <value>           Checkpoint PATH (recommended). In `all` mode the path
                            is given to lmms-eval directly and to VLMEvalKit as a
                            run name + APERTUS_MODEL_PATH so both columns merge.
  --tasks <value>           Comma list, @file, or direct suite/task file path
  --suite <name>            Framework suite name (resolved per harness)
  --mode <value>            Framework run mode
  --submit-mode <value>     Submission mode for framework launchers
  --run-id <name>           Shared result directory name for this invocation
  --extra-framework-config <arg>
                            Extra framework-native argv token. Repeat for each
                            token to pass through without using -- separators.
  --dry-run                 Print the per-harness commands without executing

Use repeatable --extra-framework-config <arg> for framework-native argv tokens
that should pass through to the final framework command.

Examples:
  bash launchers/eval.sh --model /path/to/ckpt --suite full          # both harnesses
  bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --suite smoke
  bash launchers/eval.sh --eval-framework VLMEvalKit --suite smoke --model Apertus-1p5-8B
  bash launchers/eval.sh --eval-framework VLMEvalKit --model Apertus-1p5-8B --submit-mode interactive
  bash launchers/eval.sh --eval-framework Evaluator --suite smoke --model /path/to/ckpt
  bash launchers/eval.sh --eval-framework lm-evaluation-harness --suite smoke --model /path/to/ckpt
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
THINKING=0
DRY_ALL=0
PASSTHROUGH=()

# Thinking-mode config — single source for both harnesses' translation below.
THINK_GEN_KWARGS="max_new_tokens=32768,temperature=0.6,top_p=0.95"
THINK_SUFFIX="-thinking-32k"
set_thinking_env() { export APERTUS_ENABLE_THINKING=1 APERTUS_TEMPERATURE=0.6 APERTUS_TOP_P=0.95 APERTUS_MAX_NEW_TOKENS=32768; }

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
    --extra-framework-config|--extra-framework-arg)
      PASSTHROUGH+=(--extra-framework-config "${2:-}")
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
  EVAL_FRAMEWORK="all"
fi

case "${EVAL_FRAMEWORK}" in
  Evaluator|evaluator|lm-evaluation-harness|lm-eval-harness|lm_eval|lm-eval) ;;
  *) prefetch_emu35_vision_tokenizer ;;
esac

if [[ -n "${RUN_ID_ARG}" ]]; then
  export RUN_ID="${RUN_ID_ARG}"
fi

case "${EVAL_FRAMEWORK}" in
  Evaluator|evaluator)
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
    if [[ -n "${SUBMIT_MODE_ARG}" ]]; then
      ARGS+=(--submit-mode "${SUBMIT_MODE_ARG}")
    fi
    ARGS+=("${PASSTHROUGH[@]}")
    exec "${ORCH_REPO_ROOT}/launchers/Evaluator/eval.sh" "${ARGS[@]}"
    ;;
  lm-evaluation-harness|lm-eval-harness|lm_eval|lm-eval)
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
    if [[ -n "${SUBMIT_MODE_ARG}" ]]; then
      ARGS+=(--submit-mode "${SUBMIT_MODE_ARG}")
    fi
    ARGS+=("${PASSTHROUGH[@]}")
    exec "${ORCH_REPO_ROOT}/launchers/lm-evaluation-harness/eval.sh" "${ARGS[@]}"
    ;;
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
    if [[ "${THINKING}" -eq 1 ]]; then
      set_thinking_env
      for i in "${!ARGS[@]}"; do
        if [[ "${ARGS[$i]}" == "--model" ]]; then ARGS[$((i+1))]="${ARGS[$((i+1))]}${THINK_SUFFIX}"; break; fi
      done
    fi
    ARGS+=("${PASSTHROUGH[@]}")
    exec "${ORCH_REPO_ROOT}/launchers/VLMEvalKit/eval.sh" "${ARGS[@]}"
    ;;
  all|both)
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

    # VLMEvalKit takes a run NAME; forward a checkpoint path via APERTUS_MODEL_PATH
    # so the run identity matches lmms-eval and the dashboard merges the columns.
    VK_MODEL="${MODEL_ARG}"
    if [[ -e "${MODEL_ARG}" ]]; then
      export APERTUS_MODEL_PATH="${MODEL_ARG}"
      VK_MODEL="$(basename "${MODEL_ARG%/}")"
    fi
    if [[ "${THINKING}" -eq 1 ]]; then
      set_thinking_env
      VK_MODEL="${VK_MODEL}${THINK_SUFFIX}"
    fi
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
    echo "Supported values: all, lmms-eval, VLMEvalKit, Evaluator, lm-evaluation-harness" >&2
    exit 2
    ;;
esac
