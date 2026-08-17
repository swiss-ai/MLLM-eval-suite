#!/bin/bash
# account / environment / reservation come from the launcher via
# slurm/shared/sbatch_overrides.sh (EVAL_ACCOUNT / EVAL_ENVIRONMENT /
# EVAL_RESERVATION). For direct submission: source that snippet and pass
# "${SBATCH_OVERRIDES[@]}" to sbatch.
#SBATCH --nodes=1
#SBATCH --exclusive
set -euo pipefail

ORCH_REPO_ROOT="${ORCH_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)}"

MODEL_PATH=""
TASKS=""
OUTPUT_PATH=""
NUM_PROCESSES=4
HF_HOME_PATH="${HF_HOME:-${ORCH_REPO_ROOT}/cache/hf}"
GEN_KWARGS=""
EXTRA_MODEL_ARGS=""
APPLY_CHAT_TEMPLATE=0
LIMIT=""
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

usage() {
  cat <<'EOF'
eval_job.slurm --model-path P --tasks T --output-path O
  [--num-processes N] [--hf-home P] [--gen-kwargs K] [--extra-model-args A]
  [--apply-chat-template] [--limit N] [--gpu-memory-utilization F] [--max-model-len N]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)             MODEL_PATH="$2"; shift 2 ;;
    --tasks)                  TASKS="$2"; shift 2 ;;
    --output-path)            OUTPUT_PATH="$2"; shift 2 ;;
    --num-processes)          NUM_PROCESSES="$2"; shift 2 ;;
    --hf-home)                HF_HOME_PATH="$2"; shift 2 ;;
    --gen-kwargs)             GEN_KWARGS="$2"; shift 2 ;;
    --extra-model-args)       EXTRA_MODEL_ARGS="$2"; shift 2 ;;
    --apply-chat-template)    APPLY_CHAT_TEMPLATE=1; shift ;;
    --limit)                  LIMIT="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max-model-len)          MAX_MODEL_LEN="$2"; shift 2 ;;
    -h|--help)                usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$MODEL_PATH" && -n "$TASKS" && -n "$OUTPUT_PATH" ]] || { usage; exit 1; }
mkdir -p "$OUTPUT_PATH"

export PATH="/opt/venv/bin:${PATH}"
export HF_HOME="$HF_HOME_PATH"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME_PATH/token" ]]; then
  export HF_TOKEN="$(<"$HF_HOME_PATH/token")"
fi

# The harness runs from the pinned submodule checkout; nothing is installed at
# job time. lm_eval/__init__ resolves its version from installed package
# metadata, so provide ephemeral dist-info naming the actual submodule commit
# (dies with the job; a missing checkout still fails loudly).
LM_EVAL_DEV_PATH="${LM_EVAL_DEV_PATH:-${ORCH_REPO_ROOT}/third_party/lm-eval-harness}"
META_DIR="$(mktemp -d)/meta"
DIST_INFO="${META_DIR}/lm_eval-0.dist-info"
mkdir -p "$DIST_INFO"
LM_EVAL_COMMIT="$(git -C "$LM_EVAL_DEV_PATH" rev-parse --short HEAD 2>/dev/null || echo unknown)"
printf 'Metadata-Version: 2.1\nName: lm_eval\nVersion: 0+g%s\n' "$LM_EVAL_COMMIT" > "$DIST_INFO/METADATA"
export PYTHONPATH="${LM_EVAL_DEV_PATH}:${ORCH_REPO_ROOT}/shared:${META_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_ARGS="pretrained=${MODEL_PATH},tensor_parallel_size=1,data_parallel_size=${NUM_PROCESSES},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN}"
if [[ -n "$EXTRA_MODEL_ARGS" ]]; then
  MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

CMD=(python -m lm_eval
  --model vllm
  --model_args "$MODEL_ARGS"
  --tasks "$TASKS"
  --batch_size auto
  --output_path "$OUTPUT_PATH"
  --log_samples
)
INCLUDE_DIR="${ORCH_REPO_ROOT}/task_suites/lm-eval/custom"
if compgen -G "${INCLUDE_DIR}/*.yaml" >/dev/null 2>&1; then
  CMD+=(--include_path "$INCLUDE_DIR")
fi
[[ -n "$GEN_KWARGS" ]] && CMD+=(--gen_kwargs "$GEN_KWARGS")
[[ "$APPLY_CHAT_TEMPLATE" -eq 1 ]] && CMD+=(--apply_chat_template)
[[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")

echo "========== lm-eval launch =========="
echo "LM_EVAL_COMMIT=${LM_EVAL_COMMIT}"
echo "LM_EVAL_RESOLVED=$(python -c 'import lm_eval,os;print(os.path.dirname(lm_eval.__file__))' 2>/dev/null || echo '<import failed>')"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TASKS=${TASKS}"
printf 'CMD:'; printf ' %q' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
