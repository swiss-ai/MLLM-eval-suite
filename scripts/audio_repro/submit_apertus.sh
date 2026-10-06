#!/usr/bin/env bash
# Submit the audio evaluation of the Apertus 1.5 release checkpoints from this checkout.
#
#   CKPT_DIR=/path/with/Apertus-v1.5-8B-and-70B \
#     bash scripts/audio_repro/submit_apertus.sh <8b|70b> <run-id> [task,...]
#
# Without a task list, all 16 tasks are submitted; with one (for example to
# resubmit a preempted job), only those tasks.
#
# The model is prompted with the tokenizer and chat template shipped with the
# checkpoint. Generation: each task's declared max_new_tokens (enforced by the
# pinned lmms-eval), TED-LIUM long-form 4096, greedy. See docs/audio/REPRODUCING.md.
set -euo pipefail
SIZE=${1:?8b|70b}; RUN_ID=${2:?run id}; ONLY=${3:-}
: "${CKPT_DIR:?set CKPT_DIR to the directory holding Apertus-v1.5-8B and Apertus-v1.5-70B}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
if [[ -n "$(git status --short --ignore-submodules=none)" ]]; then
  echo "refusing to submit from a checkout with local changes" >&2; exit 1
fi

export EVAL_ENVIRONMENT=${EVAL_ENVIRONMENT:-$ROOT/toml/shared/apertus-vllm-release-eval.toml}
export SKIP_PREFLIGHT=1 VLLM_MAX_AUDIO_DECODE_DURATION_S=3600   # TED-LIUM long-form talks run up to 22 min
unset GEN_KWARGS BATCH_SIZE NUM_PROCESSES GPU_MEMORY_UTILIZATION EXTRA_MODEL_ARGS

TASKS=librispeech,open_asr_voxpopuli,open_asr_spgispeech,fleurs_en_us,fleurs_de_de,fleurs_fr_fr,fleurs_it_it,fleurs_es_419,fleurs_pl_pl,fleurs_uk_ua,covost2,mmau,muchomusic,clotho_aqa,vocalsound_test

# Generation cap for tasks that declare none (MuChoMusic, VocalSound). A task's
# own max_new_tokens always takes precedence.
CAP=max_new_tokens=4096

case "$SIZE" in
  8b)  CKPT="$CKPT_DIR/Apertus-v1.5-8B"; MODEL=(--model "$CKPT" --extra-model-args "$CAP"); TED=() ;;
  # TP=4 with CUDA graphs. The fused all-reduce + RMSNorm pass crashes graph
  # capture at <=128 tokens on this vLLM build, so only that pass is disabled.
  70b) CKPT="$CKPT_DIR/Apertus-v1.5-70B"
       MODEL=(--model "$CKPT" --size 70b
              --extra-model-args "$CAP,tensor_parallel_size=4,compilation_config={\"pass_config\":{\"fuse_allreduce_rms\":false}}")
       TED=(--gpu-memory-utilization 0.75) ;;   # graphs need memory outside vLLM's share
  *)   echo "size must be 8b or 70b" >&2; exit 2 ;;
esac

# The checkpoint's own tokenizer and chat template. The launcher otherwise falls
# back to the suite's internal Apertus tokenizer, whose template renders audio
# requests with an extra newline after the audio.
export TOKENIZER_PATH="$CKPT" CHAT_TEMPLATE="$CKPT/chat_template.jinja"
[[ -f "$CHAT_TEMPLATE" ]] || { echo "missing $CHAT_TEMPLATE" >&2; exit 1; }

if [[ -n "$ONLY" ]]; then
  CORE=$(tr ',' '\n' <<<"$ONLY" | grep -vx tedlium_long_form | paste -sd, - || true)
  RUN_TED=$(tr ',' '\n' <<<"$ONLY" | grep -cx tedlium_long_form || true)
else
  CORE=$TASKS; RUN_TED=1
fi

submit() { bash launchers/eval.sh --eval-framework lmms-eval "${MODEL[@]}" --run-id "$RUN_ID" "$@"; }

echo "run $RUN_ID: suite $(git rev-parse --short HEAD), lmms-eval $(git -C third_party/lmms-eval rev-parse --short HEAD)"
if [[ -n "$CORE" ]]; then submit --tasks "$CORE"; fi
# The longest long-form talks exceed the default encoder cache, which follows
# max_num_batched_tokens. The task cap of 256 truncates the ~3,000-word transcripts.
if [[ "$RUN_TED" -gt 0 ]]; then
  submit --tasks tedlium_long_form --max-num-batched-tokens 65536 --gen-kwargs max_new_tokens=4096 "${TED[@]}"
fi
