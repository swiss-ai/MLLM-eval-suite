#!/usr/bin/env bash
# Submit the audio evaluation of the Apertus 1.5 release checkpoints from this checkout.
#
#   CKPT_DIR=/path/with/Apertus-v1.5-8B-and-70B \
#     bash scripts/audio_repro/submit_apertus.sh <8b|70b> <run-id>
#
# Generation: each task's declared max_new_tokens (enforced by the pinned
# lmms-eval), TED-LIUM long-form 4096, greedy. See docs/audio/REPRODUCING.md.
set -euo pipefail
SIZE=${1:?8b|70b}; RUN_ID=${2:?run id}
: "${CKPT_DIR:?set CKPT_DIR to the directory holding Apertus-v1.5-8B and Apertus-v1.5-70B}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
if [[ -n "$(git status --short --ignore-submodules=none)" ]]; then
  echo "refusing to submit from a checkout with local changes" >&2; exit 1
fi

export EVAL_ENVIRONMENT=${EVAL_ENVIRONMENT:-$ROOT/toml/shared/apertus-vllm-release-eval.toml}
export SKIP_PREFLIGHT=1 VLLM_MAX_AUDIO_DECODE_DURATION_S=3600   # TED-LIUM long-form talks run ~22 min
unset GEN_KWARGS BATCH_SIZE NUM_PROCESSES GPU_MEMORY_UTILIZATION EXTRA_MODEL_ARGS

TASKS=librispeech,open_asr_voxpopuli,open_asr_spgispeech,fleurs_en_us,fleurs_de_de,fleurs_fr_fr,fleurs_it_it,fleurs_es_419,fleurs_pl_pl,fleurs_uk_ua,covost2,mmau,muchomusic,clotho_aqa,vocalsound_test

case "$SIZE" in
  8b)  MODEL=(--model "$CKPT_DIR/Apertus-v1.5-8B"); TED=() ;;
  # TP=4 with CUDA graphs. The fused all-reduce + RMSNorm pass crashes graph
  # capture at <=128 tokens on this vLLM build, so only that pass is disabled.
  70b) MODEL=(--model "$CKPT_DIR/Apertus-v1.5-70B" --size 70b
              --extra-model-args 'tensor_parallel_size=4,compilation_config={"pass_config":{"fuse_allreduce_rms":false}}')
       TED=(--gpu-memory-utilization 0.75) ;;   # graphs need memory outside vLLM's share
  *)   echo "size must be 8b or 70b" >&2; exit 2 ;;
esac

submit() { bash launchers/eval.sh --eval-framework lmms-eval "${MODEL[@]}" --run-id "$RUN_ID" "$@"; }

echo "run $RUN_ID: suite $(git rev-parse --short HEAD), lmms-eval $(git -C third_party/lmms-eval rev-parse --short HEAD)"
submit --tasks "$TASKS"
# Long-form items are 36k-52k audio tokens; the encoder cache follows
# max_num_batched_tokens. The task cap of 256 truncates the ~3,000-word transcripts.
submit --tasks tedlium_long_form --max-num-batched-tokens 65536 --gen-kwargs max_new_tokens=4096 "${TED[@]}"
