#!/usr/bin/env bash
# Submit the audio evaluation of the Apertus 1.5 release checkpoints from this checkout.
#
#   CKPT_DIR=/path/with/Apertus-v1.5-8B-and-70B \
#     bash scripts/audio_repro/submit_apertus.sh <8b|70b> <run-id> [task,...]
#
# Without a task list, the 16 tasks of the report's audio table are submitted
# (one job per task); with one (for example to resubmit a preempted job), only
# those tasks. See docs/audio/REPRODUCING.md for every setting and why.
set -euo pipefail
SIZE=${1:?8b|70b}; RUN_ID=${2:?run id}; ONLY=${3:-}
: "${CKPT_DIR:?set CKPT_DIR to the directory holding Apertus-v1.5-8B and Apertus-v1.5-70B}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
if [[ -n "$(git status --short --ignore-submodules=none)" ]]; then
  echo "refusing to submit from a checkout with local changes" >&2; exit 1
fi

# Start from a known environment: only the variables below reach the launcher.
unset GEN_KWARGS BATCH_SIZE NUM_PROCESSES EXTRA_MODEL_ARGS MODEL_BACKEND MAX_MODEL_LEN \
      ENABLE_IMAGE_TOKEN_CACHE EXTRA_PYTHONPATH LMMS_EVAL_DEV_PATH
export EVAL_ENVIRONMENT=${EVAL_ENVIRONMENT:-$ROOT/toml/shared/apertus-vllm-release-eval.toml}
export RUN_ID SKIP_PREFLIGHT=1
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600   # vLLM's default (600 s) rejects the longest TED-LIUM talks

# The 16 tasks of the report's audio table. MMAU runs as mmau_test_mini, the
# split the table reports: the mmau group also runs mmau_test, which has no public
# answers, and main's run check rejects a group with an unscored member.
TASKS=librispeech,open_asr_voxpopuli,open_asr_spgispeech,fleurs_en_us,fleurs_de_de,fleurs_fr_fr,fleurs_it_it,fleurs_es_419,fleurs_pl_pl,fleurs_uk_ua,covost2,mmau_test_mini,muchomusic,clotho_aqa,vocalsound_test

# Each task's output cap, passed explicitly: main's launcher otherwise sets
# max_new_tokens=16384 on every job. These are the caps the task definitions
# declare; MuChoMusic and VocalSound declare none and get the model-level 4096.
# TED-LIUM long form declares 256, which truncates its transcripts (up to 4,164
# words), so it gets 4096.
declare -A TASK_CAP=([librispeech]=256 [open_asr_voxpopuli]=4096 [open_asr_spgispeech]=4096
  [fleurs_en_us]=256 [fleurs_de_de]=256 [fleurs_fr_fr]=256 [fleurs_it_it]=256 [fleurs_es_419]=256
  [fleurs_pl_pl]=256 [fleurs_uk_ua]=256 [covost2]=256 [mmau_test_mini]=128 [muchomusic]=4096 [clotho_aqa]=8
  [vocalsound_test]=4096 [tedlium_long_form]=4096)
CAP=max_new_tokens=4096   # model-level fallback, as in the tested runs

case "$SIZE" in
  8b)  CKPT="$CKPT_DIR/Apertus-v1.5-8B"
       export EXTRA_MODEL_ARGS="$CAP"
       PROFILE=(--size 8b) ;;
  # TP=4 with CUDA graphs. The fused all-reduce + RMSNorm pass crashes graph
  # capture at <=128 tokens on this vLLM build, so only that pass is disabled.
  # EXTRA_MODEL_ARGS replaces the 70b profile's own model args, so it repeats
  # tensor_parallel_size=4.
  70b) CKPT="$CKPT_DIR/Apertus-v1.5-70B"
       export EXTRA_MODEL_ARGS="$CAP,tensor_parallel_size=4,compilation_config={\"pass_config\":{\"fuse_allreduce_rms\":false}}"
       PROFILE=(--size 70b --allow-encode) ;;
  *)   echo "size must be 8b or 70b" >&2; exit 2 ;;
esac

# The checkpoint's own tokenizer and chat template. The launcher otherwise falls
# back to the suite's internal Apertus tokenizer, whose template renders audio
# requests with an extra newline after the audio.
export TOKENIZER_PATH="$CKPT" CHAT_TEMPLATE="$CKPT/chat_template.jinja"
[[ -f "$CHAT_TEMPLATE" ]] || { echo "missing $CHAT_TEMPLATE" >&2; exit 1; }

SELECTED=${ONLY:-$TASKS,tedlium_long_form}

# DRY_RUN=1 prints each job's full arguments without submitting; LIMIT=N
# evaluates only the first N samples of each task (for checking the setup).
JOB_ARGS=(); [[ -n "${LIMIT:-}" ]] && JOB_ARGS=(--limit "$LIMIT")
submit() { bash launchers/eval.sh --eval-framework lmms-eval --model "$CKPT" --run-id "$RUN_ID" "${PROFILE[@]}" ${DRY_RUN:+--dry-run} "$@"; }

echo "run $RUN_ID: suite $(git rev-parse --short HEAD), lmms-eval $(git -C third_party/lmms-eval rev-parse --short HEAD)"
for task in ${SELECTED//,/ }; do
  [[ -n "${TASK_CAP[$task]:-}" ]] || { echo "unknown task $task" >&2; exit 2; }
  if [[ "$task" == tedlium_long_form ]]; then
    # The longest talks exceed the encoder cache, which follows
    # max_num_batched_tokens (launcher default 49152).
    submit --tasks "$task" --gen-kwargs "max_new_tokens=${TASK_CAP[$task]}" -- --max-num-batched-tokens 65536 "${JOB_ARGS[@]}"
  else
    submit --tasks "$task" --gen-kwargs "max_new_tokens=${TASK_CAP[$task]}" ${LIMIT:+-- --limit "$LIMIT"}
  fi
done
