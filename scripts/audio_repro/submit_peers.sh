#!/usr/bin/env bash
# Submit an open audio model of the comparison with the same generation setting
# as submit_apertus.sh: each task's declared cap, TED-LIUM long form 4096, greedy.
#
#   PEER_MODEL_DIR=/path/with/model/snapshots \
#     bash scripts/audio_repro/submit_peers.sh <qwen2_audio|qwen2_5_omni|kimi_audio> <run-id> [task,...]
#
# PEER_MODEL_DIR holds `hf download --local-dir` snapshots of
#   Qwen/Qwen2-Audio-7B-Instruct      (revision 0a095220c30b7b31434169c3086508ef3ea5bf0a)
#   Qwen/Qwen2.5-Omni-7B              (revision ae9e1690543ffd5c0221dc27f79834d0294cba00)
#   moonshotai/Kimi-Audio-7B-Instruct (revision 9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b)
# The peers run on Hugging Face generate in the prod eval image. Qwen2.5-Omni also
# needs qwen-omni-utils, which the image lacks; install it (pure Python) into an
# overlay and export its path as EXTRA_PYTHONPATH (the job prepends it):
#   uv pip install --target "$DIR" --no-deps qwen-omni-utils==0.0.8 audioread==3.0.1
# Kimi-Audio runs in the archived 2026-05 image with the overlay built by
# build_kimi_overlay.sbatch on EXTRA_PYTHONPATH.
# Without a task list, the 16 tasks of the report's audio table are submitted.
set -euo pipefail
BACKEND=${1:?qwen2_audio|qwen2_5_omni|kimi_audio}; RUN_ID=${2:?run id}; ONLY=${3:-}
: "${PEER_MODEL_DIR:?set PEER_MODEL_DIR to the directory holding the model snapshots}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
if [[ -n "$(git status --short --ignore-submodules=none)" ]]; then
  echo "refusing to submit from a checkout with local changes" >&2; exit 1
fi

# Start from a known environment: only the variables below reach the launcher.
unset GEN_KWARGS NUM_PROCESSES EXTRA_MODEL_ARGS MAX_MODEL_LEN ENABLE_IMAGE_TOKEN_CACHE \
      EVAL_ENVIRONMENT TOKENIZER_PATH CHAT_TEMPLATE LMMS_EVAL_DEV_PATH
export RUN_ID MODEL_BACKEND="$BACKEND" SKIP_PREFLIGHT=1
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
# The peers run on Hugging Face generate, not vLLM, so the launcher's vLLM batch
# size of 512 does not apply: at 512 Qwen2-Audio runs out of GPU memory and its
# backend returns empty answers for the whole batch. Qwen2-Audio runs at 8 (the
# same outputs as at 1 on 64 LibriSpeech samples); the others at 1.
export BATCH_SIZE=1

ASR=librispeech,open_asr_voxpopuli,open_asr_spgispeech,fleurs_en_us,fleurs_de_de,fleurs_fr_fr,fleurs_it_it,fleurs_es_419,fleurs_pl_pl,fleurs_uk_ua
UNDERSTAND=covost2,mmau,muchomusic,clotho_aqa

case "$BACKEND" in
  qwen2_audio)  MODEL="$PEER_MODEL_DIR/Qwen2-Audio-7B-Instruct"; BATCH_SIZE=8 ;;
  qwen2_5_omni) MODEL="$PEER_MODEL_DIR/Qwen2.5-Omni-7B"
                [[ -f "${EXTRA_PYTHONPATH:-}/qwen_omni_utils/__init__.py" ]] || {
                  echo "set EXTRA_PYTHONPATH to an overlay with qwen-omni-utils (see header)" >&2; exit 1; } ;;
  kimi_audio)   MODEL="$PEER_MODEL_DIR/Kimi-Audio-7B-Instruct"
                [[ -d "${EXTRA_PYTHONPATH:-}/kimia_infer" ]] || {
                  echo "set EXTRA_PYTHONPATH to the overlay from build_kimi_overlay.sbatch" >&2; exit 1; }
                export EVAL_ENVIRONMENT=$ROOT/toml/shared/apertus-vllm-vision-eval-2026-05-torch210.toml ;;
  *) echo "backend must be qwen2_audio, qwen2_5_omni or kimi_audio" >&2; exit 2 ;;
esac

# Each task's output cap, passed explicitly: main's launcher otherwise sets
# max_new_tokens=16384 on every job. These are the caps the task definitions
# declare (TED-LIUM long form raised from 256 to 4096). MuChoMusic and VocalSound
# declare none; they get the backend's own default, as in the tested runs:
# 256 for Qwen2-Audio and Kimi-Audio, 4096 for Qwen2.5-Omni.
case "$BACKEND" in qwen2_5_omni) NOCAP=4096 ;; *) NOCAP=256 ;; esac
declare -A TASK_CAP=([librispeech]=256 [open_asr_voxpopuli]=4096 [open_asr_spgispeech]=4096
  [fleurs_en_us]=256 [fleurs_de_de]=256 [fleurs_fr_fr]=256 [fleurs_it_it]=256 [fleurs_es_419]=256
  [fleurs_pl_pl]=256 [fleurs_uk_ua]=256 [covost2]=256 [mmau]=128 [muchomusic]=$NOCAP [clotho_aqa]=8
  [vocalsound_test]=$NOCAP [tedlium_long_form]=4096)

submit() {  # <comma task group> <model args or empty>: one job per selected task
  local margs=$2 task
  for task in ${1//,/ }; do
    if [[ -n "$ONLY" ]] && ! tr ',' '\n' <<<"$ONLY" | grep -qx "$task"; then continue; fi
    # DRY_RUN=1 prints each job's full arguments without submitting.
    EXTRA_MODEL_ARGS="$margs" bash launchers/eval.sh --eval-framework lmms-eval --model "$MODEL" --run-id "$RUN_ID" \
      ${DRY_RUN:+--dry-run} --tasks "$task" --gen-kwargs "max_new_tokens=${TASK_CAP[$task]}"
  done
}

echo "run $RUN_ID: suite $(git rev-parse --short HEAD), lmms-eval $(git -C third_party/lmms-eval rev-parse --short HEAD)"
if [[ "$BACKEND" == qwen2_5_omni ]]; then
  # The per-category system prompts of the audio results page; speech
  # translation and audio QA are not listed there and use the understanding prompt.
  submit "$ASR"             "system_prompt=You are a speech recognition model."
  # Eager attention (the backend default) runs out of GPU memory on the longest
  # talks; sdpa computes the same attention without the full matrix.
  submit tedlium_long_form  "system_prompt=You are a speech recognition model.,attn_implementation=sdpa"
  submit "$UNDERSTAND"      "system_prompt=You are an audio understanding model."
  submit vocalsound_test    "system_prompt=You are a vocal sound classification model."
else
  submit "$ASR,$UNDERSTAND,vocalsound_test" ""
  BATCH_SIZE=1 submit tedlium_long_form ""   # talks of up to 22 minutes
fi
