#!/usr/bin/env bash
# Submit the peer audio baselines with the same generation setting as
# submit_apertus.sh (each task's declared cap, TED-LIUM 4096, greedy).
#
#   PEER_MODEL_DIR=/path/with/model/snapshots \
#     bash scripts/audio_repro/submit_peers.sh <qwen2_audio|qwen2_5_omni|kimi_audio> <run-id>
#
# PEER_MODEL_DIR holds `hf download --local-dir` snapshots of
#   Qwen/Qwen2-Audio-7B-Instruct  (revision 0a095220c30b7b31434169c3086508ef3ea5bf0a)
#   Qwen/Qwen2.5-Omni-7B          (revision ae9e1690543ffd5c0221dc27f79834d0294cba00)
#   moonshotai/Kimi-Audio-7B-Instruct (revision 9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b)
# Peers run in the default prod image (Hugging Face backends). Qwen2.5-Omni also
# needs qwen-omni-utils, which the image lacks; install it (pure Python) into an
# overlay and export its path as EXTRA_PYTHONPATH (the job script prepends it):
#   uv pip install --target "$DIR" --no-deps qwen-omni-utils==0.0.8 audioread==3.0.1
#   export EXTRA_PYTHONPATH=$DIR
# Kimi-Audio runs in the archived 2026-05 image with the overlay built by
# build_kimi_overlay.sbatch on EXTRA_PYTHONPATH.
# Qwen2.5-Omni uses the per-category system prompts listed on the audio
# results page; speech translation and audio QA are not listed there and use
# the audio-understanding prompt.
set -euo pipefail
BACKEND=${1:?qwen2_audio|qwen2_5_omni|kimi_audio}; RUN_ID=${2:?run id}
: "${PEER_MODEL_DIR:?set PEER_MODEL_DIR to the directory holding the model snapshots}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
if [[ -n "$(git status --short --ignore-submodules=none)" ]]; then
  echo "refusing to submit from a checkout with local changes" >&2; exit 1
fi
export SKIP_PREFLIGHT=1 VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
unset EVAL_ENVIRONMENT GEN_KWARGS BATCH_SIZE NUM_PROCESSES GPU_MEMORY_UTILIZATION EXTRA_MODEL_ARGS

ASR=librispeech,open_asr_voxpopuli,open_asr_spgispeech,fleurs_en_us,fleurs_de_de,fleurs_fr_fr,fleurs_it_it,fleurs_es_419,fleurs_pl_pl,fleurs_uk_ua
UNDERSTAND=covost2,mmau,muchomusic,clotho_aqa
VOICE=voicebench_advbench,voicebench_bbh,voicebench_ifeval,voicebench_mmsu,voicebench_openbookqa,mmsu

case "$BACKEND" in
  qwen2_audio)  MODEL="$PEER_MODEL_DIR/Qwen2-Audio-7B-Instruct" ;;
  qwen2_5_omni) MODEL="$PEER_MODEL_DIR/Qwen2.5-Omni-7B"
                [[ -f "${EXTRA_PYTHONPATH:-}/qwen_omni_utils/__init__.py" ]] || {
                  echo "set EXTRA_PYTHONPATH to an overlay with qwen-omni-utils (see header)" >&2; exit 1; } ;;
  kimi_audio)   MODEL="$PEER_MODEL_DIR/Kimi-Audio-7B-Instruct"
                [[ -d "${EXTRA_PYTHONPATH:-}/kimia_infer" ]] || {
                  echo "set EXTRA_PYTHONPATH to the overlay from build_kimi_overlay.sbatch" >&2; exit 1; }
                export EVAL_ENVIRONMENT=$ROOT/toml/shared/apertus-vllm-vision-eval-2026-05-torch210.toml ;;
  *) echo "backend must be qwen2_audio, qwen2_5_omni or kimi_audio" >&2; exit 2 ;;
esac

submit() {  # <tasks> <system prompt or empty> [launcher args...]
  local tasks=$1 prompt=$2; shift 2
  local extra=()
  [[ -n "$prompt" ]] && extra=(--extra-model-args "system_prompt=$prompt")
  bash launchers/eval.sh --eval-framework lmms-eval --model "$MODEL" --tasks "$tasks" \
    --enable-image-token-cache false --run-id "$RUN_ID" "${extra[@]}" "$@" -- --model-backend "$BACKEND"
}

echo "run $RUN_ID: suite $(git rev-parse --short HEAD), lmms-eval $(git -C third_party/lmms-eval rev-parse --short HEAD)"
if [[ "$BACKEND" == qwen2_5_omni ]]; then
  submit "$ASR"            "You are a speech recognition model."
  submit tedlium_long_form "You are a speech recognition model." --gen-kwargs max_new_tokens=4096
  submit "$UNDERSTAND"     "You are an audio understanding model."
  submit vocalsound_test   "You are a vocal sound classification model."
  submit "$VOICE"          "You are a helpful voice assistant."
else
  submit "$ASR,$UNDERSTAND,vocalsound_test,$VOICE" ""
  submit tedlium_long_form "" --gen-kwargs max_new_tokens=4096
fi
