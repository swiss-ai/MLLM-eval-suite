#!/bin/bash
# Build the Kimi-Audio Python overlay used by submit_peers.sh kimi_audio.
#
#   sbatch --account=infra01 \
#     --environment=$PWD/toml/shared/apertus-vllm-vision-eval-2026-05-torch210.toml \
#     scripts/audio_repro/build_kimi_overlay.sbatch <overlay dir>
#
# Kimi-Audio's inference package (kimia_infer) imports flash-attn
# unconditionally and targets Transformers 4.x, so it runs in the archived
# 2026-05 eval image (Transformers 4.57) with flash-attn compiled for GH200
# (sm90). Everything goes into one directory that the job puts on
# EXTRA_PYTHONPATH; the image itself is unchanged. deepspeed (training) and
# decord (video; no ARM64 build) from Kimi's requirements are not needed.
#SBATCH --job-name=kimi-overlay-build
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=kimi_overlay_build_%j.log
set -euo pipefail
OUT=${1:?overlay directory}
KIMI_COMMIT=349251e1d8f4f98d58fda59246381faecd7392e0   # MoonshotAI/Kimi-Audio (with its glm4 tokenizer submodule)
mkdir -p "$OUT"
SRC=$(mktemp -d "$OUT/../kimi-src.XXXXXX")
git clone -q https://github.com/MoonshotAI/Kimi-Audio "$SRC"
git -C "$SRC" checkout -q "$KIMI_COMMIT"
git -C "$SRC" submodule update -q --init --recursive

export PATH=/opt/venv/bin:$PATH MAX_JOBS=32 TORCH_CUDA_ARCH_LIST=9.0 FLASH_ATTN_CUDA_ARCHS=90
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
pip install --no-build-isolation --no-deps --target "$OUT" "flash-attn==2.7.4.post1"
pip install --no-deps --target "$OUT" "$SRC"
pip install --no-deps --target "$OUT" openai-whisper hyperpyyaml conformer==0.3.2 torchdyn==1.0.6 \
  easydict fire ujson wget gdown blobfile sox loguru jsonlines
PYTHONPATH="$OUT:${PYTHONPATH:-}" python -c "import flash_attn; from kimia_infer.api.kimia import KimiAudio; print('kimi overlay ok', flash_attn.__version__)"
echo "Kimi-Audio $(git -C "$SRC" rev-parse HEAD)" > "$OUT/BUILD_INFO"
rm -rf "$SRC"
