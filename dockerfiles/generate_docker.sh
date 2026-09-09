#!/usr/bin/env bash
# Compatibility entrypoint; every image now uses the snapshot builder.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export APERTUS_APT_CONFIG_DIR="${APERTUS_APT_CONFIG_DIR:-$recipe_dir}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/capstor/store/cscs/swissai/infra01/multimodal-eval/MLLM-eval-suite/build-cache/uv}"
exec bash "$recipe_dir/build_trial_image.sh" "$@"
