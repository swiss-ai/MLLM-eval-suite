#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

export IMG="apertus-vllm-vision-eval-prod"
export SQSH="${SCRIPT_DIR}/apertus-vllm-vision-eval-prod.sqsh"
export SQSH_DIR="${SCRIPT_DIR}"
export BUILD_CTX="${SCRIPT_DIR}"
# persistent layer cache: /dev/shm storage dies with each job, this survives on
# scratch so unchanged strata (apt, torch, wheels) rebuild in minutes.
export LAYER_CACHE="${LAYER_CACHE:-${SCRIPT_DIR}/build-cache}"
mkdir -p "$LAYER_CACHE"

podman build \
  --layers --cache-to "$LAYER_CACHE" --cache-from "$LAYER_CACHE" \
  -v "$SQSH_DIR/empty.sources.list:/etc/apt/sources.list:ro,z" \
  -v "$SQSH_DIR/my-sources.d:/etc/apt/sources.list.d:ro,z" \
  -v "$SQSH_DIR/99-jfrog-proxy:/etc/apt/apt.conf.d/99-jfrog-proxy:ro,z" \
  --build-arg BUILD_STAMP="$(date -u +%Y-%m-%dT%H:%MZ) dockerfiles@$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  -t "$IMG" \
  -f "$BUILD_CTX/Dockerfile.vllm-multimodal-eval-prod-cu130" \
  "$BUILD_CTX"

# import next to the current image, then rotate: the live sqsh is never
# deleted until its replacement fully exists.
enroot import -o "${SQSH}.new" "podman://$IMG"
[ -f "$SQSH" ] && mv -f "$SQSH" "${SQSH%.sqsh}-old.sqsh"
mv "${SQSH}.new" "$SQSH"
echo "built: $SQSH  (previous kept as ${SQSH%.sqsh}-old.sqsh)"
