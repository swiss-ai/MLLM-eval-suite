#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

export IMG="apertus-vllm-vision-eval-prod"
export SQSH="${SCRIPT_DIR}/apertus-vllm-vision-eval-prod.sqsh"
export SQSH_DIR="${SCRIPT_DIR}"
export BUILD_CTX="${SCRIPT_DIR}"

podman build \
  -v "$SQSH_DIR/empty.sources.list:/etc/apt/sources.list:ro,z" \
  -v "$SQSH_DIR/my-sources.d:/etc/apt/sources.list.d:ro,z" \
  -v "$SQSH_DIR/99-jfrog-proxy:/etc/apt/apt.conf.d/99-jfrog-proxy:ro,z" \
  --build-arg BUILD_STAMP="$(date -u +%Y-%m-%dT%H:%MZ) dockerfiles@$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  -t "$IMG" \
  -f "$BUILD_CTX/Dockerfile.vllm-multimodal-eval-prod-cu130" \
  "$BUILD_CTX"

# import next to the current image, then rotate: the live sqsh is never
# deleted until its replacement fully exists.
# enroot's podman:// handler exits 1 even after a successful import (temp-dir
# cleanup bug), so success is judged by the artifact: the stamp must read back.
# NEVER prune here: `podman builder prune -a` empties the layer store and takes
# the just-built image with it (measured: /dev/shm 55G -> 44M, import then had
# nothing to read). Space is not the constraint — 279G was free when the import
# died with "tar: Unexpected EOF", so capture podman's own stderr instead.
df -h /dev/shm | tail -1
podman images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | head -3
rm -f "${SQSH}.new"
enroot import -o "${SQSH}.new" "podman://$IMG" 2>&1 | tail -40 || echo "enroot import exited $? — verifying artifact"
unsquashfs -cat "${SQSH}.new" /etc/apertus_image_version || { echo "import produced no valid image" >&2; exit 1; }
if [ -f "$SQSH" ]; then mv -f "$SQSH" "${SQSH%.sqsh}-old.sqsh"; fi
mv "${SQSH}.new" "$SQSH"
echo "built: $SQSH  (previous kept as ${SQSH%.sqsh}-old.sqsh)"
