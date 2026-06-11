SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

export IMG="apertus-vllm-vision-eval-prod"
export SQSH="${SCRIPT_DIR}/apertus-vllm-vision-eval-prod.sqsh"
export SQSH_DIR="${SCRIPT_DIR}"
export BUILD_CTX="${SCRIPT_DIR}"

podman build \
  -v "$SQSH_DIR/empty.sources.list:/etc/apt/sources.list:ro,z" \
  -v "$SQSH_DIR/my-sources.d:/etc/apt/sources.list.d:ro,z" \
  -v "$SQSH_DIR/99-jfrog-proxy:/etc/apt/apt.conf.d/99-jfrog-proxy:ro,z" \
  -t "$IMG" \
  -f "$BUILD_CTX/Dockerfile.vllm-multimodal-eval-prod-cu130" \
  "$BUILD_CTX" \
  && rm -f "$SQSH" \
  && enroot import -o "$SQSH" "podman://$IMG"
