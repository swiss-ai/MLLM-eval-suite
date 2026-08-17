# Single source of truth for the Apertus image-token cache env contract, shared
# by the lmms-eval and VLMEvalKit slurm templates. Source it after deriving
# IMAGE_TOKEN_CACHE_DIR (harness-specific) and choosing IMAGE_TOKEN_CACHE_MODE.
#
# Exports exactly the eight vars the vLLM engine's cache config reads
# (apertus_image_token_cache/config.py); nothing else is plumbed. The mode is
# the only knob:
#   fill     build/extend the cache: writable, write missing tokens, lenient.
#   readonly use a sealed prebuilt cache: read-only, no writes, and STRICT so a
#            missing DB fails fast instead of silently degrading to a soft miss.

case "${IMAGE_TOKEN_CACHE_MODE:-fill}" in
  fill)
    _cache_readonly=0
    _cache_write_misses=1
    _cache_strict=0
    ;;
  readonly)
    _cache_readonly=1
    _cache_write_misses=0
    _cache_strict=1
    ;;
  *)
    echo "image-token-cache mode must be fill or readonly (got: ${IMAGE_TOKEN_CACHE_MODE})" >&2
    exit 1
    ;;
esac

mkdir -p "${IMAGE_TOKEN_CACHE_DIR}"
export VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR="${IMAGE_TOKEN_CACHE_DIR}"
export VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY="${_cache_readonly}"
export VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES="${_cache_write_misses}"
export VLLM_APERTUS_IMAGE_TOKEN_CACHE_STRICT="${_cache_strict}"
# Tuning knobs default to the engine's own defaults; override via env if ever needed.
export VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE="${VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE:-131072}"
export VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS="${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS:-5000}"
export VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE="${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE:-1073741824}"
export VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG:-0}"
