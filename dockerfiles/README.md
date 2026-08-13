# Dockerfiles

Store Dockerfiles and build documentation for evaluation environments. Keep image-specific instructions close to the Dockerfile that needs them.

## Build cache

The uv download cache is bind-mounted from
`/capstor/store/cscs/swissai/infra01/multimodal-eval/MLLM-eval-suite/build-cache/uv`,
so torch, the vLLM wheel and the 1GB flashinfer cubin come off capstor instead
of the network on every build (override with `UV_CACHE_DIR=`). Podman *layer*
caching is still not wired — that needs a registry ref, which `--cache-to` a
directory cannot provide — so every build re-executes all steps.

