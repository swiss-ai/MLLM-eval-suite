# Dockerfiles

Store Dockerfiles and build documentation for evaluation environments. Keep image-specific instructions close to the Dockerfile that needs them.

## Layer cache

Builds cache layers to
`/capstor/store/cscs/swissai/infra01/multimodal-eval/MLLM-eval-suite/build-cache`
(shared, durable; see the README there). First build after a Dockerfile change
pays the delta; later builds finish in minutes. Override with `LAYER_CACHE=`.
Build on a compute node inside the reservation, without `--environment`
(needs host podman): `sbatch --account=infra01 --reservation=<res> build_image.sbatch`.
