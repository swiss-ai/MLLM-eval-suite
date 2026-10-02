#!/usr/bin/env bash
# Build the release-vLLM eval image on a GH200 node and export it as SquashFS.
#   sbatch --account=infra01 --nodes=1 --exclusive --time=04:00:00 \
#     dockerfiles/build_release_eval_image.sh "$PWD" "$PWD/cache/image-builds/release-eval"
set -euo pipefail
suite_dir=${1:?suite dir}
output_dir=${2:?output dir}
mkdir -p "$output_dir"
image_path="$output_dir/apertus-vllm-release-eval.sqsh"
[[ ! -e "$image_path" ]] || { echo "Refusing to overwrite $image_path" >&2; exit 1; }
context=$(mktemp -d "$output_dir/context.XXXXXX")
cp "$suite_dir/dockerfiles/"{Dockerfile.vllm-apertus-release-eval,image_requirements.py,export_squashfs.sh} "$context/"
cp "$suite_dir/third_party/lmms-eval/pyproject.toml" "$context/lmms-pyproject.toml"
: > "$context/empty-vlmeval-requirements.txt"
store_root="/dev/shm/apertus-release-build-${UID}-${SLURM_JOB_ID:-local}"
mkdir -p "$store_root/"{root,runroot,enroot/data,uv-cache}
export CONTAINERS_STORAGE_CONF="$output_dir/storage.conf"
printf '[storage]\ndriver = "overlay"\nrunroot = "%s/runroot"\ngraphroot = "%s/root"\n' "$store_root" "$store_root" > "$CONTAINERS_STORAGE_CONF"
export ENROOT_TEMP_PATH="$store_root/enroot" ENROOT_DATA_PATH="$store_root/enroot/data" ENROOT_CACHE_PATH="$store_root/enroot"
build_stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ) base=ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64"
tag="localhost/apertus-vllm-release-eval:latest"
podman build --platform linux/arm64 --format docker --layers \
  --volume "$store_root/uv-cache:/root/.cache/uv:rw" \
  --build-arg "BUILD_STAMP=$build_stamp" \
  --tag "$tag" --file "$context/Dockerfile.vllm-apertus-release-eval" "$context"
podman image inspect "$tag" > "$output_dir/podman-image.json"
partial="$output_dir/.partial.sqsh"
bash "$context/export_squashfs.sh" "podman://$tag" "$partial" "$store_root"
mv "$partial" "$image_path"
sha256sum "$image_path" > "$image_path.sha256"
echo "Built image: $image_path"
