#!/usr/bin/env bash
# Run on a CSCS host with Podman and Enroot, including inside an existing allocation.
set -euo pipefail
recipe_dir=${APERTUS_RECIPE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
suite_dir=$(dirname -- "$recipe_dir")
revision=$(sed -n 's/^ARG VLLM_SRC_COMMIT=//p' "$recipe_dir/Dockerfile.vllm-multimodal-eval-prod-cu130")
[[ "$revision" =~ ^[a-f0-9]{40}$ ]] || { echo 'Expected a full vLLM commit pin' >&2; exit 1; }
output_dir=${1:-"$suite_dir/cache/image-builds/vllm-${revision:0:9}"}
mkdir -p "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
if [[ ${APERTUS_BUILD_SNAPSHOT:-} != "$output_dir" ]]; then
    launcher=$(mktemp "$output_dir/launcher.XXXXXX.sh")
    cp -- "${BASH_SOURCE[0]}" "$launcher"
    export APERTUS_RECIPE_DIR="$recipe_dir" APERTUS_BUILD_SNAPSHOT="$output_dir"
    exec bash "$launcher" "$output_dir"
fi
image_path="$output_dir/apertus-vllm-${revision:0:9}-cu130.sqsh"
[[ ! -e "$image_path" ]] || { echo "Refusing to overwrite $image_path" >&2; exit 1; }
for tool in podman enroot flock sha256sum; do
    command -v "$tool" >/dev/null || { echo "Missing host build tool: $tool" >&2; exit 1; }
done
[[ $(uname -m) == aarch64 ]] || { echo 'This recipe targets GH200 / ARM64' >&2; exit 1; }
exec 9>"$output_dir/build.lock"
flock -n 9 || { echo "A build already owns $output_dir" >&2; exit 1; }

context=$(mktemp -d "$output_dir/context.XXXXXX")
cp "$recipe_dir/"{Dockerfile.vllm-multimodal-eval-prod-cu130,runtime-constraints.txt,image_requirements.py,verify_image.py,export_squashfs.sh} "$context/"
(cd "$context" && sha256sum ./* > "$output_dir/source-sha256.txt")
build_stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ) vllm@$revision context=$(basename "$context")"
printf '%s\n' "$build_stamp" > "$output_dir/build-stamp.txt"

# Use a store owned by this output directory and allocation. Never clear a
# user's shared Podman store. Retain layers after failure for a cached retry.
store_key=$(printf '%s' "$output_dir" | sha256sum | cut -c1-12)
store_root="/dev/shm/apertus-build-${UID}-${SLURM_JOB_ID:-local}-$store_key"
# Wheels such as flashinfer-cubin contain tens of thousands of small files.
# Keep their unpacked cache off Lustre; preserve it for retries in this allocation.
mkdir -p "$store_root/"{root,runroot,enroot,uv-cache} "$output_dir/enroot-cache"
export CONTAINERS_STORAGE_CONF="$output_dir/storage.conf"
cat > "$CONTAINERS_STORAGE_CONF" <<CONF
[storage]
driver = "overlay"
runroot = "$store_root/runroot"
graphroot = "$store_root/root"
CONF
export ENROOT_TEMP_PATH="$store_root/enroot"
export ENROOT_DATA_PATH="$store_root/enroot/data"
export ENROOT_CACHE_PATH="$output_dir/enroot-cache"
mkdir -p "$ENROOT_DATA_PATH"
tag="localhost/apertus-vllm-trial:${revision:0:9}-$store_key"
# The launcher owns the lock; container helpers must not retain it after exit.
podman build --platform linux/arm64 --format docker --layers \
    --volume "$store_root/uv-cache:/root/.cache/uv:rw" \
    --build-arg "BUILD_STAMP=$build_stamp" \
    --build-arg "MAX_JOBS=${MAX_JOBS:-8}" \
    --tag "$tag" --file "$context/Dockerfile.vllm-multimodal-eval-prod-cu130" "$context" 9>&-
podman image inspect "$tag" 9>&- > "$output_dir/podman-image.json"
partial_image="$output_dir/.apertus-${revision:0:9}-${BASHPID}.partial.sqsh"
bash "$context/export_squashfs.sh" "podman://$tag" "$partial_image" "$store_root" 9>&-
[[ -s "$partial_image" ]] || { echo 'Image export is empty' >&2; exit 1; }
# Hard-link publication is atomic and fails if the final name already exists.
ln -- "$partial_image" "$image_path"
rm -- "$partial_image"
sha256sum "$image_path" > "$image_path.sha256"
printf 'Built trial image: %s\n' "$image_path"
