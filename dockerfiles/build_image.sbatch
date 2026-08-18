#!/bin/bash
# Submit with SPANK vars cleared, or an outer container session injects
# --environment and the build runs inside the eval image (no podman):
#   for v in $(env | grep -oE '^SLURM_SPANK[A-Za-z_]*'); do unset $v; done
#   sbatch --account=infra01 --reservation=<res> build_image.sbatch
#SBATCH --job-name=build-eval-image
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=build_compute_%j.log
set -euo pipefail
# CSCS-documented build setup: podman storage in /dev/shm (in-memory, slurm-cleaned);
# enroot import must happen in this same allocation before /dev/shm vanishes.
export XDG_CONFIG_HOME=/dev/shm/$USER/xdg-config
# A cancelled build leaves half-written layers in the node's /dev/shm store,
# and the next build on that node fails mid-extract ("File exists" on dirs).
# Every build starts from empty node-local storage; the persistent caches
# (uv downloads, enroot blobs) live elsewhere and are unaffected.
rm -rf /dev/shm/$USER/root /dev/shm/$USER/runroot /dev/shm/$USER/enroot-tmp
mkdir -p "$XDG_CONFIG_HOME/containers" /dev/shm/$USER/{root,runroot}
# Builds run on fuse-overlayfs: rootless podman here does not take the kernel
# overlay path (mount_program="" is treated as unset), and the flashinfer cubin
# layer is the slow extract. The real build-time lever is the persistent uv
# download cache on capstor, not the storage driver.
cat > "$XDG_CONFIG_HOME/containers/storage.conf" <<CONF
[storage]
  driver = "overlay"
  runroot = "/dev/shm/$USER/runroot"
  graphroot = "/dev/shm/$USER/root"
CONF
podman info 2>/dev/null | grep -iE "graphDriverName|Native Overlay|mount_program|Backing Filesystem" || true
# enroot's overlay mount needs a local filesystem (Lustre gives EINVAL), so
# temp stays on /dev/shm; only the mount-free blob cache goes to scratch.
export ENROOT_TEMP_PATH="/dev/shm/$USER/enroot-tmp"
export ENROOT_DATA_PATH="$ENROOT_TEMP_PATH/data"
export ENROOT_CACHE_PATH="$SLURM_SUBMIT_DIR/enroot-tmp/cache"
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"
trap 'rm -rf "$SLURM_SUBMIT_DIR/enroot-tmp" "$SLURM_SUBMIT_DIR/.build-snapshot.sh"' EXIT
cd "$SLURM_SUBMIT_DIR"
# bash reads scripts incrementally; a snapshot makes mid-build edits harmless.
cp generate_docker.sh .build-snapshot.sh; bash .build-snapshot.sh
