#!/bin/bash
#SBATCH --job-name=build-eval-image
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=build_compute_%j.log
set -euo pipefail
# CSCS-documented build setup: podman storage in /dev/shm (in-memory, slurm-cleaned);
# enroot import must happen in this same allocation before /dev/shm vanishes.
export XDG_CONFIG_HOME=/dev/shm/$USER/xdg-config
mkdir -p "$XDG_CONFIG_HOME/containers" /dev/shm/$USER/{root,runroot}
cat > "$XDG_CONFIG_HOME/containers/storage.conf" <<CONF
[storage]
  driver = "overlay"
  runroot = "/dev/shm/$USER/runroot"
  graphroot = "/dev/shm/$USER/root"
CONF
# enroot's overlay mount needs a local filesystem (Lustre gives EINVAL), so
# temp stays on /dev/shm; only the mount-free blob cache goes to scratch.
export ENROOT_TEMP_PATH="/dev/shm/$USER/enroot-tmp"
export ENROOT_DATA_PATH="/dev/shm/$USER/enroot-tmp/data"
export ENROOT_CACHE_PATH="$SLURM_SUBMIT_DIR/enroot-tmp/cache"
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"
trap 'rm -rf "$SLURM_SUBMIT_DIR/enroot-tmp"' EXIT
cd "$SLURM_SUBMIT_DIR"
cp generate_docker.sh .build-snapshot.sh; bash .build-snapshot.sh
