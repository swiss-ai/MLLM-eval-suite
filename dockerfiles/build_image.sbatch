#!/bin/bash
# Submit from the suite root, with every inherited *SPANK* variable cleared.
# A leading _SLURM_SPANK_OPTION_pyxis_environment also causes nested containers.
#   sbatch --account=infra01 dockerfiles/build_image.sbatch /absolute/suite /absolute/output
#SBATCH --job-name=build-eval-image
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=build_compute_%j.log
set -euo pipefail
suite_dir=${1:-${SLURM_SUBMIT_DIR:?}}
output_dir=${2:-"$suite_dir/cache/image-builds/vllm-trial-$SLURM_JOB_ID"}
exec bash "$suite_dir/dockerfiles/build_trial_image.sh" "$output_dir"
