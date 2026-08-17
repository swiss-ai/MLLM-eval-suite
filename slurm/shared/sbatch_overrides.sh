# Single source of truth for the cluster-account contract, so a clone can
# submit without editing any template. Sourced by both framework launchers;
# for a direct submission:  source slurm/shared/sbatch_overrides.sh &&
# sbatch "${SBATCH_OVERRIDES[@]}" slurm/<fw>/eval_job.slurm ...
#
#   EVAL_ACCOUNT      slurm account                     (default: infra01)
#   EVAL_RESERVATION  reservation; set EVAL_RESERVATION= (empty) to submit
#                     without one                        (default: SD-69241-apertus-1-5-0)
#   EVAL_ENVIRONMENT  pyxis EDF toml                     (default: this repo's toml/shared/)

_EVAL_ROOT="${ORCH_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)}"

EVAL_ACCOUNT="${EVAL_ACCOUNT:-infra01}"
EVAL_RESERVATION="${EVAL_RESERVATION-SD-69241-apertus-1-5-0}"
EVAL_ENVIRONMENT="${EVAL_ENVIRONMENT:-${_EVAL_ROOT}/toml/shared/apertus-vllm-vision-eval-prod.toml}"

[[ -f "${EVAL_ENVIRONMENT}" ]] || {
  echo "EVAL_ENVIRONMENT toml not found: ${EVAL_ENVIRONMENT}" >&2
  return 1 2>/dev/null || exit 1
}

SBATCH_OVERRIDES=(--account="${EVAL_ACCOUNT}" --environment="${EVAL_ENVIRONMENT}")
if [[ -n "${EVAL_RESERVATION}" ]]; then
  SBATCH_OVERRIDES+=(--reservation="${EVAL_RESERVATION}")
fi
