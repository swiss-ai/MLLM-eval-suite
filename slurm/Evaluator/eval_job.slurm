#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=evaluator-apertus
#SBATCH --environment=/capstor/store/cscs/swissai/infra01/multimodal-eval/MLLM-eval-suite/toml/shared/apertus-vllm-vision-eval-prod.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=04:00:00
#SBATCH --output=logs/Evaluator/evaluator_%j.out
#SBATCH --error=logs/Evaluator/evaluator_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ORCH_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ORCH_REPO_ROOT="${ORCH_REPO_ROOT:-${DEFAULT_ORCH_REPO_ROOT}}"
REPO_DIR="${EVALUATOR_REPO_DIR:-${ORCH_REPO_ROOT}/third_party/Evaluator}"
RUNNER="evaluator"
CONFIG_PATH=""
CONFIG_JSON_FILE=""
ENV_FILE=""
TASK_NAME=""
OUTPUT_DIR=""
LOG_DIR="${ORCH_REPO_ROOT}/logs/Evaluator"
DRY_RUN=0
NEL_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  eval_job.slurm --runner evaluator --config <path> --env-file <path> --output-dir <path> [options] [-- nel eval run args...]

Options:
  --runner <name>       evaluator. Default: evaluator.
  --config <path>       Evaluator YAML template copied by launchers/Evaluator/eval.sh.
  --config-json-file <path>
                        Optional JSON overrides for template variables.
  --env-file <path>     Optional shell env file with NEL_* template variables.
  --task <name>         Task name for logging.
  --output-dir <path>   Output directory for this task.
  --repo-dir <path>     Evaluator checkout. Default: $ORCH_REPO_ROOT/third_party/Evaluator.
  --log-dir <path>      Log directory for this run.
  --extra-framework-config <arg>
                        Extra nel eval run argv token. Repeat for each token.
  --dry-run             Print the final nel command without running it.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runner) RUNNER="$2"; shift 2 ;;
    --config|--config-path) CONFIG_PATH="$2"; shift 2 ;;
    --config-json-file|--config-file) CONFIG_JSON_FILE="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --task|--task-name) TASK_NAME="$2"; shift 2 ;;
    --output-dir|--output-path) OUTPUT_DIR="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --extra-framework-config|--extra-framework-arg) NEL_ARGS+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; NEL_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) NEL_ARGS+=("$1"); shift ;;
  esac
done

case "${RUNNER}" in
  evaluator) ;;
  *) echo "--runner must be evaluator (got: ${RUNNER})" >&2; exit 2 ;;
esac

[[ -n "${OUTPUT_DIR}" ]] || { echo "--output-dir is required" >&2; usage >&2; exit 2; }

case "${OUTPUT_DIR}" in /*) ;; *) OUTPUT_DIR="${ORCH_REPO_ROOT}/${OUTPUT_DIR}" ;; esac
case "${LOG_DIR}" in /*) ;; *) LOG_DIR="${ORCH_REPO_ROOT}/${LOG_DIR}" ;; esac
case "${REPO_DIR}" in /*) ;; *) REPO_DIR="${ORCH_REPO_ROOT}/${REPO_DIR}" ;; esac

[[ -n "${CONFIG_PATH}" ]] || { echo "--config is required" >&2; usage >&2; exit 2; }
case "${CONFIG_PATH}" in /*) ;; *) CONFIG_PATH="${ORCH_REPO_ROOT}/${CONFIG_PATH}" ;; esac
[[ -f "${CONFIG_PATH}" ]] || { echo "config not found: ${CONFIG_PATH}" >&2; exit 1; }
if [[ -n "${ENV_FILE}" ]]; then
  case "${ENV_FILE}" in /*) ;; *) ENV_FILE="${ORCH_REPO_ROOT}/${ENV_FILE}" ;; esac
  [[ -f "${ENV_FILE}" ]] || { echo "env file not found: ${ENV_FILE}" >&2; exit 1; }
fi
if [[ -n "${CONFIG_JSON_FILE}" ]]; then
  case "${CONFIG_JSON_FILE}" in /*) ;; *) CONFIG_JSON_FILE="${ORCH_REPO_ROOT}/${CONFIG_JSON_FILE}" ;; esac
  [[ -f "${CONFIG_JSON_FILE}" ]] || { echo "config JSON file not found: ${CONFIG_JSON_FILE}" >&2; exit 1; }
fi
[[ -d "${REPO_DIR}" ]] || { echo "Evaluator repo not found: ${REPO_DIR}" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export PATH="/opt/venv/bin:${PATH}"
export HF_HOME="${HF_HOME:-${ORCH_REPO_ROOT}/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ORCH_REPO_ROOT}/cache/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${ORCH_REPO_ROOT}/cache/vllm}"
export VLLM_APERTUS_MODELS_CACHE="${VLLM_APERTUS_MODELS_CACHE:-${ORCH_REPO_ROOT}/cache/models}"
export LMMS_EVAL_MODELS_CACHE="${LMMS_EVAL_MODELS_CACHE:-${VLLM_APERTUS_MODELS_CACHE}}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}"

if [[ -n "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

RESOLVED_CONFIG_PATH="${OUTPUT_DIR}/config.resolved.yaml"
CONFIG_PATH="${CONFIG_PATH}" CONFIG_JSON_FILE="${CONFIG_JSON_FILE}" RESOLVED_CONFIG_PATH="${RESOLVED_CONFIG_PATH}" TASK_NAME="${TASK_NAME}" python - <<'PY'
import json
import os
import re
from pathlib import Path

src = Path(os.environ["CONFIG_PATH"])
dst = Path(os.environ["RESOLVED_CONFIG_PATH"])
json_path = os.environ.get("CONFIG_JSON_FILE", "")
task_name = os.environ.get("TASK_NAME", "")
pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}")
prefixes = {
    "apertus": "NEL_APERTUS",
    "benchmark": "NEL_BENCHMARK",
    "evaluator": "NEL_BENCHMARK",
    "harbor": "NEL_HARBOR",
    "openclaw": "NEL_OPENCLAW",
    "output": "NEL_OUTPUT",
    "scoring": "NEL_SCORING",
    "service": "NEL",
    "solver": "NEL_SOLVER",
}
aggregate_names = {
    "NEL_APERTUS_CONTAINER_MOUNTS",
    "NEL_APERTUS_DEPENDS_ON",
    "NEL_APERTUS_HF_OVERRIDES",
    "NEL_APERTUS_SETUP_COMMANDS",
    "NEL_APERTUS_STOP",
    "NEL_BENCHMARK_PARAMS",
    "NEL_BENCHMARK_SANDBOX",
    "NEL_HARBOR_AGENT_KWARGS",
    "NEL_HARBOR_CONTAINER_ENV",
    "NEL_OUTPUT_EXPORT",
    "NEL_OUTPUT_EXPORT_CONFIG",
    "NEL_SCORING_METRICS",
    "NEL_SOLVER_STOP",
}

def scalar_to_template_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)

def env_name(parts: list[str]) -> str:
    if not parts:
        return ""
    first = parts[0].lower().replace("-", "_")
    rest = [part.replace("-", "_") for part in parts[1:]]
    if first.startswith("nel_"):
        tokens = [first.removeprefix("nel_"), *rest]
        return "NEL_" + "_".join(tokens).upper()
    prefix = prefixes.get(first, "NEL_" + first.upper())
    return "_".join([prefix, *rest]).upper()

def flatten(data, parts=None) -> dict[str, str]:
    parts = parts or []
    if not isinstance(data, dict):
        name = env_name(parts)
        return {name: scalar_to_template_value(data)} if name else {}

    values = {}
    for key, value in data.items():
        key = str(key)
        if key == "benchmarks" and not parts:
            continue
        child_parts = [*parts, key]
        name = env_name(child_parts)
        if isinstance(value, dict) and name not in aggregate_names and not key.upper().startswith("NEL_"):
            values.update(flatten(value, child_parts))
        else:
            values[name] = scalar_to_template_value(value)
    return values

def load_config_values() -> dict[str, str]:
    if not json_path:
        return {}
    raw = json.loads(Path(json_path).read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"config JSON must contain an object: {json_path}")

    values = flatten(raw)
    benchmarks = raw.get("benchmarks", {})
    if isinstance(benchmarks, dict) and task_name:
        task_values = benchmarks.get(task_name, benchmarks.get(task_name.replace("_", "-"), {}))
        if isinstance(task_values, dict):
            values.update(flatten(task_values))
    return values

config_values = load_config_values()

def replace(match: re.Match[str]) -> str:
    name = match.group(1)
    default = match.group(2)
    if name in config_values:
        return config_values[name]
    value = os.environ.get(name)
    if value:
        return value
    if default is not None:
        return default
    raise SystemExit(
        f"Environment variable ${{{name}}} is not set and no default was provided in {src}"
    )

dst.write_text(pattern.sub(replace, src.read_text()))
PY

if command -v nel >/dev/null 2>&1; then
  CMD=(nel eval run "${RESOLVED_CONFIG_PATH}" --output-dir "${OUTPUT_DIR}")
else
  export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
  CMD=(python -m nemo_evaluator.cli.main eval run "${RESOLVED_CONFIG_PATH}" --output-dir "${OUTPUT_DIR}")
fi
CMD+=("${NEL_ARGS[@]}")

echo "========== Evaluator launch =========="
echo "RUNNER=${RUNNER}"
echo "ORCH_REPO_ROOT=${ORCH_REPO_ROOT}"
echo "REPO_DIR=${REPO_DIR}"
echo "CONFIG_PATH=${CONFIG_PATH:-<none>}"
echo "CONFIG_JSON_FILE=${CONFIG_JSON_FILE:-<none>}"
echo "RESOLVED_CONFIG_PATH=${RESOLVED_CONFIG_PATH:-<none>}"
echo "ENV_FILE=${ENV_FILE:-<none>}"
echo "TASK_NAME=${TASK_NAME:-<none>}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "VLLM_APERTUS_MODELS_CACHE=${VLLM_APERTUS_MODELS_CACHE}"
printf 'CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "======================================"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

cd "${REPO_DIR}"
"${CMD[@]}"
