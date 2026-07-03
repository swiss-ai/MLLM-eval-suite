#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=healthbench-grader
#SBATCH --environment=/capstor/store/cscs/swissai/infra01/multimodal-eval/MLLM-eval-suite/toml/shared/apertus-vllm-vision-eval-prod.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00

# Serves the HealthBench rubric grader (Qwen3-235B-A22B-Instruct-2507-FP8) as
# an OpenAI-compatible endpoint. Writes host:port to $GRADER_ENDPOINT_FILE so
# eval submissions can discover it; point HEALTHBENCH_GRADER_BASE_URL at
# http://<host>:8080/v1.
set -euo pipefail

MODEL_PATH="${GRADER_MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-235B-A22B-Instruct-2507}"
ENDPOINT_FILE="${GRADER_ENDPOINT_FILE:-${SLURM_SUBMIT_DIR}/cache/grader/healthbench_endpoint}"
SERVED_NAME="${GRADER_SERVED_NAME:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
GRADER_EXTRA_ARGS="${GRADER_EXTRA_ARGS:---quantization fp8}"

export PATH="/opt/venv/bin:${PATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "$(hostname):8080" > "${ENDPOINT_FILE}.pending"

/opt/venv/bin/vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 --port 8080 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  ${GRADER_EXTRA_ARGS:-} &
SERVER_PID=$!

for _ in $(seq 1 540); do
  if curl -sf "http://localhost:8080/v1/models" >/dev/null 2>&1; then
    mv "${ENDPOINT_FILE}.pending" "${ENDPOINT_FILE}"
    echo "grader ready at $(hostname):8080"
    break
  fi
  sleep 10
done
[[ -f "${ENDPOINT_FILE}" ]] || { echo "grader failed to become ready"; kill "${SERVER_PID}"; exit 1; }

wait "${SERVER_PID}"
