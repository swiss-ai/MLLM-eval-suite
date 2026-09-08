# Cross-lane job environment: the parts every harness needs identically.
# Sourced by each slurm/*/eval_job.slurm after HF_HOME is chosen; the caller
# passes its harness checkout so the suite paths land behind it.
#
#   suite_pythonpath <harness-checkout> [extra-path ...]
#   resolve_hf_token
#   resolve_judge_key   (only when API_TYPE=openai)
#   suite_preflight <framework> <task> <harness-dir> [preflight-arg ...]
#   suite_run <out-dir> <framework> <task> <harness-dir> <run-id> [preflight-arg ...] -- [manifest-arg ...] -- <cmd ...>

suite_pythonpath() {
  local _root="${ORCH_REPO_ROOT:?ORCH_REPO_ROOT must be set}"
  local _paths=("$@" "${_root}/shared" "${_root}/shared/_vendor")
  local IFS=:
  export PYTHONPATH="${_paths[*]}${PYTHONPATH:+:${PYTHONPATH}}"
  # The container image sets its own PYTHONPATH, so submission-env values are
  # lost; EXTRA_PYTHONPATH survives and is prepended (e.g. a staged package the
  # container build predates).
  if [[ -n "${EXTRA_PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${EXTRA_PYTHONPATH}:${PYTHONPATH}"
  fi
}

# Submission-env tokens don't survive into the container; the file under the
# repo-mounted HF_HOME does.
resolve_hf_token() {
  if [[ -z "${HF_TOKEN:-}" && -f "${HF_HOME:?HF_HOME must be set}/token" ]]; then
    export HF_TOKEN="$(<"${HF_HOME}/token")"
  fi
}

# Tasks that build their own OpenAI client (charxiv) read these directly and
# fall back to placeholder strings, so an unset key spends the whole run
# retrying a bogus endpoint.
resolve_judge_key() {
  [[ "${API_TYPE:-dummy}" == "openai" ]] || return 0
  local _default_env="${ORCH_REPO_ROOT}/.judge.env"
  local _file="${JUDGE_ENV_FILE:-${_default_env}}"
  [[ -f "$_file" ]] || _file="${ORCH_REPO_ROOT}/third_party/VLMEvalKit/.env"
  if [[ -z "${OPENAI_API_KEY:-}" && -f "$_file" ]]; then
    export OPENAI_API_KEY="$(sed -n 's/^OPENAI_API_KEY=//p' "$_file" | head -1)"
  fi
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
  export MODEL_VERSION="${MODEL_VERSION:-gpt-4o-mini}"
  # Several tasks build their own client from a task-specific variable and
  # fail their whole judge stage when it is unset (babyvision returned a 100%
  # grader-failure rate this way). Fan the one key out to those names.
  local _alias
  for _alias in BABYVISION VIESCORE WISE STRUCTEDITBENCH MEGABENCH_OPEN; do
    export "${_alias}_API_KEY=${OPENAI_API_KEY}"
  done
  export BABYVISION_BASE_URL="${BABYVISION_BASE_URL:-${OPENAI_BASE_URL}}"
  [[ -n "${OPENAI_API_KEY:-}" ]] || {
    echo "ERROR: API_TYPE=openai but no OPENAI_API_KEY (looked in ${_file})" >&2
    exit 1
  }
}


# Refuse the launch unless suite/preflight.py passes (SKIP_PREFLIGHT=1 bypasses).
suite_preflight() {
  local framework="$1" task="$2" harness_dir="$3"
  shift 3
  [[ "${SKIP_PREFLIGHT:-0}" == "1" ]] && return 0
  export PYTHONPATH="${ORCH_REPO_ROOT:?ORCH_REPO_ROOT must be set}${PYTHONPATH:+:${PYTHONPATH}}"
  "${SUITE_PY:-/opt/venv/bin/python}" -m suite.preflight --framework "${framework}" --tasks "${task}" \
    --harness-root "${harness_dir}" "$@" || exit 2
}

# The one job-side contract for every harness: preflight, write run_meta.json
# before inference, run the harness, finalize the manifest from its logs and
# results, and exit with the manifest status (0 ok, 3 failed, 4 invalid).
# Arguments before the first "--" go to suite.preflight, those between the two
# "--" to suite.manifest start; everything after the second "--" is the harness
# command.
suite_run() {
  local out_dir="$1" framework="$2" task="$3" harness_dir="$4" run_id="$5"
  shift 5
  local -a preflight_args=() manifest_args=() cmd=()
  while (( $# )) && [[ "$1" != "--" ]]; do preflight_args+=("$1"); shift; done
  shift
  while (( $# )) && [[ "$1" != "--" ]]; do manifest_args+=("$1"); shift; done
  shift
  cmd=("$@")
  local py="${SUITE_PY:-/opt/venv/bin/python}"
  suite_preflight "${framework}" "${task}" "${harness_dir}" "${preflight_args[@]}"
  export PYTHONPATH="${ORCH_REPO_ROOT:?ORCH_REPO_ROOT must be set}${PYTHONPATH:+:${PYTHONPATH}}"
  "${py}" -m suite.manifest start --out "${out_dir}" --framework "${framework}" --task "${task}" \
    --run-id "${run_id}" --harness-dir "${harness_dir}" "${manifest_args[@]}"
  set +e
  "${cmd[@]}"
  local harness_rc=$?
  set -e
  local -a logs=()
  local pattern
  for pattern in "${SUITE_JOB_OUTPUT:-}" "${SUITE_JOB_ERROR:-}"; do
    [[ -n "${pattern}" ]] && logs+=(--log "${pattern//%j/${SLURM_JOB_ID:-interactive}}")
  done
  set +e
  "${py}" -m suite.manifest finalize --manifest "${out_dir}/run_meta.json" "${logs[@]}" \
    --results-dir "${out_dir}" --harness-rc "${harness_rc}"
  local status=$?
  set -e
  exit "${status}"
}
