# Cross-lane job environment: the parts every harness needs identically.
# Sourced by each slurm/*/eval_job.slurm after HF_HOME is chosen; the caller
# passes its harness checkout so the suite paths land behind it.
#
#   suite_pythonpath <harness-checkout> [extra-path ...]
#   resolve_hf_token
#   resolve_judge_key   (only when API_TYPE=openai)

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
  [[ -n "${OPENAI_API_KEY:-}" ]] || {
    echo "ERROR: API_TYPE=openai but no OPENAI_API_KEY (looked in ${_file})" >&2
    exit 1
  }
}
