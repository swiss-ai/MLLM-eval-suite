# Resolve the same dataset locations and model checks before submission and in jobs.
resolve_lmms_dataset_env() {
  local root="${RS_DATASETS_ROOT:-${ORCH_REPO_ROOT:?ORCH_REPO_ROOT must be set}/cache/rs_datasets}"
  export VRSBENCH_DIR="${VRSBENCH_DIR:-${root}/vrsbench}"
  export GEOBENCH_DIR="${GEOBENCH_DIR:-${root}/geobench}"
  export FRIEDA_DIR="${FRIEDA_DIR:-${root}/frieda}"
  export BIGEARTH_S2_DIR="${BIGEARTH_S2_DIR:-${root}/bigearth/BigEarthNet-S2}"
}

lmms_model_preflight_args() {
  LMMS_MODEL_PREFLIGHT_ARGS=()
  if [[ "${1:-apertus_1p5_vllm}" != apertus* ]]; then
    # These backends load their own model/tokenizer, including Hub identifiers.
    LMMS_MODEL_PREFLIGHT_ARGS+=(--skip-model)
  fi
}
