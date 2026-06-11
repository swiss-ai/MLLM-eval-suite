# VLMEvalKit Launchers

Framework-specific launchers for VLMEvalKit.

- `eval.sh`: Apertus vLLM production submitter copied from the VLMEvalKit submodule and adapted to this repository layout. It expects `ORCH_REPO_ROOT` from `launchers/eval.sh`, reads suites from `task_suites/VLMEvalKit/`, submits `slurm/VLMEvalKit/eval_job.slurm`, and defaults VLMEvalKit cache files to `cache/VLMEvalKit/` while using shared runtime caches under `cache/`.
