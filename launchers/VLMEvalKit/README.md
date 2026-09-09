# VLMEvalKit Launchers

Framework-specific launchers for VLMEvalKit.

- `eval.sh`: Apertus vLLM production submitter copied from the VLMEvalKit submodule and adapted to this repository layout. It expects `ORCH_REPO_ROOT` from `launchers/eval.sh`, reads suites from `task_suites/VLMEvalKit/`, submits `slurm/VLMEvalKit/eval_job.slurm`, writes outputs under `results/VLMEvalKit/<RUN_ID>/`, and writes logs under `logs/VLMEvalKit/<RUN_ID>/`.
