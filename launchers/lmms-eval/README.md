# lmms-eval Launchers

Framework-specific launchers for lmms-eval.

- `eval.sh`: Apertus vLLM production submitter. It expects `ORCH_REPO_ROOT` from `launchers/eval.sh`, reads suites from `task_suites/lmms-eval/`, submits `slurm/lmms-eval/eval_job.slurm`, writes outputs under `results/lmms-eval/<RUN_ID>/`, writes logs under `logs/lmms-eval/<RUN_ID>/`, and defaults image-token cache files to `cache/lmms-eval/` while using shared runtime caches under `cache/`.
