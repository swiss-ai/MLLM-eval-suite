# Launchers

Shell entrypoints for running evaluations through a common interface. Framework-specific launchers live next to the top-level dispatcher.

- `eval.sh`: combined production dispatcher. Pass `--eval-framework lmms-eval` or `--eval-framework VLMEvalKit`; the dispatcher sets `ORCH_REPO_ROOT`, accepts common args such as `--model`, `--tasks`, `--suite`, `--mode`, and `--run-id`, and calls the selected framework launcher. Put framework-specific args after `--`.
- `lmms-eval/eval.sh`: lmms-eval Apertus vLLM production submitter copied from the lmms-eval submodule and adapted to this repository layout. It submits `slurm/lmms-eval/eval_job.slurm`, reads suites from `task_suites/lmms-eval/`, and uses `cache/lmms-eval/`, shared runtime caches under `cache/`, `logs/lmms-eval/`, and `results/lmms-eval/`.
- `VLMEvalKit/eval.sh`: VLMEvalKit Apertus vLLM production submitter copied from the VLMEvalKit submodule and adapted to this repository layout. It submits `slurm/VLMEvalKit/eval_job.slurm`, reads suites from `task_suites/VLMEvalKit/`, and uses `cache/VLMEvalKit/`, shared runtime caches under `cache/`, `logs/VLMEvalKit/`, and `results/VLMEvalKit/`.

Each framework launcher creates one `RUN_ID` directory per invocation and passes that shared output/log root to all Slurm jobs from the call.
