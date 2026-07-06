# Slurm

Store reusable Slurm templates for framework-specific batch jobs and shared environment setup. Keep account, partition, image, and model paths configurable.

- `lmms-eval/eval_job.slurm`: lmms-eval Apertus vLLM production job template copied from the lmms-eval submodule and adapted to this repository layout. The container environment comes from the launcher via `slurm/shared/sbatch_overrides.sh` (default `toml/shared/apertus-vllm-vision-eval-prod.toml`); it uses `cache/lmms-eval/` for image-token cache, shared runtime caches under `cache/`, and `results/lmms-eval/`.
- `VLMEvalKit/eval_job.slurm`: VLMEvalKit Apertus vLLM production job template copied from the VLMEvalKit submodule and adapted to this repository layout. The container environment comes from the launcher via `slurm/shared/sbatch_overrides.sh` (default `toml/shared/apertus-vllm-vision-eval-prod.toml`); it uses `cache/VLMEvalKit/` for framework cache, shared runtime caches under `cache/`, and `results/VLMEvalKit/`.
