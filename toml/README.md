# TOML Configs

Store evaluation runtime configuration files here. Framework-specific configs live in `lmms-eval/` and `VLMEvalKit/`; shared runtime configs live in `shared/`.

Evaluator and lm-evaluation-harness currently reuse `toml/shared/apertus-vllm-vision-eval-prod.toml` for the Slurm/container runtime instead of having framework-specific TOML files.
