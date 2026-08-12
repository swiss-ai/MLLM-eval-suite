# Shared TOML Configs

Store shared configuration fragments or documented conventions used by both evaluation frameworks.

- `apertus-vllm-vision-eval-prod.toml`: combined Apertus vLLM vision evaluation runtime config copied from the VLMEvalKit submodule for launcher use.

## Canary deploys

`apertus-vllm-vision-eval-canary.toml` is identical to prod except the image
points at the iopsstor build output. Test a freshly built image by exporting
`EVAL_ENVIRONMENT=$PWD/toml/shared/apertus-vllm-vision-eval-canary.toml` before
launching; promote to the capstor path only after the canary passes. Judged
benchmarks carry ±2-3pt judge+inference noise at ~500 samples — compare raw
prediction agreement against a reference run, not scores alone.

Fresh-inference controls must bypass the shared response cache: pass
`-- --response-cache <fresh dir>` through the launcher (a CLI flag; a
RESPONSE_CACHE env var is silently ignored and the run replays cached
predictions and judge results, agreeing 100% with whatever filled them).
