# Shared TOML Configs

Store shared configuration fragments or documented conventions used by multiple evaluation frameworks.

- `apertus-vllm-vision-eval-prod.toml`: shared Apertus vLLM runtime config used by VLMEvalKit, Evaluator, and lm-evaluation-harness Slurm wrappers.

## Canary deploys

`apertus-vllm-vision-eval-canary.toml` is identical to prod except the image
points at a local build output (edit its `image =` line to your build path). Test a freshly built image by exporting
`EVAL_ENVIRONMENT=$PWD/toml/shared/apertus-vllm-vision-eval-canary.toml` before
launching; promote to the capstor path only after the canary passes. Judged
benchmarks carry ±2-3pt judge+inference noise at ~500 samples — compare raw
prediction agreement against a reference run, not scores alone.

Fresh-inference controls must bypass the shared response cache: call the
VLMEvalKit launcher directly (`launchers/eval.sh --eval-framework VLMEvalKit`;
the default `all` dispatcher forwards the flag to lmms-eval, which rejects it)
and pass `-- --response-cache <fresh dir>`; the image-token cache follows it (a CLI flag; a
RESPONSE_CACHE env var is silently ignored and the run replays cached
predictions and judge results, agreeing 100% with whatever filled them).

`apertus-vllm-vision-eval-2026-05-torch210.toml` pins the archived May image —
for reproducing pre-Aug-11 results only. Rollback is the `-old.sqsh` kept beside
the live image by each build, not this toml. All three tomls differ from
prod only in `image =`; when editing the shared env block, edit prod and
re-copy it into the variants — drift between them invalidates canary
comparisons.
