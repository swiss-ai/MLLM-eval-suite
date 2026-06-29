# lm-evaluation-harness Launchers

Framework-specific launcher for direct EleutherAI lm-evaluation-harness text runs.

- `eval.sh`: resolves lm-evaluation-harness suites from `task_suites/lm-evaluation-harness/`, submits one `slurm/lm-evaluation-harness/eval_job.slurm` job per task/model pair, and writes under `results/lm-evaluation-harness/<RUN_ID>/` with logs under `logs/lm-evaluation-harness/<RUN_ID>/`.

The only supported backend for this launcher is HuggingFace (`--model hf` inside lm-evaluation-harness). The Slurm job now enables `parallelize=True` by default, so large checkpoints can be sharded across all visible GPUs in a single eval process. If you want model sharding for a checkpoint that does not fit on one GPU, keep `--num-processes 1`; only increase `--num-processes` when you want multiple eval replicas via `accelerate launch`.

You can pass extra generation kwargs with `--gen-kwargs`, and you can pass additional framework-native argv tokens with repeatable `--extra-framework-config`. For example:

```bash
bash launchers/eval.sh \
  --eval-framework lm-evaluation-harness \
  --model /path/to/checkpoint \
  --suite smoke \
  --submit-mode batch \
  --parallelize true \
  --num-processes 1 \
  --gen-kwargs 'temperature=0.6,max_new_tokens=8192,top_p=0.95' \
  --extra-framework-config --model_args \
  --extra-framework-config '{"enable_thinking": true, "chat_template_args": {"reasoning_effort": "low"}}'
```

Use `--gen-kwargs` for sampling/generation controls and `--extra-framework-config` for model/chat-template controls such as `--model_args`, `enable_thinking`, `think_end_token`, or `chat_template_args`. Repeat the flag once per argv token; this keeps JSON values intact without `--` separators.

Example:

```bash
bash launchers/eval.sh \
  --eval-framework lm-evaluation-harness \
  --model /path/to/checkpoint \
  --tasks hellaswag \
  --num-processes 1
```
