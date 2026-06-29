# lm-evaluation-harness Suites

Suite files for the direct `third_party/lm-evaluation-harness` launcher.

- `smoke.txt`: one quick likelihood-style task for pipeline checks.
- `text.txt`: common text benchmarks, including loglikelihood/multiple-choice tasks.

The launcher uses the HuggingFace backend. Pass `--num-processes N` after `--` in the top-level dispatcher to evaluate one task with `accelerate launch` across N local processes.
