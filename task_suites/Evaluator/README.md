# Evaluator Suites

Suite files for NeMo Evaluator. Entries are benchmark names accepted by
`nel eval run` configs.

- `smoke.txt`: native Evaluator text smoke suite (`gsm8k`).
- `text_builtin.txt`: native text-oriented Evaluator benchmarks.
- `all_builtin.txt`: all native benchmarks registered in this Evaluator checkout,
  including sandbox-heavy code and agentic benchmarks.

Use `task_suites/lm-evaluation-harness/` for direct EleutherAI lm-evaluation-harness suites, especially loglikelihood and multiple-choice tasks.
