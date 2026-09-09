# Evaluator Config Templates

`apertus_text_vllm_template.yaml` is the generic fallback template for NeMo
Evaluator jobs. It exposes model service, generation, benchmark, solver,
scoring, sandbox, and output fields through `${NEL_*}` placeholders.

`templates/` contains one template per native benchmark registered in the
checked-out NeMo Evaluator submodule. The Evaluator launcher automatically uses
`templates/<benchmark>.yaml` when it exists, then copies the selected template
into the task result directory as `config.template.yaml`.

Use an explicit template for all tasks with:

```bash
bash launchers/eval.sh --eval-framework Evaluator \
  --tasks gsm8k \
  --config-template configs/Evaluator/apertus_text_vllm_template.yaml
```

Use an alternate per-benchmark template directory with:

```bash
bash launchers/eval.sh --eval-framework Evaluator \
  --suite text-builtin \
  --template-dir /path/to/my/templates
```

The text benchmarks use the simple solver by default. `humaneval` defaults to a
Docker Python sandbox. `nmp_harbor`, `pinchbench`, and `terminal-bench-*`
templates expose Harbor/OpenClaw and sandbox fields, but they still require the
matching external agent/runtime prerequisites before they can run successfully.

Prefer JSON overrides for run-specific values instead of adding more launcher
flags. The launcher copies the selected template to each task output directory,
writes generated defaults to `config.env`, copies the user JSON to
`config.overrides.json`, then `slurm/Evaluator/eval_job.slurm` renders
`config.resolved.yaml` before invoking `nel eval run`.

```bash
bash launchers/eval.sh --eval-framework Evaluator \
  --tasks humaneval \
  --config-file configs/Evaluator/my_run.json
```

JSON keys can use exact `NEL_*` names or grouped names. Shared values apply to
every task; values under `benchmarks.<task>` override only that benchmark:

```json
{
  "apertus": {
    "max_tokens": 1024,
    "temperature": 0.0
  },
  "benchmark": {
    "max_problems": 100,
    "timeout": 3600
  },
  "benchmarks": {
    "humaneval": {
      "benchmark": {
        "sandbox": {
          "type": "docker",
          "image": "python:3.12-slim",
          "timeout": 120.0
        },
        "params": {
          "num_examples": 20
        }
      }
    }
  }
}
```

Legacy launcher flags such as `--max-problems`, `--benchmark-sandbox`, and
`--harbor-agent` are still accepted for compatibility. For fields that are
easier to express as native NeMo overrides, keep using repeatable
`--extra-framework-config`, for example:

```bash
bash launchers/eval.sh --eval-framework Evaluator --tasks gsm8k \
  --extra-framework-config -O \
  --extra-framework-config benchmarks.0.timeout=3600
```
