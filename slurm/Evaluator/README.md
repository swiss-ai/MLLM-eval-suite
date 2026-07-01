# Evaluator Slurm

`eval_job.slurm` is the final entrypoint for one text task. The outer launcher
`launchers/Evaluator/eval.sh` resolves suites and submits one Slurm job per
task.

For Evaluator tasks, the launcher writes one Evaluator template/env pair per
task. This script receives `config.template.yaml`, sources `config.env` to bind
the `${NEL_*}` defaults, applies optional JSON overrides from
`config.overrides.json`, and runs `nel eval run` directly inside the container.

The template uses the existing shared Apertus runtime TOML:

```text
toml/shared/apertus-vllm-vision-eval-prod.toml
```

For interactive use, the outer launcher runs this same script with `bash` on the
current allocation:

```bash
bash launchers/eval.sh --eval-framework Evaluator --suite smoke --submit-mode interactive
```

For batch use, prefer changing Slurm resources through the outer launcher rather
than editing this script. The launcher forwards overrides such as `--account`,
`--reservation`, `--environment`, `--nodes`, `--ntasks-per-node`,
`--cpus-per-task`, `--gpus-per-node`, `--constraint`, and `--time` to `sbatch`.
