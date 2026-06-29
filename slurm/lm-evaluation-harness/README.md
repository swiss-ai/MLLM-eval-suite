# lm-evaluation-harness Slurm

`eval_job.slurm` is the single-task job wrapper used by `launchers/lm-evaluation-harness/eval.sh`.

It runs the HuggingFace backend from `third_party/lm-evaluation-harness` by prepending that checkout to `PYTHONPATH`.

When `--num-processes` is greater than 1, it invokes `accelerate launch --num_processes N -m lm_eval`; otherwise it runs `python -m lm_eval`. Only the HuggingFace backend is supported here for now.
