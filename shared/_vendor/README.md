# Transitional vendored dependencies

These are pure-python packages the running image predates, staged here with
`pip install --target` and put on PYTHONPATH by `slurm/shared/job_env.sh` so a
fleet already in flight did not have to wait for an image rebuild.

| package | needed by |
|---|---|
| `emoji`, `syllapy` | `shared/ifbench_verifiers/instructions.py` |
| `func_timeout` | VLMEvalKit `OmniDocBench`, lmms-eval `mdpbench` |
| `langdetect`, `immutabledict` | lm-eval `ifeval` |
| `rouge_score`, `absl-py` | lm-eval `truthfulqa_gen`, `math_verify` family |

**This directory is temporary.** All three are installed by
`dockerfiles/Dockerfile.vllm-multimodal-eval-prod-cu130`. Once an image built
from that Dockerfile is rolled out, delete this directory and drop
`shared/_vendor` from `suite_pythonpath` in `slurm/shared/job_env.sh`.

Until then it shadows site-packages for every lane, so a newer version in the
image would be silently overridden by the copy here. Nothing else belongs in
this directory: new dependencies go in the Dockerfile.
