# MLLM Evaluation Suite

## Overview

This repository is a unified orchestration layer for evaluating models with VLMEvalKit, lmms-eval, NeMo Evaluator, and EleutherAI lm-evaluation-harness.

It does not merge, fork, or reimplement the evaluation frameworks. Instead, frameworks are kept as pinned Git submodules under `third_party/`, while this repository owns the shared launcher interface, configuration layout, container metadata, Slurm templates, logs, results, and post-processing utilities.

## Goals

- Provide a common launcher interface for multiple evaluation frameworks.
- Make evaluation runs reproducible through explicit configs, task lists, and metadata.
- Centralize TOML configs for framework-specific and shared settings.
- Manage Dockerfiles used to build or document evaluation environments.
- Provide Slurm job templates for batch execution.
- Provide launcher-level support for managed/offline vLLM and OpenAI-compatible local endpoints where the selected framework supports them.
- Keep logs and results in structured, predictable locations.
- Normalize and compare outputs across evaluation tools where possible.

## Repository Structure

- `third_party/`: Git submodules for upstream evaluation frameworks.
- `dockerfiles/`: Dockerfile definitions and build documentation.
- `toml/`: Centralized TOML configuration files, split by framework plus shared settings.
- `launchers/`: Shell entrypoints for interactive or Slurm-submitted evaluation runs.
- `slurm/`: Slurm templates and shared Slurm environment snippets.
- `task_suites/`: Suite files for lmms-eval, VLMEvalKit, Evaluator, and lm-evaluation-harness. Pass these paths directly to `--tasks` or select them by name with `--suite`.
- `cache/`: Local cache root. Framework data caches are split under framework-specific folders where needed; shared runtime caches use common folders such as `cache/hf`, `cache/nltk_data`, `cache/xdg`, `cache/vllm`, and `cache/models`. Generated contents are ignored.
- `results/`: Evaluation outputs, separated by framework. Generated contents are ignored.
- `logs/`: Runtime logs, separated by framework. Generated contents are ignored.
- `scripts/`: Utility scripts for result normalization, comparison, and log collection.

## Submodules

Initialize all submodules after cloning:

```bash
git submodule update --init --recursive
```

Update submodules to their configured branch tips:

```bash
git submodule update --remote --merge
```

The submodules are configured as branch-tracking submodules:

- `third_party/lmms-eval`: [github.com/swiss-ai/lmms-eval](https://github.com/swiss-ai/lmms-eval), branch `apertus-1p5-eval`
- `third_party/VLMEvalKit`: [github.com/swiss-ai/VLMEvalKit](https://github.com/swiss-ai/VLMEvalKit), branch `apertus-1p5-eval`
- `third_party/Evaluator`: NeMo Evaluator checkout used for text benchmark orchestration.
- `third_party/lm-evaluation-harness`: EleutherAI lm-evaluation-harness checkout used for direct text/loglikelihood benchmarks.

## Example Usage

The main entrypoint is `launchers/eval.sh`. It sets `ORCH_REPO_ROOT`, resolves common options, and dispatches to a framework-specific launcher.

Run lmms-eval production jobs:

```bash
bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --suite smoke
bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --tasks task_suites/lmms-eval/visual_full.txt
```

Run lmms-eval audio benchmarks through the same launcher by selecting an audio suite or a specific audio task:

```bash
bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --suite audio-smoke
bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --suite audio-full
bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --tasks google_fleurs
```

For local audio task changes in `third_party/lmms-eval`, run interactively with `LMMS_EVAL_DEV_PATH` and pass lmms-eval launcher args directly:

```bash
export MODEL="/capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus-1p5-8B-sft-capfilter-lr6e-5-constant-innovator-fix-it23409"
export TOK="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
export VLLM_APERTUS_AUDIO_TOKENIZER_CODEBASE="/workspace/benchmark-audio-tokenizer"

LMMS_EVAL_DEV_PATH="$PWD/third_party/lmms-eval" \
ENABLE_WANDB=false \
bash launchers/eval.sh \
  --eval-framework lmms-eval \
  --model "$MODEL" \
  --tasks fleurs_en_us \
  --submit-mode interactive \
  --tokenizer-path "$TOK" \
  --gpu-memory-utilization 0.75 \
  --trust-remote-code True \
  --extra-model-args 'allowed_local_media_path=/,limit_mm_per_prompt={"audio":1,"image":1},mm_processor_kwargs={"apertus_audio_tokenizer_path":"/capstor/store/cscs/swissai/infra01/MLLM/wavtokenizer"}'
```

The lmms-eval launcher automatically uses `$TOK/chat_template.jinja` when `TOK` is the default Apertus tokenizer path. For any other tokenizer, pass the template explicitly:

```bash
bash launchers/eval.sh --eval-framework lmms-eval --model "$MODEL" --tasks google_fleurs \
  --tokenizer-path "$TOK" \
  --chat-template "$TOK/chat_template.jinja"
```

Submit VLMEvalKit production jobs through the combined production launcher:

```bash
bash launchers/eval.sh --eval-framework VLMEvalKit --suite smoke --model Apertus-1p5-8B
bash launchers/eval.sh --eval-framework VLMEvalKit --tasks task_suites/VLMEvalKit/full.txt --model Apertus-1p5-8B
```

Run NeMo Evaluator text benchmarks through the combined launcher. The Evaluator launcher resolves the suite and launches one task per job. Each task job starts vLLM locally in the job runtime and runs the benchmark against the local OpenAI-compatible endpoint:

```bash
bash launchers/eval.sh --eval-framework Evaluator --suite smoke
bash launchers/eval.sh --eval-framework Evaluator --suite text-builtin --model /path/to/apertus
```

Evaluator uses `configs/Evaluator/apertus_text_vllm_template.yaml` as a template. The launcher copies that template into each task output directory as `config.template.yaml`, writes the resolved `NEL_*` values into `config.env`, and the Slurm entrypoint sources that env file before calling `nel eval run`. This means the run directory records the exact model, tokenizer, solver, generation, vLLM, cache, and output settings used for that task.

Most template values can be changed directly from the top-level entrypoint:

```bash
bash launchers/eval.sh --eval-framework Evaluator --suite smoke \
  --solver-type simple \
  --max-problems 100 \
  --max-tokens 1024 \
  --temperature 0 \
  --gpu-memory-utilization 0.75
```

For native Evaluator config fields that are not first-class launcher flags, pass normal `nel eval run` overrides with repeatable `--extra-framework-config`:

```bash
bash launchers/eval.sh --eval-framework Evaluator --suite smoke \
  --extra-framework-config -O \
  --extra-framework-config benchmarks.0.timeout=3600 \
  --extra-framework-config -O \
  --extra-framework-config output.progress_interval=30
```

Use `--config-template /path/to/template.yaml` when you want to keep the same launcher flow but replace the entire NeMo Evaluator template.

Run direct lm-evaluation-harness text benchmarks through the combined launcher. This is the recommended path for classic loglikelihood and multiple-choice tasks such as MMLU, ARC, HellaSwag, Winogrande, PIQA, and TruthfulQA MC:

```bash
bash launchers/eval.sh \
  --eval-framework lm-evaluation-harness \
  --model /capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-70b-sft-262k-3000 \
  --suite smoke \
  --submit-mode batch

bash launchers/eval.sh \
  --eval-framework lm-evaluation-harness \
  --model /capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-70b-sft-262k-3000 \
  --suite text \
  --submit-mode batch
```

The lm-evaluation-harness launcher uses the HuggingFace backend. For large checkpoints that do not fit on a single GPU, the Slurm wrapper now enables `parallelize=True`, which shards the model itself across all visible GPUs in one eval process. In that mode, lower `--num-processes` to `1` unless you explicitly want additional eval replicas. You can also pass generation controls with `--gen-kwargs` and extra framework-native argv tokens with repeatable `--extra-framework-config`, for example:

```bash
bash launchers/eval.sh \
  --eval-framework lm-evaluation-harness \
  --model /capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-70b-sft-262k-3000 \
  --tasks hellaswag \
  --submit-mode batch \
  --num-processes 1

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

Only the HuggingFace backend is wired for direct lm-evaluation-harness runs right now. Add more backends explicitly when needed.

Each production launcher call creates one shared run directory under both the framework results and logs folders. Per-task jobs submitted by that call write into that same result/log run directory. Override `RUN_ID` to choose the directory name explicitly.

Image-token cache defaults are framework-specific and persistent under `cache/lmms-eval/` or `cache/VLMEvalKit/`. Jobs use the shared cache directly with local copy disabled, preload enabled, read access enabled, and write-misses enabled. lmms-eval defaults to `--mode fill`.

Batch-size defaults are framework-specific. lmms-eval defaults to `512`; lm-evaluation-harness defaults to `auto` for HuggingFace; other launchers expose their own concurrency and batch flags. Override first-class launcher values directly, or use repeatable `--extra-framework-config` for native framework argv tokens.

The combined launcher prefetches `BAAI/Emu3.5-VisionTokenizer` into `cache/models/BAAI/Emu3.5-VisionTokenizer` before it submits multimodal jobs, so the tokenizer files are present before evaluation starts.

For `--eval-framework Evaluator` and `--eval-framework lm-evaluation-harness`, the combined launcher skips the vision-tokenizer prefetch because these integrations target text benchmarks.

## Suites

Suite files live under `task_suites/` and can be passed directly to the launchers with `--tasks`.

- `task_suites/lmms-eval/visual_smoke.txt`: `gqa,mmstar,pope`
- `task_suites/lmms-eval/visual_full.txt`: full lmms-eval visual evaluation suite.
- `task_suites/lmms-eval/audio_smoke.txt`: `fleurs`
- `task_suites/lmms-eval/audio_full.txt`: full lmms-eval audio evaluation suite.
- `task_suites/lmms-eval/audio_llm_eval.txt`: audio tasks intended for LLM-eval style runs.
- `task_suites/VLMEvalKit/`: suite files copied from the VLMEvalKit Apertus vLLM scripts.
- `task_suites/Evaluator/smoke.txt`: native Evaluator text smoke suite (`gsm8k`).
- `task_suites/Evaluator/text_builtin.txt`: native text-oriented Evaluator benchmarks: `gsm8k`, `math500`, `mgsm`, `drop`, `triviaqa`, `mmlu`, `mmlu_pro`, `gpqa`, `simpleqa`, `healthbench`, `xstest`.
- `task_suites/lm-evaluation-harness/smoke.txt`: direct lm-evaluation-harness smoke suite (`hellaswag`).
- `task_suites/lm-evaluation-harness/text.txt`: common direct lm-evaluation-harness text/loglikelihood suite.

## Recommended Run Metadata

Every run should create a `run_meta.json` in its output directory with:

- `tool`
- `tool_commit`
- `launcher_commit`
- `model`
- `backend`
- `config`
- `tasks`
- `container_image`
- `slurm_job_id`
- `date`
- `output_dir`

## Development Notes

Changes to evaluation framework code should happen inside the corresponding submodule branch. This repository should own configs, launchers, containers, logs, results, Slurm templates, and utility scripts.

Framework-specific production launchers under `launchers/lmms-eval/`, `launchers/VLMEvalKit/`, `launchers/Evaluator/`, and `launchers/lm-evaluation-harness/` expect `ORCH_REPO_ROOT` to be set by `launchers/eval.sh`.

The production runtime expects `lmms-eval` and `VLMEvalKit` to be available under `/workspace` inside the job container. Evaluator is run directly from `third_party/Evaluator` when `nel` is not installed, via `PYTHONPATH=third_party/Evaluator/src`. The direct lm-evaluation-harness wrapper always prepends `third_party/lm-evaluation-harness` to `PYTHONPATH` and invokes `lm_eval` directly, using `accelerate launch` when `--num-processes` is greater than 1, so local changes in that checkout are used by the job. If you need a custom implementation, make the changes inside the matching `third_party/` checkout, install or update that version from `third_party/`, and use `--submit-mode interactive` so the job runs with the current shell and node allocation.

Evaluator's launcher uses `slurm/Evaluator/eval_job.slurm` for one task at a time. The launcher writes each task's Evaluator template/env pair at `results/Evaluator/<RUN_ID>/<task>/config.template.yaml` and `results/Evaluator/<RUN_ID>/<task>/config.env`, then passes both paths to the Slurm script. The Slurm script is the final entrypoint and runs `nel eval run` directly inside the container. The script intentionally reuses the shared Apertus runtime TOML at `toml/shared/apertus-vllm-vision-eval-prod.toml`; there is no Evaluator-specific TOML or sqsh. In `--submit-mode interactive`, the launcher runs that same script with `bash` on the current allocation.

lm-evaluation-harness uses `slurm/lm-evaluation-harness/eval_job.slurm` for one task at a time. The launcher writes outputs to `results/lm-evaluation-harness/<RUN_ID>/<model>/<task>/` and logs to `logs/lm-evaluation-harness/<RUN_ID>/`. The Slurm script reuses `toml/shared/apertus-vllm-vision-eval-prod.toml` for the runtime image and runs the HuggingFace lm-eval backend from the local third-party checkout.

For local `third_party/lmms-eval` changes to be used by the lmms-eval Slurm wrapper, set `LMMS_EVAL_DEV_PATH` to this checkout. Otherwise the job prepends `/workspace/lmms-eval` to `PYTHONPATH` and your local task/model changes may not be visible:

```bash
LMMS_EVAL_DEV_PATH="$PWD/third_party/lmms-eval" \
bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --tasks google_fleurs --submit-mode interactive
```

You can sanity-check task registration before launching a full run, but this requires `python` to be available in the active environment:

```bash
PYTHONPATH="$PWD/third_party/lmms-eval:/workspace/lmms-eval:${PYTHONPATH:-}" \
python -m lmms_eval --tasks list | grep google_fleurs
```

When you want to install the custom checkouts from this repository directly, use:

```bash
cd third_party/lmms-eval
uv pip install --python /opt/venv/bin/python --no-build-isolation --editable . ".[all]"
```

```bash
cd third_party/VLMEvalKit
uv pip install --python /opt/venv/bin/python --no-deps --editable .
```

```bash
cd third_party/Evaluator
uv pip install --python /opt/venv/bin/python --editable ".[lm-eval]"
```

```bash
cd third_party/lm-evaluation-harness
uv pip install --python /opt/venv/bin/python --editable ".[hf]"
```

The combined launcher accepts common top-level arguments such as `--model`, `--tasks`, `--suite`, `--mode`, `--submit-mode`, and `--run-id`. Use repeatable `--extra-framework-config <arg>` for framework-native argv tokens that are not first-class launcher flags.

Use `--submit-mode interactive` when you want a framework launcher to run its job script directly with `bash` on the current node allocation instead of submitting a new Slurm job.
