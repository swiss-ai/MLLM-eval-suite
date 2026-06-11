# MLLM Evaluation Suite

## Overview

This repository is a unified orchestration layer for evaluating multimodal models with both VLMEvalKit and lmms-eval.

It does not merge, fork, or reimplement either evaluation framework. Instead, both frameworks are kept as pinned Git submodules under `third_party/`, while this repository owns the shared launcher interface, configuration layout, container metadata, Slurm templates, logs, results, and post-processing utilities.

## Goals

- Provide a common launcher interface for multiple evaluation frameworks.
- Make evaluation runs reproducible through explicit configs, task lists, and metadata.
- Centralize TOML configs for framework-specific and shared settings.
- Manage Dockerfiles used to build or document evaluation environments.
- Provide Slurm job templates for batch execution.
- Abstract model serving backends for vLLM, SGLang, and Hugging Face.
- Keep logs and results in structured, predictable locations.
- Normalize and compare outputs across evaluation tools where possible.

## Repository Structure

- `third_party/`: Git submodules for upstream evaluation frameworks.
- `dockerfiles/`: Dockerfile definitions and build documentation.
- `toml/`: Centralized TOML configuration files, split by framework plus shared settings.
- `launchers/`: Shell entrypoints for local or scripted evaluation runs.
- `launchers/backends/`: Backend adapters for vLLM, SGLang, and Hugging Face.
- `slurm/`: Slurm templates and shared Slurm environment snippets.
- `task_suites/`: Suite files for lmms-eval and VLMEvalKit. Pass these paths directly to `--tasks`.
- `task_lists/`: Legacy plain-text task lists for ad hoc or common subsets.
- `cache/`: Local cache root. Image-token and framework data caches are split under `cache/lmms-eval/` and `cache/VLMEvalKit/`; shared runtime caches use common folders such as `cache/hf`, `cache/nltk_data`, `cache/xdg`, `cache/vllm`, and `cache/models`. Generated contents are ignored.
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

- `third_party/lmms-eval`: `https://github.com/swiss-ai/lmms-eval.git`, branch `apertus-1p5-eval`
- `third_party/VLMEvalKit`: `https://github.com/swiss-ai/VLMEvalKit.git`, branch `apertus-1p5-eval`

## Example Usage

Run lmms-eval through the unified launcher:

```bash
bash launchers/run_eval.sh --tool lmms-eval --backend vllm --config toml/lmms-eval/apertus-vllm-lmms-eval-prod.toml --tasks task_suites/lmms-eval/visual_smoke.txt --output results/lmms-eval/example_run
```

Run VLMEvalKit through the unified launcher:

```bash
bash launchers/run_eval.sh --tool VLMEvalKit --backend vllm --config toml/VLMEvalKit/example.toml --tasks task_suites/VLMEvalKit/smoke.txt --output results/VLMEvalKit/example_run
```

Submit lmms-eval production jobs through the combined production launcher:

```bash
bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --suite smoke
bash launchers/eval.sh --eval-framework lmms-eval --model /path/to/model --tasks task_suites/lmms-eval/visual_full.txt
```

Submit VLMEvalKit production jobs through the combined production launcher:

```bash
bash launchers/eval.sh --eval-framework VLMEvalKit --suite smoke --model Apertus-1p5-8B
bash launchers/eval.sh --eval-framework VLMEvalKit --tasks task_suites/VLMEvalKit/full.txt --model Apertus-1p5-8B
```

Each production launcher call creates one shared run directory under both the framework results and logs folders. All per-task Slurm jobs submitted by that call write into that same result/log run directory. Override `RUN_ID` to choose the directory name explicitly.

Image-token cache defaults are framework-specific and persistent under `cache/lmms-eval/` or `cache/VLMEvalKit/`. Jobs use the shared cache directly with local copy disabled, preload enabled, read access enabled, and write-misses enabled. lmms-eval defaults to `--mode fill`.

The default batch size is `512` for both production launchers unless overridden with `--batch-size` after `--` or via framework-specific environment variables.

The combined launcher prefetches `BAAI/Emu3.5-VisionTokenizer` into `cache/models/BAAI/Emu3.5-VisionTokenizer` before it submits jobs, so the tokenizer files are present before evaluation starts.

## Suites

Suite files live under `task_suites/` and can be passed directly to the launchers with `--tasks`.

- `task_suites/lmms-eval/visual_smoke.txt`: `gqa,mmstar,pope`
- `task_suites/lmms-eval/visual_full.txt`: full lmms-eval visual evaluation suite.
- `task_suites/lmms-eval/audio_smoke.txt`: `fleurs`
- `task_suites/lmms-eval/audio_full.txt`: full lmms-eval audio evaluation suite.
- `task_suites/lmms-eval/audio_llm_eval.txt`: audio tasks intended for LLM-eval style runs.
- `task_suites/VLMEvalKit/`: suite files copied from the VLMEvalKit Apertus vLLM scripts.

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

Framework-specific production launchers under `launchers/lmms-eval/` and `launchers/VLMEvalKit/` expect `ORCH_REPO_ROOT` to be set by `launchers/eval.sh`.

The combined launcher accepts common top-level arguments such as `--model`, `--tasks`, `--suite`, `--mode`, and `--run-id`. Framework-specific options can be passed after `--`.
