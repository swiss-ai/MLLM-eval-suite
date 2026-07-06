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
- Keep logs and results in structured, predictable locations.
- Normalize and compare outputs across evaluation tools where possible.

## Repository Structure

- `third_party/`: Git submodules for upstream evaluation frameworks.
- `dockerfiles/`: Dockerfile definitions and build documentation.
- `toml/`: Centralized TOML configuration files, split by framework plus shared settings.
- `launchers/`: Production launch entrypoints (`eval.sh` plus per-framework launchers).
- `slurm/`: Slurm templates and shared Slurm environment snippets.
- `task_suites/`: Suite files for lmms-eval and VLMEvalKit. Pass these paths directly to `--tasks`.
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

- `third_party/lmms-eval`: [github.com/swiss-ai/lmms-eval](https://github.com/swiss-ai/lmms-eval), branch `apertus-1p5-eval`
- `third_party/VLMEvalKit`: [github.com/swiss-ai/VLMEvalKit](https://github.com/swiss-ai/VLMEvalKit), branch `apertus-1p5-eval`

## Quickstart (clone and run)

Evaluate your checkpoint with one command — no file edits required:

```bash
git clone --recurse-submodules https://github.com/swiss-ai/MLLM-eval-suite
cd MLLM-eval-suite
bash launchers/eval.sh --model /path/to/your/checkpoint --suite smoke   # both harnesses
```

Results land under `results/<framework>/<run-id>/`, logs under `logs/<framework>/<run-id>/`.

Cluster-account knobs (defaults target the current Apertus reservation; override per user/site,
single source of truth in `slurm/shared/sbatch_overrides.sh`):

```bash
EVAL_ACCOUNT=<account>            # slurm account            (default: infra01)
EVAL_RESERVATION=<name>           # reservation; set EVAL_RESERVATION= (empty) to submit without one
EVAL_ENVIRONMENT=<edf.toml>       # pyxis container config   (default: this repo's toml/shared/)
```

Judge-scored benchmarks (`task_suites/VLMEvalKit/llm_judge.txt`) need an OpenAI key: export
`OPENAI_API_KEY` or put it in `third_party/VLMEvalKit/.env`. The launcher refuses to submit judge
tasks without one (ALLOW_NO_JUDGE=1 overrides, scoring falls back to regex parsing).

Note: `--mode` is framework-specific (lmms-eval: `fill|readonly`; VLMEvalKit: `all|infer|eval`)
and is rejected with `--eval-framework all` — the defaults are correct for production runs.

Adding new benchmarks, staging benchmark data, and dashboard regeneration go through the repo
admin; as a user you only need the commands above.

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

Suite files under `task_suites/` are the source of truth for what each named `--suite` runs
(one comma-separated task list per file; edit the file, no script changes needed).

- `task_suites/lmms-eval/`: `visual_smoke`, `visual_full`, `geospatial_smoke`, `geospatial_full`
  (remote-sensing tasks; need staged imagery, see `RS_DATASETS_ROOT`), `audio_smoke`, `audio_full`,
  `audio_llm_eval`.
- `task_suites/VLMEvalKit/`: `smoke`, `full`, `spatial` (EASI spatial-intelligence set),
  `llm_judge` (benchmarks scored by an OpenAI judge — key required, see Quickstart).

## Dashboard

Aggregated results are published as a self-contained page at
[swiss-ai.github.io/MLLM-eval-suite](https://swiss-ai.github.io/MLLM-eval-suite/) (source:
`docs/index.html`). Regeneration is an admin flow — see `scripts/README.md`.

## Development Notes

Changes to evaluation framework code should happen inside the corresponding submodule branch. This repository should own configs, launchers, containers, logs, results, Slurm templates, and utility scripts.

Framework-specific production launchers under `launchers/lmms-eval/` and `launchers/VLMEvalKit/` expect `ORCH_REPO_ROOT` to be set by `launchers/eval.sh`.

The production runtime expects `lmms-eval` and `VLMEvalKit` to be available under `/workspace` inside the job container. If you need a custom implementation, make the changes inside the matching `third_party/` checkout, install or update that version from `third_party/`, and use `--submit-mode interactive` so the job runs with the current shell and node allocation.

When you want to install the custom checkouts from this repository directly, use:

```bash
cd third_party/lmms-eval
uv pip install --python /opt/venv/bin/python --no-build-isolation --editable . ".[all]"
```

```bash
cd third_party/VLMEvalKit
uv pip install --python /opt/venv/bin/python --no-deps --editable .
```

The combined launcher accepts common top-level arguments such as `--model`, `--tasks`, `--suite`, `--mode`, `--submit-mode`, and `--run-id`. Framework-specific options can be passed after `--`.

Use `--submit-mode interactive` with either framework when you want the launcher to run the job script directly with `bash` on the current node allocation instead of submitting a new Slurm job.
