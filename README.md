# MLLM Evaluation Suite

Orchestration layer for evaluating the Apertus multimodal models on a Slurm cluster, using two
established harnesses — [lmms-eval](https://github.com/swiss-ai/lmms-eval) and
[VLMEvalKit](https://github.com/swiss-ai/VLMEvalKit) — kept as pinned submodule forks under
`third_party/`. This repository does not reimplement either framework; it owns the launcher
interface, Slurm templates, container configs, task suites, and the result pipeline that produces
the published dashboard: **[swiss-ai.github.io/MLLM-eval-suite](https://swiss-ai.github.io/MLLM-eval-suite/)**.

## Why two harnesses

Each benchmark is *owned* by exactly one harness, chosen for comparability with how the rest of
the field reports that benchmark — not for convenience:

| harness | owns | why |
|---|---|---|
| **VLMEvalKit** | Spatial-intelligence suite (CV-Bench, MMSI-Bench, OmniSpatial, SITE, ViewSpatial, EmbSpatial, MindCube, SparBench, ERQA, BLINK, …) | These follow the [EASI leaderboard](https://easi.lmms-lab.com/leaderboard/) protocol ([*Holistic Evaluation of Multimodal LLMs on Spatial Intelligence*](https://arxiv.org/abs/2508.13142)), which runs on VLMEvalKit. Every other model on that leaderboard was evaluated through this exact pipeline — using it makes our numbers directly comparable, down to prompt variants (e.g. OmniSpatial is reported with EASI's *manual-CoT* prompt). |
| **VLMEvalKit** | Most judge-scored benchmarks (MMVet, MIA-Bench, MMSafetyBench, CharXiv, MathVista extraction, …) | VLMEvalKit ships the mature judge pipeline the community reports with: per-benchmark judge pins (e.g. `gpt-4-turbo-2024-04-09` for MMVet, `gpt-4o-mini` for CharXiv as in the CharXiv paper), retry/caching, and judge-failure accounting. Published numbers for other models used these same judge configs. |
| **lmms-eval** | Everything else: the broad visual suite (VQA, OCR/doc/chart, math, hallucination, grounding), text-medical, **HealthBench**, and our custom remote-sensing tasks (VRSBench, GEOBench-VLM, BigEarthNet.txt, RSRCC, FRIEDA) | Largest task library, straightforward custom-task authoring, and vLLM data-parallel throughput for bulk evaluation. |

The dashboard badges each row with its owning harness; a benchmark is never double-reported.

## Evaluation protocol

**Generation** follows the [Artificial Analysis intelligence-benchmarking
methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking) so our operating
point matches the most widely cited third-party evaluations:

- **Non-thinking runs:** `max_new_tokens = 16384`, `temperature = 0`. (AA caps non-reasoning
  models at 16,384 output tokens, greedy.)
- **Thinking runs:** `max_new_tokens = 32768`, `temperature = 0.6`, `top_p = 0.95` — AA lets
  reasoning models use the creator-disclosed budget and sampling recipe. Thinking rows are marked
  on the dashboard, including per-task truncation rates where the 32k budget was hit.

These defaults live in one place per layer (`launchers/*/eval.sh`, the slurm templates, and the
Apertus wrapper in the VLMEvalKit fork) and agree by construction.

**Judged benchmarks.** Two distinct uses of an LLM judge, which we keep apart deliberately:

- *Answer extraction* (MathVista, MathVerse, LogicVista, DynaMath-style): the judge only pulls the
  final answer out of free-form text; correctness is exact-match against gold. Judge choice barely
  moves scores.
- *Actual judging* (MMVet, MIA-Bench, MMSafetyBench, CharXiv, HealthBench): the judge scores
  quality against rubrics or references, so the grader is part of the benchmark definition and we
  pin whatever the benchmark's authors validated.

**HealthBench** ([Arora et al., OpenAI](https://arxiv.org/abs/2505.08775)) is the heaviest judged
benchmark: 5,000 physician-authored conversations scored criterion-by-criterion against 48.5k
rubric items. We report it twice, on purpose:

- `healthbench` — graded by **Qwen3-235B-A22B-Instruct** served on-cluster, the fully-open
  protocol of [FullyOpenMeditron](https://arxiv.org/abs/2605.16215); directly comparable to their
  published table and free to run per checkpoint (`slurm/shared/healthbench_grader_server.slurm`).
- `healthbench_gpt41` — re-graded with **GPT-4.1**, the official grader OpenAI validated against
  physicians; this is the row to quote against public leaderboards. Our calibration shows the open
  judge is uniformly ~14 points more lenient with ranking preserved (r = 0.82), so both rows tell
  one consistent story. (`scripts/regrade_healthbench.py` re-grades cached completions without
  re-running inference.)

## Quickstart (clone and run)

Evaluate your checkpoint with one command — no file edits required:

```bash
git clone --recurse-submodules https://github.com/swiss-ai/MLLM-eval-suite
cd MLLM-eval-suite
bash launchers/eval.sh --model /path/to/your/checkpoint --suite smoke   # both harnesses
```

Results land under `results/<framework>/<run-id>/`, logs under `logs/<framework>/<run-id>/`.
Add `--thinking` for the reasoning recipe described above.

Cluster-account knobs (defaults target the current Apertus reservation; single source of truth in
`slurm/shared/sbatch_overrides.sh`):

```bash
EVAL_ACCOUNT=<account>        # slurm account            (default: infra01)
EVAL_RESERVATION=<name>       # set EVAL_RESERVATION= (empty) to submit without a reservation
EVAL_ENVIRONMENT=<edf.toml>   # pyxis container config   (default: this repo's toml/shared/)
```

Judge-scored benchmarks (`task_suites/VLMEvalKit/llm_judge.txt`) need an OpenAI key: export
`OPENAI_API_KEY` or put it in `third_party/VLMEvalKit/.env`. The launcher refuses to submit judge
tasks without one rather than silently falling back to regex scoring (`ALLOW_NO_JUDGE=1` overrides).

`--mode` is framework-specific (lmms-eval: `fill|readonly`; VLMEvalKit: `all|infer|eval`) and is
rejected with `--eval-framework all` — the defaults are correct for production runs.

## Text evaluation integrations

Run NeMo Evaluator text benchmarks through the combined launcher. The Evaluator launcher resolves the suite and launches one task per job. Each task job starts vLLM locally in the job runtime and runs the benchmark against the local OpenAI-compatible endpoint:

```bash
bash launchers/eval.sh --eval-framework Evaluator --suite smoke
bash launchers/eval.sh --eval-framework Evaluator --suite text-builtin --model /path/to/apertus
```

Evaluator uses per-benchmark templates from `configs/Evaluator/templates/` when available, falling back to `configs/Evaluator/apertus_text_vllm_template.yaml`. The launcher copies the selected template into each task output directory as `config.template.yaml`, writes the resolved `NEL_*` values into `config.env`, and the Slurm entrypoint sources that env file before calling `nel eval run`. This means the run directory records the exact model, tokenizer, solver, generation, vLLM, cache, benchmark, scoring, sandbox, and output settings used for that task.

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

Use `--template-dir /path/to/templates` when you want benchmark-specific templates but do not want to modify the repository defaults. Template file names should match the resolved task slug, for example `gsm8k.yaml`, `mmlu_pro.yaml`, or `terminal-bench-v1.yaml`.

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

See `configs/Evaluator/README.md`, `launchers/lm-evaluation-harness/README.md`, and the corresponding `slurm/` READMEs for runtime configuration. Initialize all four pinned dependencies with `git submodule update --init --recursive`.

## Example Usage

Submit lmms-eval production jobs through the combined production launcher:

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

For local audio task changes in `third_party/lmms-eval`, run interactively with `LMMS_EVAL_DEV_PATH` and pass vLLM/audio-tokenizer args after `--`:

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
  -- \
  --tokenizer-path "$TOK" \
  --gpu-memory-utilization 0.75 \
  --trust-remote-code True \
  --extra-model-args 'allowed_local_media_path=/,limit_mm_per_prompt={"audio":1,"image":1},mm_processor_kwargs={"apertus_audio_tokenizer_path":"/capstor/store/cscs/swissai/infra01/MLLM/wavtokenizer"}'
```

The lmms-eval launcher automatically uses `$TOK/chat_template.jinja` when `TOK` is the default Apertus tokenizer path. For any other tokenizer, pass the template explicitly after `--`:

```bash
bash launchers/eval.sh --eval-framework lmms-eval --model "$MODEL" --tasks google_fleurs -- \
  --tokenizer-path "$TOK" \
  --chat-template "$TOK/chat_template.jinja"
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

Adding new benchmarks, staging benchmark data, and dashboard regeneration go through the repo
admin; as a user you only need the commands above.

## Suites

Suite files under `task_suites/` are the source of truth for what each named `--suite` runs
(one file per suite; edit the file, no script changes needed).

- `task_suites/lmms-eval/`: `visual_smoke`, `visual_full`, `geospatial_smoke`, `geospatial_full`
  (remote-sensing tasks; need staged imagery, see `RS_DATASETS_ROOT`), `audio_smoke`, `audio_full`,
  `audio_llm_eval`.
- `task_suites/VLMEvalKit/`: `smoke`, `full`, `spatial` (the EASI set), `llm_judge`
  (judge-scored benchmarks — key required, see Quickstart).

## Repository structure

- `shared/` — engine-independent runtime modules importable by both harnesses (Apertus image tokenization; on every job's PYTHONPATH).
- `third_party/` — the two harness forks (lmms-eval on `apertus-1p5-eval-v2`, VLMEvalKit on `apertus-1p5-eval`), pinned by commit.
- `launchers/` — production entrypoints: `eval.sh` dispatcher plus per-framework launchers.
- `slurm/` — job templates and shared snippets (`sbatch_overrides.sh`, image-token cache env,
  the HealthBench grader server).
- `toml/` — pyxis EDF container configs per framework plus the shared production image.
- `task_suites/` — the suite files above.
- `scripts/` — admin surface: the results→dashboard pipeline, verification, re-grading, staging.
  See `scripts/README.md` for the operations guide.
- `docs/` — the published dashboard (`index.html`) and audit/improvement notes.
- `cache/`, `results/`, `logs/` — generated, git-ignored.

## Submodules

```bash
git submodule update --init --recursive          # after cloning
git submodule update --remote --merge            # move to branch tips (admin)
```

- `third_party/lmms-eval`: [swiss-ai/lmms-eval](https://github.com/swiss-ai/lmms-eval), branch `apertus-1p5-eval-v2`
- `third_party/VLMEvalKit`: [swiss-ai/VLMEvalKit](https://github.com/swiss-ai/VLMEvalKit), branch `apertus-1p5-eval`

## Development notes (admin)

Framework code changes happen inside the corresponding submodule branch; this repository owns
configs, launchers, containers, Slurm templates, and utility scripts. The per-framework launchers
expect `ORCH_REPO_ROOT` from `launchers/eval.sh`. The production containers bake both harnesses
under `/workspace`; a checkout under `third_party/` can be layered over the baked copy via
`LMMS_EVAL_DEV_PATH` (lmms-eval) or editable installs — see the launcher READMEs. Use
`--submit-mode interactive` to run a job script directly on an existing allocation.

The results pipeline, dashboard regeneration, HealthBench grading flows, and data staging are
documented in `scripts/README.md`. Deeper background: `docs/eval_suite_audit.md` (state audit) and
`docs/repo_improvement_note.md` (prioritized improvement roadmap).
