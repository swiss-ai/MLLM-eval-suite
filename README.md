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

## Suite contracts

The `suite/` package makes the launch and reporting path checkable. All commands run from the repository root with the host Python (3.12, standard library only); the job scripts call the same modules with the container Python.

- **Task registry** `suite/tasks.toml` declares every launchable task: owning harness, harness task id, judge requirement and key variable, local dataset assets with their sources, required context length, and membership in the HF model-card set (`card`) and the report table (`report`). Its `[dashboard]` section is the dashboard's benchmark table verbatim. `python3 -m suite.tasks --check` verifies the generated judge lists (`task_suites/*/llm_judge*.txt`) against it; `--write-suite-lists` regenerates them; `--list card|report|all` prints task names; `--max-model-len TASK --framework FW` prints the context a task needs. Each `[dashboard.<task>]` entry also declares the benchmark's headline metric (`headline`, an ordered list; `lmms_headline` for a harness that only aliases the task; `headline_strict` to forbid fallbacks), so what a number means lives with the benchmark rather than in script code.
- **Preflight** `python3 -m suite.preflight --framework FW --model PATH --tasks a,b [--thinking] --tokenizer DIR --vision-tokenizer DIR --container-image IMG --max-model-len N` checks that weights resolve and are readable, the tokenizer and chat template exist, the VQ tokenizer weights exist, every declared asset directory holds at least its minimum file count, a judge key is present when a task needs one, the image exists, and the context length suffices. Exit 2 lists every failing check. Launchers run it once per model before `sbatch` and refuse to submit on failure; jobs run it again before loading the model. `SKIP_PREFLIGHT=1` bypasses both. With `--harness-root DIR` (the job scripts pass the lmms-eval checkout) every launched lmms-eval task must be declared exactly once under `lmms_eval/tasks`; a name declared in two files fails the launch instead of resolving by directory order.
- **Run manifest** every job writes `run_meta.json` at its task output root before inference (checkpoint path, real path, config hash, shard sizes, tokenizer and template hashes, requested thinking flag, generation settings, harness and suite commits, container image, Slurm job) and finalizes it after the harness with `status` `ok`, `failed`, or `invalid`, an error excerpt, the results file, and output-token statistics. Job exit codes follow the status: 3 for `failed`, 4 for `invalid`. A thinking run whose mean output tokens fall below 8, or whose canary failed, is `invalid`. Runs without a manifest show as legacy on the dashboard. `manifest start` records the harness checkout the job imports (`--harness-dir`, a worktree or the pinned submodule) and the generation settings (`--tp --dp --batch-size --gpu-memory-utilization --max-model-len --limit`).
- **Thinking canary** the Apertus wrappers in both harness forks spend one text-only request before the first scored generation when thinking is requested and raise unless a deliberation block appears. `APERTUS_SKIP_THINKING_CANARY=1` disables it.
- **Datasets** `python3 -m suite.stage_datasets --task frieda` restores a task's declared assets from their sources into `cache/rs_datasets`.
- **Tokenize-only pass** `bash launchers/eval.sh --eval-framework lmms-eval --model PATH --tasks a,b --mode tokenize` fills the image-token cache with the VQ encoder alone, one shard per GPU, without loading the language model. The `--size 70b` profile then reads the cache strictly, so a large model never encodes images beside its tensor-parallel worker; pass `--allow-encode` to permit encoding anyway. The 70b profile runs at `gpu_memory_utilization=0.75`.
- **Status** `python3 -m suite.status [--run-id X] [--slurm]` lists every manifest with its state; `--slurm` marks running manifests whose job has ended as aborted.
- **Dashboard** `scripts/refresh_dashboard.sh` checks the registry, builds the page from results and manifests, imports cells from earlier builds under `docs/legacy/` for slots no current run covers (marked legacy), and writes `docs/dashboard.json` and `docs/coverage.json` next to the page. The build prints the coverage over card and report tasks and every registry column without runs. A run whose manifest records a sample limit is never a dashboard number, so gate checks and smoke tests cannot displace a full run; keep such runs under `cache/validation/results/<harness>` (`OUTPUT_PATH`, `WORK_BASE`, `OUTPUT_BASE` for the three launchers) rather than `results/`.
- **Tests** `python3 -m pytest tests -q` covers the registry, manifests, preflight, dataset restoration, coverage, and status without a GPU or network.
- **Harness sync gate** `scripts/sync_audit.sh cache/worktrees/<harness>-sync [merge-commit] docs/sync/<date>-<harness>.accept` lists every fork-changed file whose merged content equals upstream (dropped) or lacks fork-added lines (partial); the gate fails unless each finding is in the accept list with the reason it was superseded or restored elsewhere. Submodule pointers move only after this passes and the merged harness reproduces the pinned harness on the deterministic A/B runs recorded in the design spec.
