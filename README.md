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
