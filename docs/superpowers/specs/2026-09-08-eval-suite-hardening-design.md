# Eval Suite Hardening Design

Date: 2026-09-08. Status: approved in direction by the suite owner ("fire"); written for review.

## 1. Why

On 2026-09-08 a gap-fill sweep for the Apertus 1.5 release exposed a set of defects that share one shape: the suite produced plausible output while a critical condition was silently false.

- Every lmms-eval image run with `--thinking` since the July harness bump generated 2 to 4 output tokens. The wrapper stored `enable_thinking` before calling the base constructor, and the new upstream base reset the attribute to `None`. Scores looked normal.
- Judge-scored tasks fall back to an all-zero dummy judge when the OpenAI key is absent. The guard that should block them reads a suite file that had been purged, so it silently passed.
- Jobs whose evaluation crashed reported `COMPLETED` to Slurm because lmms-eval catches the exception and exits zero. Failures were visible only by reading logs.
- The 70B TP4 profile OOMed on GPU 0 because the Emu3.5 VQ encoder runs in the main process on the same GPU as the TP0 worker, one image at a time, with no bound on the caching allocator. A first mitigation, `expandable_segments`, crashed every TP worker during FlashInfer all-reduce setup and had to be reverted.
- A 70B job loaded weights for ten minutes before discovering its dataset images were gone. Another exceeded the fixed 131k context on a multi-image benchmark.
- The scratch purge removed raw results, logs, caches, 462 harness files, task suite lists, the judge key file, and loose git objects. The Aug 21 dashboard HTML was the only record of the corrected 8B sweep. One VLMEvalKit commit lost its tree object.
- The dashboard build dropped 927 cells without failing, and its registry had drifted from the built page.

Each of these is a missing contract, not an isolated bug. This design names the contracts and where they live.

## 2. Goals and non-goals

Goals:
1. A number on the dashboard can be traced to the exact checkpoint, tokenizer, template, flags, harness commit, and container that produced it, without re-deriving anything.
2. A run whose critical configuration is not in effect fails, loudly, before it produces numbers.
3. A submission that cannot succeed is refused before it consumes a node; a job that cannot succeed fails within seconds of starting.
4. Large-model runs have an explicit memory contract and never tokenize images.
5. Results, manifests, and the dashboard live on durable storage and can be rebuilt from manifests.
6. Every change lands as reviewable commits in the canonical GitHub repositories.

Non-goals for this iteration:
- Rewriting the bash launchers in Python. The launchers keep their interface; new logic is added as Python modules they call.
- Changing benchmark semantics, prompts, or metrics.
- Syncing the harness forks with upstream before the contracts exist. The sync is a later phase gated by the contracts, because the last bump introduced a silent regression the contracts would have caught.

## 3. Contracts

### C1. Task registry: `suite/tasks.yaml`
One declarative table for every benchmark the suite runs. Fields per task: `framework`, `harness_task`, `category`, `headline` metric, `judge` (provider or none, plus the env var it reads), `assets` (env var, relative path, source URL, extraction, minimum file count), `max_model_len` override, `multi_image`, and membership flags `card` (HF model card set) and `report` (technical report table). Consumers: preflight, the launchers, the dashboard, the dataset restorer, and the judge suite lists, which become generated from it and verified against it. `scripts/make_dashboard.py` keeps its `BENCHMARKS` dict shape but obtains it from the registry.

### C2. Run manifest: `run_meta.json`
Written by every job before inference and finalized after. Start fields: run id, framework, task, model name, model path and real path, config hash, weight index hash and per-shard size and mtime, tokenizer hash, chat template hash, requested thinking flag, generation settings, TP and DP, batch size, memory utilization, context length, harness name and commit and dirty flag, suite commit and dirty flag, container image path and size, Slurm job id and node, dataset assets with file counts. Finalize fields: status (`ok`, `failed`, `invalid`), error excerpt, results file, sample count, output-token statistics, effective thinking flag, canary result. A run without a manifest is `legacy` on the dashboard.

### C3. Exit status
A job's exit code reflects evaluation success. After the harness returns, a finalizer inspects the log and the results directory and sets the manifest status; the job exits non-zero on `failed` or `invalid`. `invalid` means the run completed but a contract was violated, for example thinking requested and mean output tokens below a threshold.

### C4. Effective-configuration assertion
The lmms-eval Apertus wrapper assigns the thinking flag after the base constructor and logs the effective value. When thinking is requested it runs a text-only canary through the engine on first use and raises if no deliberation block appears. The VLMEvalKit wrapper already applies the flag correctly and gains the same canary. The finalizer reads both signals into the manifest.

### C5. Preflight
`python -m suite.preflight` checks, for a given model, framework, tasks, and flags: model files resolve and are readable, tokenizer and template exist, the vision tokenizer weights exist, each task's assets exist with at least the declared file count, the judge key is present when a task needs one, the container image exists, and the context length satisfies every task. The launchers run it before `sbatch` and refuse to submit on failure. The job scripts run it again before loading the model.

### C6. Resource contract for large models
A tokenize-only mode fills the image-token cache for a task list using only the VQ encoder across the node's GPUs, without loading the language model. The 70B profile runs with the cache in read-only strict mode so a miss fails immediately instead of encoding on the worker's GPU. The encoder additionally releases cached blocks after every render so that when encoding is allowed, its footprint stays bounded. The 70B memory utilization default is 0.75.

### C7. Durability and observability
The canonical checkout lives on capstor, so results and logs are durable by construction. The dashboard is built from manifests, exports its JSON next to the HTML, and prints a coverage report listing every missing cell with its cause. A status command reads manifests and Slurm accounting and prints per-run states, replacing ad-hoc log grepping. The registry is validated against manifests at build time.

### C8. Partial runs never publish

A run launched with a sample limit (gate checks, smoke tests, timing pairs) records that limit in its manifest, and every dashboard collector skips such runs. They are kept under `cache/validation/results/<harness>` rather than `results/` so that "newest result wins" can never let a 96-sample check displace a full sweep; the 32-sample gsm8k gate run had done exactly that before this rule existed. The manifest also records the harness checkout the job actually imported (worktree or pinned submodule), so a number's provenance survives a sync.

### C9. The registry names the headline metric

`[dashboard.<task>]` declares the ordered metric names that make a benchmark's headline (`headline`), the names to use when a harness only aliases the task (`lmms_headline`), and whether nothing else may stand in (`headline_strict`). Script code keeps only computation and fallback: the MME total, secondary rows such as the MMVP pair accuracy, and the generic priority for tasks the registry does not describe. The migration from the script-side priority list was gated on a cell-for-cell identical dashboard rebuild.

## 4. Architecture

New Python package `suite/` at the repository root, importable by scripts, launchers, and job scripts through the existing `shared/` path mechanism:

- `suite/tasks.yaml` and `suite/tasks.py`: registry and loader, with `--check` to verify generated suite lists.
- `suite/manifest.py`: `start` and `finalize` subcommands.
- `suite/preflight.py`: checks and a machine-readable report.
- `suite/stage_datasets.py`: restores declared assets from their sources; generalizes the ad-hoc restoration done today.
- `suite/tokenize_cache.py`: tokenize-only pass over a task's images.
- `suite/status.py`: run states from manifests and accounting.
- `tests/`: pytest suite for the package, using temporary directories and fixture manifests; no GPU.

Hooks:
- `launchers/lmms-eval/eval.sh`, `launchers/VLMEvalKit/eval.sh`, `launchers/lm-eval/eval.sh`: call preflight before submission; pass the run id.
- `slurm/*/eval_job.slurm`: preflight, manifest start, harness, manifest finalize, exit with its status.
- `scripts/make_dashboard.py`: read manifests, coverage report, JSON export, registry validation.

Harness forks:
- `swiss-ai/lmms-eval`: wrapper fix, canary, GEOBench presence filters bypass the datasets cache.
- `swiss-ai/VLMEvalKit`: recreated vsibench commit, canary.

## 5. Data flow

Launch: user runs a launcher, which resolves tasks through the registry, runs preflight, and submits one job per task with the run id. Job: preflight again, manifest start, harness, manifest finalize, exit code. Dashboard: refresh script walks results roots, reads manifests, builds the page, exports JSON, writes coverage.

## 6. Error handling

Preflight failures list every failing check and exit 2. Manifest finalize exits 3 on `failed` and 4 on `invalid`; the job script propagates it. The canary raises inside the harness so the failure shows up as `failed` with the canary message in the error excerpt. The read-only cache raises on miss with the task and cache path in the message and a pointer to the tokenize-only command.

## 7. Testing

Unit tests for the registry loader and validator, manifest start and finalize including the invalid-thinking rule, preflight against synthetic directory trees, the dashboard coverage report, and the suite-list generator. Integration on the cluster, using benchmarks with no judge: the 8B on POPE, MMVP, and MMStar to confirm the launch path and exit codes reproduce production numbers; the 8B thinking on MMVP to confirm the canary and manifest statistics; a tokenize-only pass followed by a 70B read-only run on MMVP; and a deliberately broken preflight. Integration runs only after the unit tests pass and only when the owner has not paused submissions.

## 8. Harness sync

After the contracts land: merge upstream `EvolvingLMMs-Lab/lmms-eval` and `open-compass/VLMEvalKit` into the forks on branches, update `lm-eval-harness` to a tagged release, run each harness's own tests where feasible, then the integration set above. A sync that changes any production number beyond noise blocks until explained.

## 9. Assumptions and non-compatibility

- Existing results without manifests remain visible as `legacy`; no attempt is made to backfill provenance the purge destroyed.
- Task names in the registry are the dashboard task names; harness task ids map from them.
- The scratch checkout is retired for launching once the capstor checkout passes integration.
- No backward compatibility for the hand-maintained judge suite lists; they become generated.

## 10. Phases

1. Foundation: capstor clone, rescued commits, carried-over local work, mirrored results. Done or in progress.
2. Contracts: registry, manifest, exit status, canary.
3. Preflight, dataset restorer, tokenize-only mode, read-only large-model profile.
4. Observability: dashboard from manifests, coverage, JSON export, status command.
5. Validation and pull requests.
6. Harness sync.

## 11. Open questions for the owner

- Whether the xBD portion of GEOBench should be restored from the xView2 download or permanently excluded with the drop count recorded in the manifest.
- Whether TP4 remains the 70B default given the measured TP2xPP2 slowdown (11 percent prefill, 17 percent decode), which this design assumes.

## 12. Harness sync record (phase 6, started 2026-09-08 late)

Measured distance before the sync: lmms-eval fork 45 commits behind `EvolvingLMMs-Lab/lmms-eval` main (3b72e104, 2026-09-07) and 75 ahead; VLMEvalKit fork 20 behind `open-compass/VLMEvalKit` main (d21c5e9, 2026-09-07) and 75 ahead; lm-eval-harness pinned at 8a07e111 (2026-08-14), latest tag v0.4.13 (2026-09-01).

Merges live on `yxu/sync-upstream-2026-09-08` in both forks, built in worktrees so the pinned checkouts stayed untouched during validation.

- lmms-eval conflicts: the vLLM base's sampling-parameter handling and video decoding follow upstream (`read_video` with a selectable backend replaces the decord path); the base's batched chat loop keeps the fork's version, which carries the chat-template and tokenization kwargs that foreign models need; the evaluator's sample logging follows upstream, which now records `token_counts` itself, with the fork's oversized-string guard re-applied; MMMU and VLMsAreBlind scoring follow upstream; `logging_utils.py` stays removed; `tools/batch_watchdog.py` follows upstream. New declared dependencies: jieba, distance, editdistance, Levenshtein, apted, ruff, pytest; only task modules import the first five.
- VLMEvalKit conflicts: `run.py` follows upstream's per-dataset judge resolver; the fork's 3DSRBench exact-matching override moves onto its dataset class as `DEFAULT_JUDGE_MODEL`, MMSafetyBench already declared gpt-4o-mini. New declared dependency: rdkit (chemistry datasets only).
- lm-eval-harness: candidate is the v0.4.13 tag; the only change touching the suite's path is 23 lines in `vllm_causallms.py`.

Gate: the merged harnesses must reproduce the pinned harnesses' numbers on deterministic (temperature 0) runs of the released 8B, MMVP and POPE at 96 samples on lmms-eval, MMVP on VLMEvalKit, GSM8K at 32 samples on lm-eval, and the thinking canary must pass on the merged lmms-eval. Pointer bumps land only after the gate, on a branch stacked on the hardening branch.

### 12.1 Dropped-change audit (2026-09-08, 03:00 to 03:40)

A merge that builds is not a merge that kept the fork. `scripts/sync_audit.sh` compares every file the fork changed since the merge base against the merged tree: byte-identical to upstream means the fork's change was dropped; fork-added lines absent from the merge means partial. Findings and their resolution live in `docs/sync/2026-09-08-{lmms-eval,vlmevalkit}.accept`; the gate exits non-zero on any finding not in the list.

| Harness | Fork-changed files | Dropped | Partial | Resolution |
|---|---|---|---|---|
| lmms-eval | 169 | 4 | 2 | `vlmsareblind/utils.py` answer normalization restored on top of upstream's aggregation; `mmmu/utils.py` and `aero_realtime_vllm.py` superseded by equivalent upstream changes; upstream's `tools/batch_watchdog.py` kept and its test restored |
| VLMEvalKit | 61 | 2 | 0 | `run.py` had been taken from upstream wholesale, losing `--response-cache` and its pass-through, the rank-to-GPU split, the absolute-model-path work_dir fix, the shadow-file filter, and the removal of `--use-vllm` forwarding to the judge; all restored by a three-way apply, judge resolver kept upstream's; `mmlongbench_metrics.py` reset to upstream with only the lazy `rouge_score` import re-applied |

### 12.2 Residual triage (fork over upstream after the merge)

Every remaining fork-versus-upstream difference was classified: fork-only files (Apertus wrappers, tokenizer bridge, our tasks: 109 in lmms-eval, 20 in VLMEvalKit), whitespace-only churn (reset to upstream: the three covost2 yamls), useless (removed: a duplicated `ocrbench_v2` block under `[tool.setuptools.package-data]`; import reordering and a trailing newline in mmlongbench files; a cosmetic indentation hunk in `run.py`), and real residual fixes. `simple/vllm.py` is rebuilt as upstream plus the fork's deltas (per-request sampling params through the TP gather, `chat_template_kwargs` and `tokenization_kwargs` pass-through, `enable_thinking`, `_format_context`), which also stops deleting upstream's watchdog heartbeats. Residual fixes that change what a benchmark measures are kept deliberately and must not be treated as churn in a later sync: answer normalization in chartqa, pope, scienceqa, countbench, vlmsareblind, the MMLU generative filter, the refcoco/screenspot bracket regex, MathVista extraction, and the VLMBlind scorer in VLMEvalKit.

### 12.3 SLAKE name collision

Upstream lmms-eval added a `slake` group (slake_en + slake_zh, overall accuracy) with the same name as the fork's MedEvalKit task (English test split, closed-question accuracy, the card's number). lmms-eval registered one as `slake-1`, so which definition `slake` resolved to depended on directory order; the 2026-09-08 runs of both corrected checkpoints measured the upstream group. The fork task is renamed `slake_medevalkit`, the registry points `slake` at it, dashboard rows are keyed by registry name, and preflight now fails a launch whose harness id is declared in two files or in none. Both corrected checkpoints were rerun on the MedEvalKit definition on 2026-09-08 ~06:10 (run ids `20260908T_8b_slake_medevalkit`, `20260908T_70b_slake_medevalkit`): released 8B closed-question accuracy 77.4, corrected 70B 70.9. The first rerun attempt had measured the upstream group again because the launcher handed the registry name to the harness; it now passes the harness id the registry declares.

### 12.4 Validation record (deterministic, released 8B, temperature 0)

| Harness | Pinned | Candidate | Run | Result |
|---|---|---|---|---|
| lmms-eval | 8fd62f0f | e8e1b2ac | mmvp, pope at 96 samples | identical metrics (mmvp 0.625 / pair 0.3542, pope 0.8125), re-run after the wrapper rebuild identical |
| lmms-eval | 8fd62f0f | e8e1b2ac | mmvp at 8 samples, thinking | canary passed, mean 444 output tokens |
| lmms-eval | 8fd62f0f | e8e1b2ac | vlmsareblind at 48 samples | 48/48 identical generations, accuracy 0.0833 |
| VLMEvalKit | b44d4ca | 5f34422 | MMVP, fresh response-cache root on both sides, both run orders | 300/300 identical predictions, 0.6833 / 0.4067 |
| lm-eval-harness | 8a07e111 | v0.4.13 (ddd67220) | gsm8k at 32 samples | 32/32 identical generations, 0.6562 |

A first VLMEvalKit comparison replayed the baseline's answers from the shared response cache (75 lookups, 75 hits, 0 misses) and was discarded; a VLMEvalKit candidate must run with a fresh `--response-cache` root, and the gate scripts now do. Every gate run's manifest names the worktree it imported and its sample limit, and lives under `cache/validation/results/` (contract C8).

Pointers move on `yxu/harness-sync-2026-09-08`, stacked on the hardening branch (swiss-ai/MLLM-eval-suite#9; fork PRs swiss-ai/lmms-eval#25 and swiss-ai/VLMEvalKit#9).

### 12.5 Does the sync change inference speed?

Wall-clock seconds per run on one node of the coding allocation, the same 8B and settings. These small comparisons include cold initialization and variable cache state; the table does not show a consistent benefit from either the candidate or running second. They establish no reliable harness-version speedup. Both sides use the same engine image (vLLM 0.26.1). The vLLM 0.28 trial image requires a separate controlled comparison before making a speed claim.

| Pair | Pinned first | Merged second | Merged first | Pinned second |
|---|---|---|---|---|
| lmms-eval mmvp, 96 samples | 149 | 98 | 209 | 306 |
| VLMEvalKit MMVP, fresh response cache | 442 (cold) / 185 | 363 | 309 | |
| lm-eval gsm8k, 32 samples (whole job) | 375 | 146 | 484 | 496 |
| lm-eval gsm8k, generation phase only | 1:59 | 1:29 | 2:03 | 1:26 |

### 12.6 Does the vLLM 0.28 trial image change inference speed or numbers?

Sbatch jobs on the reservation, released 8B, 4-way data parallel on one node each, two repetitions per image, the production image (vLLM 0.26.1) against the trial image (`cache/image-builds/vllm-51da0ca66-20260907`, vLLM 0.28.1). Seconds from `run_meta.json` started to finished, which spans model load and generation inside the container.

| Task | Production r1 | Production r2 | Trial r1 | Trial r2 |
|---|---|---|---|---|
| mmvp, 300 samples | 168 | 191 | 125 | 141 |
| mmstar, 1500 samples | 306 | 303 (replacement; the original took 1906 on a slow node) | 271 | 270 |

The trial image is faster in every pair: about a quarter less on MMVP and about a tenth less on MMStar. The original second production MMStar job generated at 3.7 seconds per step on a node whose siblings ran several times faster, produced the same score, and is replaced by a rerun for the timing table. Headline scores agree within 0.3 points (mmvp 68.67 versus 69.0; mmstar 44.83 on all three production runs versus 44.83 and 44.87 on the trial image), so the production image is repeatable run to run at this sample size and the trial image varies by one sample. The trial image stays unpromoted until a full sweep reproduces the production numbers; these runs live under `cache/validation`.

### 12.7 Released 8B text numbers: the purged `Apertus-v1.5-8B` tree

The collision gate added in the review follow-up flagged two lm-eval trees under the released 8B column: `results/lm-eval/Apertus-v1.5-8B` (2026-08-17 runs on `cache/models/textview/Apertus-v1.5-8B`, a directory the scratch purge removed) and `results/lm-eval/8B-Final-correct-rope` (the 2026-08-18 `final_text` run on the checkpoint that still resolves to the capstor release weights). Both used identical lm-eval settings, yet they disagree by 6.6 points on ARC-Challenge in one direction and 17 points on IFEval in the other, which is a configuration difference, not noise. Because the older directory no longer exists, its numbers cannot be attributed to the release, so the alias is split: the released 8B columns keep only what `8B-Final-correct-rope` produced, and the older runs, including the 2026-08-18 thinking-mode text runs on the same purged directory, are shown under their own labeled columns. The text tasks that lose their released-8B cell (aime24, aime25, hmmt_feb_2025, math500_verify, math_lvl5_verify, hellaswag, mmlu, mmlu_pro, gsm8k) were resubmitted on the release checkpoint on 2026-09-08 ~18:00 (run id `20260908T_8b_text_rerun`). The lm-eval launcher has no thinking mode, so the thinking-column text cells stay attributed to the older tree until that path exists.
