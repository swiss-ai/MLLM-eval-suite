# MLLM-eval-suite — Improvement Note

> Synthesized from a 7-subsystem parallel audit (readers + adversarial verifiers + synthesis, 74 raw findings verified), re-grounded against HEAD on `yxu/bump-lmms-eval`, 2026-07-05. Framing: **single admin maintains internals; others clone-and-run the launch script.** Findings are sorted into that model — *others-launch* (high bar) vs *admin-maintenance* (complexity OK unless fragile/drifting).

> Delta since the 07-02 `eval_suite_audit.md`: both fork trees now clean, judge `gpt-4-turbo` hotfix committed (`24645ce`, `f1a46ec`), submodule pins bumped; lm-eval/NeMo PR #1 still open and unmerged. The remaining work is **decoupling and wiring, not firefighting.**

## Verdict

**Clone-and-launchable for another user today: not for the full/dual path; marginally for a single-framework smoke run.** A different user on a different node cannot submit the advertised one-command run because the SLURM templates hardcode an expiring reservation (`SD-69241-apertus-1-5-0`, EndTime 31 Jul, still ACTIVE), `--account=infra01`, and an `--environment` pointing at the *admin's* capstor clone — with no launcher override flag — while the README quick-start invokes four files that don't exist and the `all` mode aborts half-submitted on disjoint `--mode` value spaces. **Maintainable for the admin: good bones, eroding at the edges.** The layer contracts (ORCH_REPO_ROOT guard, suite-file-as-data, slurm preflight asserts, the one shared `image_token_cache_env.sh`), the env-driven checkpoint registry, and the centralized metric policy with honest curation are genuinely strong; but three drifting submission stacks, personal paths plus `|| true` in the dashboard pipeline, a verifier that was written for a real incident yet never wired in, and mtime-based cell selection with no dataset-name guard mean any published number is re-derivable only by the admin on this cluster. The encouraging delta since 07-02: both "time bombs" are partly defused (judge hotfix committed, both forks clean, pins bumped), so the remaining work is **decoupling and wiring, not firefighting.**

---

## A. Ease of use for others (clone-and-run path)

### A1. One-account/one-cluster hardcoding blocks every non-admin submission — no override flag `[M]` — Critical (others-launch, compounds admin drift)
Reservation + `--account` + `--environment` are baked into every SLURM template, and the `--environment` toml lives in a *second clone* on capstor (two writers, drift-by-luck), so a cloner on their own scratch loads the admin's container config, not their repo's.
- `slurm/lmms-eval/eval_job.slurm:2,4,7`; `slurm/VLMEvalKit/eval_job.slurm:2,4,7`; `slurm/shared/healthbench_grader_server.slurm:2,4,7` (+`:18` grader model path).
- Reservation `SD-69241-apertus-1-5-0` → `scontrol`: `EndTime=31 Jul 12:00, State=ACTIVE`. Every submission breaks after that date.
- Launchers parse no `--reservation/--account/--environment` (lmms `eval.sh` argparse ~`:84-102`, VK `eval.sh` ~`:109-167`).
- **Compound signal:** the post-audit `healthbench_grader_server.slurm` replicated the identical triple — evidence the `#SBATCH` header has no shared contract (ties to B1).
- **Fix:** add `--reservation/--account/--environment` passthrough flags (default to current values) plumbed into the `sbatch` invocation; make `--environment` resolve to the submitting repo's `toml/shared/…` instead of the capstor absolute path. One change clears the single largest others-launch blocker and one drift source.

### A2. README cannot get a teammate through a run `[S]` — High (others-launch)
The entry-point doc references files that do not exist and examples that exit 2.
- Missing paths, all referenced: `launchers/run_eval.sh` + `toml/VLMEvalKit/example.toml` (`README.md:59,65`), `launchers/backends/` (`:26`), `task_lists/` (`:29`); `run_meta.json` (`:103`) is written by nothing.
- Framework-launcher header examples (`launchers/VLMEvalKit/eval.sh:5-9`) exit 2 under the ORCH_REPO_ROOT guard; `slurm/README.md` misstates which toml the templates use.
- **Fix:** rewrite the quick-start to the real `launchers/eval.sh --model … --suite …` entrypoint; delete the phantom-file section and the `run_meta.json` claim.

### A3. `all` mode is incoherent — the one-command dual-harness promise aborts half-submitted `[M]` — High (both)
The unified launcher forwards one `--mode` value to two harnesses with disjoint value spaces and different suite defaults; under `set -euo` lmms submits first, then VK rejects the mode and kills the run mid-flight.
- `launchers/lmms-eval/eval.sh:109` accepts only `fill|readonly` (default `fill`, `:81`); `launchers/VLMEvalKit/eval.sh:61` accepts only `all|infer|eval` (default `all`, `:37`; suite default `smoke` vs lmms `full`). `launchers/eval.sh:186,203` pass the same `MODE_ARG` to both.
- **Fix:** either give `all` mode a harness-aware mode map, or drop the shared `--mode` and document per-framework invocation as the supported dual path. Whichever, make the default `all` run reproduce the dashboard's actual set without aborting.

### A4. Judge suites silently degrade to regex parsing — no OPENAI key plumbing, no error `[S–M]` — High (both)
No key handling exists anywhere on the launch path, so a user running `llm_judge` gets exact-match fallback with no signal — already happening in production.
- `grep OPENAI_API_KEY/URL` over `launchers/ slurm/ toml/` → nothing.
- Live proof (audit critic): `logs/VLMEvalKit/…/vlmeval-MMSafetyBench_*.err` → "Judge model is not working, fallback to exact label parsing".
- **Fix:** plumb `OPENAI_API_KEY` (env → sbatch → container) and **fail loud** when a judge suite is launched without it, rather than silently scoring by regex.

### A5. `--enable-wandb` is a dead flag with a scary-but-false warning `[S]` — Low (others-launch confusion)
Launcher defaults `ENABLE_WANDB=true` and warns "Jobs will fail fast" without a key; the SLURM template forces it off and ignores the flag.
- `launchers/lmms-eval/eval.sh:197` (default true), `:204` (warning), `:291` (passes) vs `slurm/lmms-eval/eval_job.slurm:79` (`ENABLE_WANDB=false`), `:184` ("W&B is disabled in this script; ignoring").
- **Fix:** default the launcher to `false` and delete the false warning, or honor the flag in the template. Pick one side of the contract.

---

## B. Reduce complexity / maintenance burden (admin surface)

### B1. Three drifting submission stacks + intra-parent duplication — name the shared abstraction `[M–L]` — High (admin)
The parent stack is authoritative, but each fork still ships a full superseded copy, and they have already drifted — including a copy that computes a *different headline metric*.
- Fork copies: `third_party/lmms-eval/examples/apertus-vllm/scripts/{eval.sh,eval_job.slurm,gather_results.py,push_results_to_wandb.py}` — its `gather_results.py` **differs** from `scripts/gather_results.py` (headlines `mme_perception_score` vs the repo's `mme_total`); `third_party/VLMEvalKit/scripts/apertus-vllm/{eval.sh,eval_job.slurm,suites/}` — `suites/llm_judge.txt` drifted from `task_suites/VLMEvalKit/llm_judge.txt`.
- Intra-parent: Emu3.5 prefetch ×3 (`launchers/eval.sh` + `slurm/VLMEvalKit/eval_job.slurm` + `slurm/lmms-eval/eval_job.slurm`); model-label derivation ×3 (`launchers/lmms-eval/eval.sh:250-252` +2); `resolve_list` divergent (`launchers/lmms-eval/eval.sh:115-126` strips comments+commas vs `launchers/VLMEvalKit/eval.sh:92-101` commas only).
- **Fix:** delete the two `examples/apertus-vllm` / `scripts/apertus-vllm` stacks from the forks (they are dead relative to the parent); lift prefetch + model-label + `resolve_list` into `slurm/shared/` the way `image_token_cache_env.sh` already demonstrates. The drifted `gather_results.py` metric divergence compounds B3 (a stale duplicate silently produces a different number).

### B2. Dashboard pipeline hardwired to personal paths + swallows failures `[S]` — High (admin bus-factor)
No second admin can regenerate `docs/index.html`, and breakage produces stale numbers with no signal.
- `scripts/derive_vlmeval_acc.py:15` `SUITE = Path("/iopsstor/scratch/cscs/xyixuan/apertus/MLLM-eval-suite")` — **not** env-overridable, drives `:16 LOGS`, `:17 RESULTS`, `:22 task_suites`.
- `scripts/refresh_dashboard.sh:56` `TRUNC_TOOL` defaults to a personal checkout **outside the repo** (`/iopsstor/scratch/cscs/xyixuan/apertus/lmms-eval/…/truncation_report.py`); `:9` `RUNS_ROOT` personal default; `:21` and `:64` `|| true` swallow derive + truncation failures.
- **Fix:** derive the repo root from `git rev-parse --show-toplevel` (or script dir); vendor `truncation_report.py` into `scripts/`; drop the `|| true` guards so a failed derive/truncation aborts the refresh.

### B3. The verifier exists but is unwired; cells picked by mtime with no dataset-name guard `[M]` — High (admin trust)
The exact contract for the "wrong number on the page" incident class is written and never enforced.
- `scripts/verify_dashboard.py` has **zero callers**; `refresh_dashboard.sh:69` runs `make_dashboard.py` directly. `verify_dashboard.py:26-36` parses only lmms `*_results.json`, so all 19 VLMEvalKit rows regenerate unaudited.
- `scripts/make_dashboard.py:249` globs `**/*acc*.csv` within each benchmark dir and takes the newest with **no assertion that the CSV's dataset matches the row** — a foreign `acc.csv` dropped into a bench dir would be silently ingested. Currently *latent* (spot-checked 2026-07-05: no such contamination on the page; the shadow-name filter + `verify_dashboard.py` catch the collision class, but the verifier is run manually, not wired — see below).
- `scripts/gather_results.py:34-46` selects newest by `st_mtime` — a re-run with a regression silently replaces a good cell; no run pinning.
- **Fix:** wire `verify_dashboard.py` into `refresh_dashboard.sh`, extend it to the VK CSV trees, and add a dataset-name assertion at the `accs[-1]` pick. This enforces the "validates-once" contract that already exists in intent.

### B4. Runtime code isn't what git pins (lmms side) `[M]` — Medium (admin reproducibility)
lmms-eval executes from the container-baked `/workspace/lmms-eval`, built from a moving branch with no SHA recorded, so the submodule pin the audit reviewed is not what runs.
- `slurm/lmms-eval/eval_job.slurm:92,236` put `${CONTAINER_REPO_ROOT}/lmms-eval` at the head of `PYTHONPATH`; `dockerfiles/Dockerfile` `ARG LMMS_EVAL_BRANCH` (no SHA). (The VLMEvalKit dirty-tree half of this is now resolved — tree clean.)
- **Fix:** record the fork SHA at image build and echo it in the job log. The active `yxu/bump-lmms-eval` branch is the moment to pin this.

### B5. Dead code: `summarize_results.py` (549 lines, zero callers, wrong layout) `[S]` — Low (admin)
A second complete HTML dashboard generator expecting an array-run layout nothing produces.
- `scripts/summarize_results.py` — no invocations anywhere; `:47` expects `run_root/results/<task>/` which the launchers never emit.
- **Fix:** delete it.

### B6. Coverage: defined-but-never-run and produced-but-not-ingested `[S–M]` — Medium (admin curation)
Several capabilities are advertised on the page's premise but carry no data, and one silent-skip path hides real results.
- 27 audio tasks: `task_suites/lmms-eval/audio_full.txt` (22) + `audio_llm_eval.txt` (5), zero results, zero rows.
- GUI (ScreenSpot×4), 3DSRBench, VSI-Bench ran but lack `acc.csv` → `make_dashboard.py:229-231` silently skips.
- `mmlu_medical` (8.1–63.0 swing) kept though `mmlu_flan` was dropped for the identical "format-fragile exact-match" reason (`make_dashboard.py:58-62`).
- MMSafetyBench scores were produced (`MMSafetyBench_score.csv` for 4/6 models) but never ingested (no `acc.csv`) — compounds A4.
- **Fix:** decide commit-or-delete per capability (run audio or remove the suites); turn the no-`acc.csv` silent skip into a loud warning; drop or annotate `mmlu_medical`.

### B7. Point-paren repair rewrites free-form math answers unconditionally `[S]` — Medium (admin, fork-internal correctness)
The EASI `[x, y) → [x, y]` fix is applied to every dataset, so a legitimate half-open interval in a MathVista/MathVerse/LogicVista answer is silently flipped; the gating info is already in hand but unused.
- `third_party/VLMEvalKit/vlmeval/vlm/apertus_1p5.py:200` (`_POINT_PAREN_RE.sub`, regex `:27`); `generate_inner` receives `dataset` (`:185`) but never consults it.
- **Fix:** gate the substitution on the spatial/grounding datasets it was written for, using the already-available `dataset` arg.

### B8. Governance / bus factor: public repo, stale main, no parent CI `[M]` — Medium (admin)
Everything reviewed lives on a personal branch; `main` shows none of it.
- `origin/main` is **61 commits behind** HEAD (was 42 at audit); no PR to land it. Repo is public; `docs/index.html` with unreleased-checkpoint numbers is served from the personal branch. No `.github/` (zero parent CI), no tests on `metric_selection.py`/`make_dashboard.py`/`derive_vlmeval_acc.py` — the code computing every published number.
- **Fix:** land the operational branch (or a docs-only branch) to `main`; add one smoke test over `metric_selection`/`make_dashboard` on a fixture; confirm public exposure of the checkpoint numbers is intended.

### B9. Geo PR #13 landing nits (fork) `[M]` — Low (admin, if finishing the landing)
Tasks are in the fork; suite-side runnability and a few code nits remain.
- Env-var staging (`VRSBENCH_DIR`/`GEOBENCH_DIR`/`FRIEDA_DIR`, BigEarth LMDB) is referenced nowhere in `task_suites/ launchers/ slurm/ toml/`; `lmdb` is undeclared in `third_party/lmms-eval/pyproject.toml` (METEOR/Java already resolved — openjdk-17 in Dockerfile). Code nits: caption-scorer block copy-pasted ×3 (bigearth/geobench/vrsbench utils — same "name the abstraction" as B1), FRIEDA F1 uses `set` instead of `Counter` overlap, bbox rescale inconsistent (`geobench_ref` `/100` vs `bigearth_bbox` raw).
- **Fix:** plumb the staging env vars through the slurm template, declare `lmdb`, extract the caption scorer; the F1/bbox nits are optional correctness polish.

---

## C. Deliberately leave as-is (do not over-refactor)

- **Already resolved since the 07-02 audit — do not re-open:** judge `gpt-4-turbo` hotfix is committed (`third_party/VLMEvalKit` `24645ce`, `f1a46ec`); both submodule working trees are clean (apple.jpg / task jsonls / TUI dist restored); lmms-eval object-store corruption repaired; pins bumped (VK `bc6652f`, lmms `9a94c879`); geo PR #13 merged into the fork with `geospatial_full/smoke.txt` wired.
- **The layer contracts are essential complexity — keep them:** ORCH_REPO_ROOT guard, suite-file-as-data discovery, the slurm GPU/TP preflight arithmetic and the path-shaped-model loud-fail backstop, and `slurm/shared/image_token_cache_env.sh` as the single-source cache contract. These are the good version of the abstraction B1 asks you to extend, not rewrite.
- **The env-driven checkpoint registry** (VLMEvalKit `058d317`, `APERTUS_RUN_NAME`/`APERTUS_MODEL_PATH`) replaced a 55-entry hardcoded table — keep; do not regress to per-checkpoint entries.
- **Centralized metric policy + honest curation** (`metric_selection.py`, judge-fail-rate guard, documented CharXiv/cmmmu/contaminated-cell drops, per-cell provenance) is above internal-tooling norm — keep.
- **The `response_cache.py` design** (per-rank SQLite merged at finalize, failure markers excluded, deterministic-only gating) — keep.
- **Two maintained forks** is inherent to the two-harness mission — do not attempt to collapse to one. Rebase debt and upstream divergence are real but are the cost of the mission, not accidental complexity.
- **Proportionality:** the audit's B-grade is about *trust wiring*, not code quality. Do not rewrite battle-tested slurm machinery for elegance; the cache-fingerprint staleness on warm reruns is a known EASI tradeoff, not a defect to chase unless a stale-typo cell actually surfaces.

---

## D. Suggested sequence (dependency-ordered)

1. **Decouple from one account/cluster (contracts first).** A1 + the shared-header half of B1: add `--reservation/--account/--environment` passthrough flags across the three SLURM templates and fold their `#SBATCH` preamble into one shared snippet; point `--environment` at the submitting repo's toml. This is the single biggest others-launch unblock and kills the capstor second-clone drift at the same time.
2. **Re-establish the source-of-truth contracts on the admin side.** B2 (personal paths + `|| true` out of the dashboard pipeline), B3 (wire `verify_dashboard`, extend to VK, add the dataset-name guard, drop mtime-only selection), B4 (record the runtime fork SHA on `yxu/bump-lmms-eval`). These must land before any doc describes the flow, because they define what "the published number" means.
3. **Make the launch path coherent, then document it.** A3 (fix/replace `all`-mode), A4 (judge-key plumbing + fail-loud), A5 (wandb flag), then A2 (rewrite the README quick-start to the now-coherent entrypoint). Docs come after the flow stops half-aborting.
4. **Delete drift and dead surface.** Remainder of B1 (remove the two `*/apertus-vllm` fork stacks; lift prefetch/model-label/`resolve_list` to `slurm/shared/`), B5 (`summarize_results.py`), B7 (gate the point-paren regex).
5. **Curation + governance.** B6 (commit-or-delete audio/GUI/MMSafety; loud no-`acc.csv` skip), B8 (land `main`, add a minimal metric/dashboard test), B9 (finish geo-PR staging + nits if committing to those benchmarks).