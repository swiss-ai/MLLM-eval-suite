# Scripts (admin surface)

Post-processing and operations tooling. Users never need this directory — results ingestion,
dashboard regeneration, benchmark staging, and re-grading are admin flows.

## Results → dashboard flow

```
results/lmms-eval/<model>/<run-id>/**/*_results.json ─┐
runs root (capstor, RUNS_ROOT) ───────────────────────┤
results/VLMEvalKit/<run-id>/<model>/<dataset>/  ──────┤   derive_vlmeval_acc.py  (judge logs → derived_acc.csv)
                                                      ├─► refresh_dashboard.sh
                                                      │     ├─ bridges VK runs into RUNS_ROOT (cache/vlmeval_bridge,
                                                      │     │  union of runs, <bench>__<run-id> links)
                                                      │     └─ make_dashboard.py  → docs/index.html
                                                      │          (headline metrics via metric_selection.py,
                                                      │           newest result per (model, task),
                                                      │           TAXONOMY bands, curated columns: CURATED array)
                                                      └─► verify_dashboard.py  (collision audit — run after refresh)
```

- **`refresh_dashboard.sh`** — one-shot regeneration of `docs/index.html`. Knobs: `RUNS_ROOT`
  (lmms results tree), `VLMEVAL_OUTPUTS`, `TRUNC_TOOL`. Column curation lives in
  `dashboard_models.txt` (`key[|alias]=Label`, one line per dashboard column).
- **`make_dashboard.py`** — the generator. Benchmark→category mapping is the `TAXONOMY` dict;
  display-dropped tasks are `DROPPED_TASK_PREFIXES` (drop policy for *runs* lives in `task_suites/`).
  Audio report headings map to native checkpoint identities, with display labels kept
  separately. Historical report cells fill gaps; native results take precedence, including
  after manifest alias merging. `--models` accepts report labels or canonical keys, and a
  pretrain-only filter retains all 13 supplied FLEURS values.
  Error-rate rows and homogeneous error-rate means are lower-is-better. Micro means
  weight common rows equally; macro means weight their categories equally. Cards and
  category bands omit a combined mean when their common cohort mixes error rates and
  higher-is-better scores. Raw values and source runs remain available on each cell.
- **`metric_selection.py`** — single source of truth for each benchmark's headline metric and score
  normalization; imported by every reporting script.
- **`gather_results.py`** — newest-result-per-task selection + CLI table.
- **`verify_dashboard.py`** — audits canonical-model-key collisions (the wrong-number-on-page
  failure class). Run it after every refresh; PASS expected.
- **`derive_vlmeval_acc.py`** — reconstructs `derived_acc.csv` for VK judge benchmarks whose score
  only exists in job logs. Skips runs with high judge-failure rates rather than fabricating scores.
  Its derived metrics are higher-is-better (mm_safetybench: attack_rate -> safety_rate,
  labeled `safety_rate` on the row). Audio WER/CER retain their native error-rate direction.

CPU dashboard and launcher regressions: `python3 -m unittest discover -s tests -v`.
The browser-logic test runs the generated inline script with Node.js and a minimal DOM
sink, without extra packages; set `NODE=/path/to/node` if Node is not on `PATH`.

## HealthBench grading

- Serving the Qwen judge (Meditron protocol): `sbatch "${SBATCH_OVERRIDES[@]}"
  slurm/shared/healthbench_grader_server.slurm` (source `slurm/shared/sbatch_overrides.sh` first).
  It writes `host:port` to `cache/grader/healthbench_endpoint` once ready; point
  `HEALTHBENCH_GRADER_BASE_URL=http://<host>:8080/v1` at it when submitting healthbench jobs.
- **`regrade_healthbench.py`** — re-grades cached completions with a different judge (e.g.
  `--grader gpt-4.1`, the official protocol) without re-running inference. Resumable; grades
  append per rubric. Used to produce the `healthbench_gpt41` dashboard row.

## Per-model environments

One shared container (the `EVAL_ENVIRONMENT` toml via `slurm/shared/sbatch_overrides.sh`)
runs every job; there are no per-model venvs and the GPU stack is never touched. Model
differences are handled in four layers:

1. **Configuration, not environment** — `MODEL_BACKEND`, `EXTRA_MODEL_ARGS`,
   `FOREIGN_MODEL`, `VLMEVAL_MODEL_PATH_OVERRIDES` select code paths inside the same env.
2. **Library overlays** — when a model needs a newer library than the container ships
   (e.g. Gemma 4 → transformers ≥ 5.5 vs the container's 4.57), stage it once with a
   `pip install --target=cache/pylibs/<name>` job, then submit with
   `EXTRA_PYTHONPATH=cache/pylibs/<name>` (honored by both slurm templates; container
   images clobber plain `PYTHONPATH`). Strip packages the container already provides
   ABI-matched copies of (numpy) from the overlay.
3. **Fork-code selection** — `LMMS_EVAL_DEV_PATH` points jobs at `third_party/lmms-eval`
   instead of the container's baked copy (needed for tasks newer than the image, e.g. the
   RS/geo suite).
4. **Never edit the fork working trees while jobs are queued** — jobs import the shared
   checkout at start time; land commits between fleets.

## Benchmark data staging

- **`stage_geobench_xbd.py`** — stages the xBD imagery slice GEOBench-VLM needs.
- Other remote-sensing trees (VRSBench, BigEarthNet-S2 patches, FRIEDA) live under
  `cache/rs_datasets/` and are staged manually by the admin; jobs find them via
  `RS_DATASETS_ROOT` (+ per-dataset `*_DIR` overrides) exported in `slurm/lmms-eval/eval_job.slurm`.
