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
  (lmms results tree), `VLMEVAL_OUTPUTS`, `TRUNC_TOOL`. Column curation lives in its `CURATED`
  array (`key=Label`, one line per dashboard column).
- **`make_dashboard.py`** — the generator. Benchmark→category mapping is the `TAXONOMY` dict;
  display-dropped tasks are `DROPPED_TASK_PREFIXES` (drop policy for *runs* lives in `task_suites/`).
- **`metric_selection.py`** — single source of truth for each benchmark's headline metric and score
  normalization; imported by every reporting script.
- **`gather_results.py`** — newest-result-per-task selection + CLI table.
- **`verify_dashboard.py`** — audits canonical-model-key collisions (the wrong-number-on-page
  failure class). Run it after every refresh; PASS expected.
- **`derive_vlmeval_acc.py`** — reconstructs `derived_acc.csv` for VK judge benchmarks whose score
  only exists in job logs. Skips runs with high judge-failure rates rather than fabricating scores.

## HealthBench grading

- Serving the Qwen judge (Meditron protocol): `sbatch "${SBATCH_OVERRIDES[@]}"
  slurm/shared/healthbench_grader_server.slurm` (source `slurm/shared/sbatch_overrides.sh` first).
  It writes `host:port` to `cache/grader/healthbench_endpoint` once ready; point
  `HEALTHBENCH_GRADER_BASE_URL=http://<host>:8080/v1` at it when submitting healthbench jobs.
- **`regrade_healthbench.py`** — re-grades cached completions with a different judge (e.g.
  `--grader gpt-4.1`, the official protocol) without re-running inference. Resumable; grades
  append per rubric. Used to produce the `healthbench_gpt41` dashboard row.

## Benchmark data staging

- **`stage_geobench_xbd.py`** — stages the xBD imagery slice GEOBench-VLM needs.
- Other remote-sensing trees (VRSBench, BigEarthNet-S2 patches, FRIEDA) live under
  `cache/rs_datasets/` and are staged manually by the admin; jobs find them via
  `RS_DATASETS_ROOT` (+ per-dataset `*_DIR` overrides) exported in `slurm/lmms-eval/eval_job.slurm`.
