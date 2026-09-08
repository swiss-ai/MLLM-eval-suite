# Eval Suite Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every dashboard number traceable, every silent failure loud, every doomed job refused early, and the 70B path memory-deterministic, with all changes landing in the canonical GitHub repositories.

**Architecture:** A dependency-free Python package `suite/` (TOML task registry, run manifests, preflight, dataset staging, tokenize-only cache fill, status) that the existing bash launchers and Slurm job scripts call at fixed points: preflight before submit and before model load, manifest start before inference, manifest finalize after the harness with its status as the job exit code. The dashboard reads manifests and reports coverage. Harness forks gain a thinking canary and a cache-safe presence filter.

**Tech Stack:** Python 3.12 stdlib (`tomllib`, `json`, `hashlib`, `subprocess`), pytest 9 (host and container), bash launchers, Slurm, vLLM harness forks.

**Spec:** `docs/superpowers/specs/2026-09-08-eval-suite-hardening-design.md`

## Global Constraints

- Python 3.12; no new third-party dependencies in `suite/` (stdlib only). Tests run with `python3 -m pytest tests -q` from the repository root on the host, and must not need a GPU or the network.
- Registry file is `suite/tasks.toml`; task keys are the dashboard task names (e.g. `gqa`, `mmsi_bench`, `mathvista_mini`).
- Manifest file name is `run_meta.json`, schema version 1, written at the task output root: lmms-eval `results/lmms-eval/<model>/<run_id>/<task>/run_meta.json`, VLMEvalKit `results/VLMEvalKit/<run_id>/<model>/<dataset>/run_meta.json`.
- Exit codes: preflight failure 2, manifest finalize `failed` 3, `invalid` 4, harness crash without results 3.
- Thinking-invalid threshold: mean `output_tokens` below 8 when thinking was requested.
- 70B lmms-eval profile: `gpu_memory_utilization=0.75`, image-token cache `readonly` strict unless `--allow-encode`.
- Commit author `Yixuan Xu <cavendishyixuan@gmail.com>`; no Co-Authored-By lines. Comments in the file's existing voice; no trailing inline comments.
- No job submissions during implementation; integration tasks run only after the owner lifts the pause.

---

## File structure

Created:
- `suite/__init__.py` — package marker, `REPO_ROOT`.
- `suite/tasks.toml` — task registry (C1).
- `suite/tasks.py` — `load_registry()`, `Task`, `benchmarks_dict()`, `judge_tasks(framework)`, `suite_list_text(framework)`, `check_suite_lists()`, CLI.
- `suite/hashing.py` — `sha256_file(path)`, `file_stat(path)`.
- `suite/manifest.py` — `start(...)`, `finalize(...)`, `output_token_stats(samples_paths)`, CLI `start|finalize`.
- `suite/preflight.py` — `run_checks(...) -> list[Check]`, CLI.
- `suite/stage_datasets.py` — `restore(task, root, downloader)`, CLI.
- `suite/tokenize_cache.py` — tokenize-only pass, CLI (container, GPU).
- `suite/status.py` — `scan(results_root) -> list[dict]`, CLI.
- `suite/coverage.py` — `coverage(table, models, registry) -> dict`, used by the dashboard.
- `slurm/lmms-eval/tokenize_job.slurm` — Slurm template for the tokenize-only pass.
- `tests/conftest.py`, `tests/test_tasks.py`, `tests/test_manifest.py`, `tests/test_preflight.py`, `tests/test_stage_datasets.py`, `tests/test_status.py`, `tests/test_coverage.py`.

Modified:
- `launchers/lmms-eval/eval.sh` — preflight call; 70b readonly-strict default; `--allow-encode`; `--mode tokenize`.
- `launchers/VLMEvalKit/eval.sh` — preflight call.
- `slurm/lmms-eval/eval_job.slurm`, `slurm/VLMEvalKit/eval_job.slurm` — preflight, manifest start, finalize, exit code.
- `scripts/make_dashboard.py` — `BENCHMARKS` from registry; manifest provenance; coverage; JSON export.
- `scripts/refresh_dashboard.sh` — coverage and JSON outputs; registry check.
- `README.md` — suite commands.
- Harness forks: `third_party/lmms-eval/lmms_eval/models/chat/apertus_1p5_vllm.py` (canary), `third_party/lmms-eval/lmms_eval/tasks/geobench/utils.py` (already patched), `third_party/VLMEvalKit/vlmeval/vlm/apertus_1p5.py` (canary).

---

### Task 1: Foundation carry-over

**Files:**
- Copy from `/iopsstor/scratch/cscs/xyixuan/apertus/MLLM-eval-suite` into the clone: `dockerfiles/*` (new and modified), `HANDOVER_VLLM_IMAGE_2026-09-08.md`, `shared/_vendor/{absl,absl_py-2.5.0.dist-info,immutabledict,immutabledict-4.3.1.dist-info,langdetect,langdetect-1.0.9.dist-info,rouge_score,rouge_score-0.1.2.dist-info}`, `shared/_vendor/README.md`, `slurm/VLMEvalKit/eval_job.slurm`, `slurm/shared/job_env.sh`, `scripts/dashboard_models.txt`, `docs/index.html`, `launchers/lmms-eval/eval.sh`, `shared/apertus_image_tokenizer/tokenizer.py`.
- Mirror: `results/`, `logs/`, `quarantine/` from scratch into the clone (rsync, gitignored).

**Interfaces:**
- Produces: a clean branch `yxu/hardening-2026-09-08` whose working tree equals the scratch working tree for tracked areas, in five commits.

- [ ] **Step 1: Copy and commit the image-build work**

```bash
SRC=/iopsstor/scratch/cscs/xyixuan/apertus/MLLM-eval-suite; D=/capstor/store/cscs/swissai/infra01/users/xyixuan/MLLM-eval-suite; cd $D
rsync -a $SRC/dockerfiles/ dockerfiles/ --exclude __pycache__
cp $SRC/HANDOVER_VLLM_IMAGE_2026-09-08.md docs/
git add dockerfiles docs/HANDOVER_VLLM_IMAGE_2026-09-08.md
git commit -m "image build: trial vLLM 0.28 cu130 recipe, hash-verified wheel, validation tooling, handover"
```

- [ ] **Step 2: Copy and commit vendored deps, job-script changes, registry, dashboard, fixes**

```bash
rsync -a $SRC/shared/_vendor/ shared/_vendor/ --exclude __pycache__ --exclude '*.pyc'
git add shared/_vendor && git commit -m "vendor: langdetect, immutabledict, rouge_score, absl for lm-eval ifeval and truthfulqa_gen"
cp $SRC/slurm/VLMEvalKit/eval_job.slurm slurm/VLMEvalKit/eval_job.slurm; cp $SRC/slurm/shared/job_env.sh slurm/shared/job_env.sh
git add slurm && git commit -m "jobs: VLMEvalKit --reuse resume flag; fan the OpenAI key out to task-specific judge variables"
cp $SRC/scripts/dashboard_models.txt scripts/; cp $SRC/docs/index.html docs/
git add scripts/dashboard_models.txt docs/index.html && git commit -m "dashboard: register corrected-RoPE 70B columns; Aug 21 build"
cp $SRC/launchers/lmms-eval/eval.sh launchers/lmms-eval/eval.sh; cp $SRC/shared/apertus_image_tokenizer/tokenizer.py shared/apertus_image_tokenizer/tokenizer.py
bash -n launchers/lmms-eval/eval.sh && python3 -m py_compile shared/apertus_image_tokenizer/tokenizer.py
git add launchers shared/apertus_image_tokenizer && git commit -m "70B: gpu_memory_utilization 0.75; release VQ encoder cache after each render"
```

- [ ] **Step 3: Mirror generated data and verify the tree**

```bash
rsync -a $SRC/results/ results/ ; rsync -a $SRC/logs/ logs/ ; rsync -a $SRC/quarantine/ quarantine/
diff -rq $SRC $D -x .git -x results -x logs -x cache -x quarantine -x third_party -x __pycache__ -x '*.pyc' -x .claude -x docs | grep -v "Only in $D" ; git status --short | head
```
Expected: no differing tracked files; `git status` shows only the submodule pointers as modified (handled in Task 8).

---

### Task 2: Task registry

**Files:**
- Create: `suite/__init__.py`, `suite/tasks.toml`, `suite/tasks.py`
- Test: `tests/conftest.py`, `tests/test_tasks.py`

**Interfaces:**
- Produces: `load_registry(path=None) -> Registry`; `Registry.tasks: dict[str, Task]`; `Task` fields `name, framework, harness_task, category, headline, judge, judge_env, assets: list[Asset], max_model_len, multi_image, card, report`; `Asset` fields `env, relative, source, extract, min_files`; `benchmarks_dict(registry) -> dict[str, dict]` (same shape as `make_dashboard.BENCHMARKS`: `{"cat":..., "vk":..., "vk_prefix":..., "headline": (...)}`); `judge_tasks(registry, framework) -> list[str]` (harness task ids); `suite_list_text(registry, framework) -> str`; `check_suite_lists(registry, repo_root) -> list[str]` (mismatch messages).

- [ ] **Step 1: Write the failing tests**

```python
# tests/conftest.py
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# tests/test_tasks.py
import textwrap
from pathlib import Path
import pytest
from suite.tasks import load_registry, benchmarks_dict, judge_tasks, suite_list_text, check_suite_lists

MINI = textwrap.dedent('''
    version = 1
    [defaults]
    max_model_len = 131072

    [tasks.gqa]
    framework = "lmms-eval"
    harness_task = "gqa"
    category = "General VQA & Perception"
    headline = "exact_match"
    card = true
    report = true

    [tasks.mmsi_bench]
    framework = "VLMEvalKit"
    harness_task = "MMSIBench_wo_circular"
    category = "Spatial & Embodied"
    headline = "acc"
    card = true

    [tasks.mathvista_mini]
    framework = "VLMEvalKit"
    harness_task = "MathVista_MINI"
    category = "Math & Logic"
    headline = "acc"
    judge = "openai"

    [tasks.frieda]
    framework = "lmms-eval"
    harness_task = "frieda"
    category = "Remote Sensing"
    headline = "f1"
    card = true
    [[tasks.frieda.assets]]
    env = "FRIEDA_DIR"
    relative = "images"
    source = "hf://datasets/knowledge-computing/FRIEDA/images.tar"
    extract = "tar"
    min_files = 1000

    [tasks.viewspatial]
    framework = "VLMEvalKit"
    harness_task = "ViewSpatialBench"
    category = "Spatial & Embodied"
    headline = "acc"
    max_model_len = 262144
    multi_image = true
''')

@pytest.fixture
def reg(tmp_path):
    p = tmp_path / "tasks.toml"; p.write_text(MINI); return load_registry(p)

def test_loads_fields_and_defaults(reg):
    t = reg.tasks["gqa"]
    assert t.framework == "lmms-eval" and t.max_model_len == 131072 and t.card and t.report
    assert reg.tasks["viewspatial"].max_model_len == 262144 and reg.tasks["viewspatial"].multi_image
    a = reg.tasks["frieda"].assets[0]
    assert (a.env, a.relative, a.extract, a.min_files) == ("FRIEDA_DIR", "images", "tar", 1000)
    assert reg.tasks["mathvista_mini"].judge == "openai" and reg.tasks["mathvista_mini"].judge_env == "OPENAI_API_KEY"

def test_benchmarks_dict_shape(reg):
    b = benchmarks_dict(reg)
    assert b["gqa"] == {"cat": "General VQA & Perception"}
    assert b["mmsi_bench"] == {"cat": "Spatial & Embodied", "vk": "MMSIBench_wo_circular", "vk_prefix": True}

def test_judge_tasks_and_suite_list(reg):
    assert judge_tasks(reg, "VLMEvalKit") == ["MathVista_MINI"]
    assert judge_tasks(reg, "lmms-eval") == []
    assert suite_list_text(reg, "VLMEvalKit").strip().splitlines()[-1] == "MathVista_MINI"

def test_check_suite_lists_reports_drift(reg, tmp_path):
    d = tmp_path / "task_suites" / "VLMEvalKit"; d.mkdir(parents=True)
    (d / "llm_judge.txt").write_text("# judged\nMMVet\n")
    msgs = check_suite_lists(reg, tmp_path)
    assert any("llm_judge.txt" in m and "MathVista_MINI" in m for m in msgs)

def test_unknown_field_rejected(tmp_path):
    p = tmp_path / "t.toml"; p.write_text('version = 1\n[tasks.x]\nframework = "lmms-eval"\nharness_task = "x"\ncategory = "c"\nheadline = "acc"\nbogus = 1\n')
    with pytest.raises(ValueError):
        load_registry(p)

def test_real_registry_loads_and_is_consistent():
    reg = load_registry()
    assert len([t for t in reg.tasks.values() if t.card]) == 33
    assert "gqa" in reg.tasks and reg.tasks["mmsi_bench"].framework == "VLMEvalKit"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd $D && python3 -m pytest tests/test_tasks.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'suite'`

- [ ] **Step 3: Write the registry and loader**

`suite/__init__.py`:
```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
```

`suite/tasks.py`:
```python
"""Task registry: one declarative table for every benchmark the suite runs."""
from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from suite import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "suite" / "tasks.toml"
FRAMEWORKS = ("lmms-eval", "VLMEvalKit", "lm-eval")
JUDGE_ENV = {"openai": "OPENAI_API_KEY"}
SUITE_LIST_FILES = {"lmms-eval": "task_suites/lmms-eval/visual_llm_judge.txt",
                    "VLMEvalKit": "task_suites/VLMEvalKit/llm_judge.txt"}
TASK_FIELDS = {"framework", "harness_task", "category", "headline", "judge", "judge_env",
               "assets", "max_model_len", "multi_image", "card", "report"}
ASSET_FIELDS = {"env", "relative", "source", "extract", "min_files"}


@dataclass(frozen=True)
class Asset:
    env: str
    relative: str
    source: str
    extract: str = "none"
    min_files: int = 1


@dataclass(frozen=True)
class Task:
    name: str
    framework: str
    harness_task: str
    category: str
    headline: str
    judge: str | None = None
    judge_env: str | None = None
    assets: tuple[Asset, ...] = ()
    max_model_len: int = 131072
    multi_image: bool = False
    card: bool = False
    report: bool = False


@dataclass
class Registry:
    tasks: dict[str, Task]
    defaults: dict = field(default_factory=dict)

    def by_framework(self, framework: str) -> list[Task]:
        return [t for t in self.tasks.values() if t.framework == framework]


def _task(name: str, raw: dict, defaults: dict) -> Task:
    unknown = set(raw) - TASK_FIELDS
    if unknown:
        raise ValueError(f"task {name!r}: unknown fields {sorted(unknown)}")
    for required in ("framework", "harness_task", "category", "headline"):
        if required not in raw:
            raise ValueError(f"task {name!r}: missing {required}")
    if raw["framework"] not in FRAMEWORKS:
        raise ValueError(f"task {name!r}: framework must be one of {FRAMEWORKS}")
    assets = []
    for a in raw.get("assets", []):
        bad = set(a) - ASSET_FIELDS
        if bad:
            raise ValueError(f"task {name!r}: unknown asset fields {sorted(bad)}")
        assets.append(Asset(**a))
    judge = raw.get("judge")
    judge_env = raw.get("judge_env") or (JUDGE_ENV.get(judge) if judge else None)
    return Task(name=name, framework=raw["framework"], harness_task=raw["harness_task"],
                category=raw["category"], headline=raw["headline"], judge=judge, judge_env=judge_env,
                assets=tuple(assets), max_model_len=int(raw.get("max_model_len", defaults.get("max_model_len", 131072))),
                multi_image=bool(raw.get("multi_image", False)), card=bool(raw.get("card", False)),
                report=bool(raw.get("report", False)))


def load_registry(path: Path | None = None) -> Registry:
    path = Path(path) if path else REGISTRY_PATH
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    if data.get("version") != 1:
        raise ValueError(f"{path}: unsupported registry version {data.get('version')!r}")
    defaults = data.get("defaults", {})
    tasks = {name: _task(name, raw, defaults) for name, raw in data.get("tasks", {}).items()}
    return Registry(tasks=tasks, defaults=defaults)


def benchmarks_dict(reg: Registry) -> dict[str, dict]:
    out = {}
    for t in reg.tasks.values():
        entry = {"cat": t.category}
        if t.framework == "VLMEvalKit":
            entry["vk"] = t.harness_task
            entry["vk_prefix"] = True
        out[t.name] = entry
    return out


def judge_tasks(reg: Registry, framework: str) -> list[str]:
    return sorted(t.harness_task for t in reg.by_framework(framework) if t.judge)


def suite_list_text(reg: Registry, framework: str) -> str:
    lines = ["# Generated from suite/tasks.toml by `python3 -m suite.tasks --write-suite-lists`; do not edit."]
    lines += judge_tasks(reg, framework)
    return "\n".join(lines) + "\n"


def _read_list(path: Path) -> list[str]:
    return sorted(l.strip() for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#"))


def check_suite_lists(reg: Registry, repo_root: Path | None = None) -> list[str]:
    root = Path(repo_root) if repo_root else REPO_ROOT
    msgs = []
    for framework, rel in SUITE_LIST_FILES.items():
        path = root / rel
        want = judge_tasks(reg, framework)
        have = _read_list(path) if path.exists() else []
        if want != have:
            msgs.append(f"{rel}: registry says {want}, file has {have}")
    return msgs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Task registry utilities")
    p.add_argument("--check", action="store_true", help="verify generated suite lists match the registry")
    p.add_argument("--write-suite-lists", action="store_true", help="regenerate judge suite lists from the registry")
    p.add_argument("--list", choices=["card", "report", "all"], help="print task names in a set")
    p.add_argument("--framework", choices=FRAMEWORKS)
    a = p.parse_args(argv)
    reg = load_registry()
    if a.write_suite_lists:
        for framework, rel in SUITE_LIST_FILES.items():
            (REPO_ROOT / rel).write_text(suite_list_text(reg, framework))
            print(f"wrote {rel}")
    if a.check:
        msgs = check_suite_lists(reg)
        for m in msgs:
            print("MISMATCH", m)
        return 1 if msgs else 0
    if a.list:
        for t in reg.tasks.values():
            if a.framework and t.framework != a.framework:
                continue
            if a.list == "all" or getattr(t, a.list):
                print(t.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`suite/tasks.toml`: one entry per task in the dashboard's `BENCHMARKS` dict, with `card = true` on exactly the 33 HF-card benchmarks (`gqa mmstar realworldqa vqav2_val vstar_bench countbench mmvp vlms_are_biased vlmsareblind chartqa docvqa_val seedbench_2_plus infovqa_val omnidocbench babyvision mathvision_test mathvista_mini ai2d mmmu_val mmmu_pro_standard scienceqa mmsi_bench viewspatial mindcube embspatial frieda geobench_single vrsbench_vqa pope path_vqa pmc_vqa slake vqa_rad`), `report = true` on the 19 report rows (the 14 card rows in the report plus `mmbench_en_dev ocrbench textvqa_val blink mm_safetybench`), `judge = "openai"` on the judge sets from the existing `visual_llm_judge.txt` and `llm_judge.txt` (`babyvision mathvision_reason_test mathvision_reason_testmini mathverse healthbench` for lmms-eval; `MathVista_MINI MathVerse_MINI CharXiv_descriptive_val CharXiv_reasoning_val MMVet HallusionBench MIA-Bench LogicVista MMSafetyBench MM-IFEval` for VLMEvalKit), `judge_env = "BABYVISION_API_KEY"` on babyvision, assets on `frieda` (FRIEDA_DIR, images, `hf://datasets/knowledge-computing/FRIEDA/images.tar`, tar, 1000), `vrsbench_vqa` (VRSBENCH_DIR, Images_val, `hf://datasets/xiang709/VRSBench/Images_val.zip`, zip, 1000), `geobench_single` (GEOBENCH_DIR, Single/images, `hf://datasets/aialliance/GEOBench-VLM/Single.zip`, zip, 2000), and `max_model_len = 262144` with `multi_image = true` on `viewspatial`, `blink`, `muirbench`, `mmsi_bench`. Categories, `vk` names, and headline metrics are copied verbatim from `scripts/make_dashboard.py` `BENCHMARKS` and `metric_selection.py` so the dashboard output is unchanged. Write the file by transcribing that dict; the test `test_real_registry_loads_and_is_consistent` pins the card count.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_tasks.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add suite tests && git commit -m "suite: task registry (tasks.toml) with loader, dashboard shape, and judge-list generation"
```

---

### Task 3: Run manifest

**Files:**
- Create: `suite/hashing.py`, `suite/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `suite.tasks.load_registry`.
- Produces: `sha256_file(path) -> str`; `file_stat(path) -> dict(size, mtime)`; `start(out_dir, framework, task, run_id, model_path, tokenizer_path, chat_template, model_args, gen_kwargs, thinking, extra) -> dict` writes `run_meta.json`; `finalize(manifest_path, log_paths, results_dir, harness_rc, thinking_min_tokens=8) -> tuple[str, dict]` returns `(status, manifest)` and rewrites the file; `output_token_stats(sample_files) -> dict|None`; CLI `python -m suite.manifest start ...|finalize ...` with exit codes 0/3/4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_manifest.py
import json
from pathlib import Path
import pytest
from suite.manifest import start, finalize, output_token_stats, EXIT_FAILED, EXIT_INVALID

def _model_dir(tmp_path):
    m = tmp_path / "model"; m.mkdir()
    (m / "config.json").write_text('{"architectures": ["ApertusForCausalLM"]}')
    (m / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (m / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 16)
    t = tmp_path / "tok"; t.mkdir()
    (t / "tokenizer.json").write_text("{}"); (t / "chat_template.jinja").write_text("{{ bos_token }}")
    return m, t

def test_start_records_identity(tmp_path):
    m, t = _model_dir(tmp_path); out = tmp_path / "out"
    man = start(out, framework="lmms-eval", task="gqa", run_id="r1", model_path=m, tokenizer_path=t,
                chat_template=t / "chat_template.jinja", model_args="model=x,enable_thinking=True",
                gen_kwargs="max_new_tokens=32768", thinking=True, extra={"slurm": {"job_id": "1"}})
    on_disk = json.loads((out / "run_meta.json").read_text())
    assert on_disk == man and man["schema"] == 1 and man["status"] == "running"
    assert man["model"]["config_sha256"] and man["model"]["weights"]["shards"][0]["size"] == 16
    assert man["tokenizer"]["chat_template_sha256"] and man["thinking"]["requested"] is True
    assert man["slurm"]["job_id"] == "1"

def test_output_token_stats(tmp_path):
    s = tmp_path / "samples_x.jsonl"
    s.write_text("\n".join(json.dumps({"token_counts": [{"output_tokens": n}]}) for n in (2, 4, 600)) + "\n")
    st = output_token_stats([s])
    assert st["n"] == 3 and st["mean"] == pytest.approx(202) and st["max"] == 600

def _run(tmp_path, thinking, tokens, log_text="", results=True, rc=0):
    m, t = _model_dir(tmp_path); out = tmp_path / "out"
    start(out, framework="lmms-eval", task="mmvp", run_id="r1", model_path=m, tokenizer_path=t,
          chat_template=None, model_args="", gen_kwargs="", thinking=thinking, extra={})
    sub = out / "textview__m"; sub.mkdir()
    if results:
        (sub / "20260908_x_results.json").write_text(json.dumps({"results": {"mmvp": {"mmvp_accuracy,none": 0.7}}}))
        (sub / "20260908_x_samples_mmvp.jsonl").write_text("\n".join(json.dumps({"token_counts": [{"output_tokens": n}]}) for n in tokens) + "\n")
    log = tmp_path / "job.out"; log.write_text(log_text)
    return finalize(out / "run_meta.json", [log], out, harness_rc=rc)

def test_finalize_ok(tmp_path):
    status, man = _run(tmp_path, thinking=True, tokens=[300, 500], log_text="apertus_1p5_vllm: enable_thinking=True\napertus_1p5_vllm: thinking canary passed (120 tokens)\n")
    assert status == "ok" and man["thinking"]["effective"] is True and man["thinking"]["canary"] == "passed"
    assert man["results"]["n_samples"] == 2 and man["results"]["file"].endswith("_results.json")

def test_finalize_invalid_when_thinking_did_not_engage(tmp_path):
    status, man = _run(tmp_path, thinking=True, tokens=[2, 3, 4], log_text="apertus_1p5_vllm: enable_thinking=None\n")
    assert status == "invalid" and man["thinking"]["effective"] is None and "output tokens" in man["error"]

def test_finalize_failed_on_harness_error(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[], log_text="Error during evaluation: boom\n", results=False)
    assert status == "failed" and "boom" in man["error"]

def test_finalize_failed_when_no_results(tmp_path):
    status, _ = _run(tmp_path, thinking=False, tokens=[], results=False)
    assert status == "failed"

def test_cli_exit_codes(tmp_path, capsys):
    from suite.manifest import main
    m, t = _model_dir(tmp_path); out = tmp_path / "out"
    assert main(["start", "--out", str(out), "--framework", "lmms-eval", "--task", "gqa", "--run-id", "r", "--model", str(m), "--tokenizer", str(t)]) == 0
    log = tmp_path / "l.out"; log.write_text("Error during evaluation: x\n")
    assert main(["finalize", "--manifest", str(out / "run_meta.json"), "--log", str(log), "--results-dir", str(out), "--harness-rc", "0"]) == EXIT_FAILED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_manifest.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'suite.manifest'`

- [ ] **Step 3: Write hashing and manifest**

`suite/hashing.py`:
```python
import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_stat(path: Path) -> dict:
    st = Path(path).stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}
```

`suite/manifest.py`:
```python
"""Run manifests: what a job actually ran with, and how it ended."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from suite import REPO_ROOT
from suite.hashing import file_stat, sha256_file

SCHEMA = 1
EXIT_FAILED = 3
EXIT_INVALID = 4
THINKING_MIN_MEAN_TOKENS = 8
ERROR_PATTERNS = (
    r"Error during evaluation: (.{0,300})",
    r"(torch\.OutOfMemoryError: .{0,200})",
    r"(EngineDeadError.{0,200})",
    r"(RuntimeError: cancelled)",
    r"(thinking canary failed.{0,200})",
    r"(Engine core initialization failed.{0,100})",
)
FLAG_RE = re.compile(r"apertus_1p5_vllm: enable_thinking=(\S+)")
CANARY_RE = re.compile(r"thinking canary (passed|failed)")


def _git(path: Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = run("rev-parse", "HEAD")
    return {"commit": commit or None, "dirty": bool(run("status", "--porcelain", "--untracked-files=no")) if commit else None,
            "checkout": str(path)}


def _model_identity(model_path: Path) -> dict:
    model_path = Path(model_path)
    real = model_path.resolve()
    ident = {"path": str(model_path), "realpath": str(real), "config_sha256": None, "weights": {"index_sha256": None, "shards": []}}
    cfg = model_path / "config.json"
    if cfg.exists():
        ident["config_sha256"] = sha256_file(cfg)
    idx = model_path / "model.safetensors.index.json"
    if idx.exists():
        ident["weights"]["index_sha256"] = sha256_file(idx)
    for shard in sorted(model_path.glob("*.safetensors")):
        target = shard.resolve()
        entry = {"name": shard.name, "realpath": str(target)}
        entry.update(file_stat(target) if target.exists() else {"size": None, "mtime": None, "missing": True})
        ident["weights"]["shards"].append(entry)
    return ident


def _tokenizer_identity(tokenizer_path: Path | None, chat_template: Path | None) -> dict:
    out = {"path": str(tokenizer_path) if tokenizer_path else None, "tokenizer_json_sha256": None,
           "chat_template": str(chat_template) if chat_template else None, "chat_template_sha256": None}
    if tokenizer_path and (Path(tokenizer_path) / "tokenizer.json").exists():
        out["tokenizer_json_sha256"] = sha256_file(Path(tokenizer_path) / "tokenizer.json")
    template = Path(chat_template) if chat_template else (Path(tokenizer_path) / "chat_template.jinja" if tokenizer_path else None)
    if template and template.exists():
        out["chat_template"] = str(template)
        out["chat_template_sha256"] = sha256_file(template)
    return out


def start(out_dir: Path, *, framework: str, task: str, run_id: str, model_path: Path, tokenizer_path: Path | None,
          chat_template: Path | None, model_args: str, gen_kwargs: str, thinking: bool, extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    harness_dir = REPO_ROOT / "third_party" / ("lmms-eval" if framework == "lmms-eval" else "VLMEvalKit" if framework == "VLMEvalKit" else "lm-eval-harness")
    manifest = {
        "schema": SCHEMA, "run_id": run_id, "framework": framework, "task": task,
        "status": "running", "error": None, "started_at": int(time.time()), "finished_at": None,
        "model": {"name": Path(model_path).name, **_model_identity(Path(model_path))},
        "tokenizer": _tokenizer_identity(tokenizer_path, chat_template),
        "thinking": {"requested": bool(thinking), "effective": None, "canary": "not-run"},
        "generation": {"model_args": model_args, "gen_kwargs": gen_kwargs},
        "harness": {"name": framework, **_git(harness_dir)},
        "suite": _git(REPO_ROOT),
        "container": {"image": os.environ.get("SUITE_CONTAINER_IMAGE"), "size": None},
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME")},
        "results": None,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("APERTUS_", "VLLM_APERTUS_", "IMAGE_TOKEN_CACHE", "PYTORCH_CUDA_ALLOC_CONF"))},
    }
    if manifest["container"]["image"] and Path(manifest["container"]["image"]).exists():
        manifest["container"].update(file_stat(Path(manifest["container"]["image"])))
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key].update(value)
        else:
            manifest[key] = value
    (out_dir / "run_meta.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def output_token_stats(sample_files) -> dict | None:
    counts = []
    for path in sample_files:
        with open(path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tc = rec.get("token_counts")
                while isinstance(tc, list) and tc:
                    tc = tc[0]
                if isinstance(tc, dict) and isinstance(tc.get("output_tokens"), (int, float)):
                    counts.append(int(tc["output_tokens"]))
    if not counts:
        return None
    return {"n": len(counts), "mean": statistics.fmean(counts), "median": statistics.median(counts), "max": max(counts)}


def _scan_logs(log_paths) -> tuple[str | None, bool | None, str]:
    error, effective, canary = None, None, "not-run"
    for path in log_paths:
        try:
            text = Path(path).read_text(errors="ignore")
        except OSError:
            continue
        for pat in ERROR_PATTERNS:
            m = re.search(pat, text)
            if m and error is None:
                error = m.group(1).strip()
        m = FLAG_RE.search(text)
        if m:
            effective = {"True": True, "False": False}.get(m.group(1))
        m = CANARY_RE.search(text)
        if m:
            canary = m.group(1)
    return error, effective, canary


def _find_results(results_dir: Path, framework: str) -> tuple[Path | None, list[Path]]:
    results_dir = Path(results_dir)
    if framework == "VLMEvalKit":
        files = sorted(results_dir.rglob("*_acc.csv")) + sorted(results_dir.rglob("*_score.csv"))
        return (files[-1] if files else None), []
    files = sorted(results_dir.rglob("*_results.json"), key=lambda p: p.stat().st_mtime)
    samples = sorted(results_dir.rglob("*samples_*.jsonl"))
    return (files[-1] if files else None), samples


def finalize(manifest_path: Path, log_paths, results_dir: Path, harness_rc: int, thinking_min_tokens: int = THINKING_MIN_MEAN_TOKENS) -> tuple[str, dict]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    error, effective, canary = _scan_logs(log_paths)
    manifest["thinking"]["effective"] = effective
    manifest["thinking"]["canary"] = canary
    result_file, samples = _find_results(results_dir, manifest["framework"])
    status = "ok"
    if error or harness_rc != 0 or result_file is None:
        status = "failed"
        error = error or (f"harness exit code {harness_rc}" if harness_rc else "no results file produced")
    else:
        stats = output_token_stats(samples) if samples else None
        manifest["results"] = {"file": str(result_file), "n_samples": stats["n"] if stats else None, "output_tokens": stats}
        if manifest["thinking"]["requested"]:
            if canary == "failed" or effective is False:
                status, error = "invalid", "thinking requested but not in effect"
            elif stats and stats["mean"] < thinking_min_tokens:
                status, error = "invalid", f"thinking requested but mean output tokens {stats['mean']:.1f} < {thinking_min_tokens}"
    manifest["status"], manifest["error"], manifest["finished_at"] = status, error, int(time.time())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return status, manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run manifest start/finalize")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--out", required=True); s.add_argument("--framework", required=True); s.add_argument("--task", required=True)
    s.add_argument("--run-id", required=True); s.add_argument("--model", required=True); s.add_argument("--tokenizer")
    s.add_argument("--chat-template"); s.add_argument("--model-args", default=""); s.add_argument("--gen-kwargs", default="")
    s.add_argument("--thinking", action="store_true"); s.add_argument("--extra-json", default="{}")
    f = sub.add_parser("finalize")
    f.add_argument("--manifest", required=True); f.add_argument("--log", action="append", default=[])
    f.add_argument("--results-dir", required=True); f.add_argument("--harness-rc", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "start":
        start(Path(a.out), framework=a.framework, task=a.task, run_id=a.run_id, model_path=Path(a.model),
              tokenizer_path=Path(a.tokenizer) if a.tokenizer else None, chat_template=Path(a.chat_template) if a.chat_template else None,
              model_args=a.model_args, gen_kwargs=a.gen_kwargs, thinking=a.thinking, extra=json.loads(a.extra_json))
        return 0
    status, manifest = finalize(Path(a.manifest), a.log, Path(a.results_dir), a.harness_rc)
    print(f"run_meta: status={status} error={manifest.get('error')}")
    return {"ok": 0, "failed": EXIT_FAILED, "invalid": EXIT_INVALID}[status]


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_manifest.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add suite/hashing.py suite/manifest.py tests/test_manifest.py && git commit -m "suite: run manifest with identity hashes, log-derived status, and thinking validity"
```

---

### Task 4: Job-script hooks for manifests and exit codes

**Files:**
- Modify: `slurm/lmms-eval/eval_job.slurm` (the final `"${CMD[@]}"` line and the env block above it), `slurm/VLMEvalKit/eval_job.slurm` (the `run.py` invocation block near line 293-340).

**Interfaces:**
- Consumes: `python -m suite.manifest start|finalize` CLI from Task 3.
- Produces: jobs that write `run_meta.json` before inference and exit with the finalize status.

- [ ] **Step 1: Hook the lmms-eval job**

Replace the last line `"${CMD[@]}"` in `slurm/lmms-eval/eval_job.slurm` with:
```bash
export SUITE_CONTAINER_IMAGE="${SUITE_CONTAINER_IMAGE:-}"
SUITE_PY="${SUITE_PY:-/opt/venv/bin/python}"
export PYTHONPATH="${ORCH_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
THINK_FLAG=()
[[ "${EXTRA_MODEL_ARGS}" == *enable_thinking=True* || "${EXTRA_MODEL_ARGS}" == *enable_thinking=true* ]] && THINK_FLAG=(--thinking)
"${SUITE_PY}" -m suite.manifest start --out "${OUTPUT_PATH}" --framework lmms-eval --task "${TASKS}" --run-id "$(basename "$(dirname "${OUTPUT_PATH}")")" \
  --model "${MODEL_PATH}" --tokenizer "${TOKENIZER_PATH}" ${CHAT_TEMPLATE:+--chat-template "${CHAT_TEMPLATE}"} \
  --model-args "${MODEL_ARGS}" --gen-kwargs "${GEN_KWARGS:-}" "${THINK_FLAG[@]}" \
  --extra-json "{\"generation\": {\"tp\": ${TP_SIZE_FOR_MANIFEST:-1}, \"dp\": ${NUM_PROCESSES}, \"batch_size\": ${BATCH_SIZE}, \"gpu_memory_utilization\": ${GPU_MEMORY_UTILIZATION}, \"max_model_len\": ${MAX_MODEL_LEN}}}"
set +e
"${CMD[@]}"
HARNESS_RC=$?
set -e
"${SUITE_PY}" -m suite.manifest finalize --manifest "${OUTPUT_PATH}/run_meta.json" \
  --log "${LOG_DIR}/eval_fill_${SLURM_JOB_ID}.out" --log "${LOG_DIR}/eval_fill_${SLURM_JOB_ID}.err" \
  --results-dir "${OUTPUT_PATH}" --harness-rc "${HARNESS_RC}"
exit $?
```
Add `TP_SIZE_FOR_MANIFEST` derivation right after `MODEL_ARGS` is assembled: `TP_SIZE_FOR_MANIFEST=$(sed -n 's/.*tensor_parallel_size=\([0-9]*\).*/\1/p' <<<"${MODEL_ARGS}" | tail -1)`; and in `launchers/lmms-eval/eval.sh` export `SUITE_CONTAINER_IMAGE="$(sed -n 's/^image *= *"\(.*\)"/\1/p' "${EVAL_ENVIRONMENT}")"` after `sbatch_overrides.sh` is sourced. Check `LOG_DIR` is the variable the script already uses for `--log-dir`.

- [ ] **Step 2: Hook the VLMEvalKit job**

Wrap the `run.py` invocation the same way: before it, `manifest start --framework VLMEvalKit --task "${DATA}" --run-id "${RUN_ID}" --model "${APERTUS_MODEL_PATH:-${MODEL}}" --out "${WORK_DIR}"` with `--thinking` when `APERTUS_ENABLE_THINKING` is truthy; after it, `finalize --manifest "${WORK_DIR}/run_meta.json" --log "<this job's .out>" --log "<.err>" --results-dir "${WORK_DIR}" --harness-rc $rc; exit $?`. Use the script's own variable names for the dataset, run id, work dir, and log paths (read them from the `usage()` block and the `--work-dir`/`--log-dir` parsing).

- [ ] **Step 3: Verify**

Run: `bash -n slurm/lmms-eval/eval_job.slurm slurm/VLMEvalKit/eval_job.slurm && grep -n "suite.manifest" slurm/*/eval_job.slurm`
Expected: syntax ok; two `start` and two `finalize` calls.

- [ ] **Step 4: Commit**

```bash
git add slurm launchers && git commit -m "jobs: write run_meta.json before inference; exit with the manifest status after the harness"
```

---

### Task 5: Preflight

**Files:**
- Create: `suite/preflight.py`
- Modify: `launchers/lmms-eval/eval.sh` (before the submission loop), `launchers/VLMEvalKit/eval.sh` (before the submission loop), `slurm/lmms-eval/eval_job.slurm` and `slurm/VLMEvalKit/eval_job.slurm` (before model load, after env setup).
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `load_registry`, `Task`, `Asset`.
- Produces: `Check(name, ok, detail)`; `run_checks(*, registry, framework, model_path, tasks, thinking, tokenizer_path, vision_tokenizer_dir, container_image, env, max_model_len) -> list[Check]`; CLI exit 0 or 2, prints one line per check.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preflight.py
import os, textwrap
from pathlib import Path
from suite.tasks import load_registry
from suite.preflight import run_checks, EXIT_PREFLIGHT

REG = textwrap.dedent('''
    version = 1
    [tasks.pope]
    framework = "lmms-eval"
    harness_task = "pope"
    category = "Alignment"
    headline = "pope_accuracy"
    [tasks.mathvista_mini]
    framework = "VLMEvalKit"
    harness_task = "MathVista_MINI"
    category = "Math & Logic"
    headline = "acc"
    judge = "openai"
    [tasks.frieda]
    framework = "lmms-eval"
    harness_task = "frieda"
    category = "Remote Sensing"
    headline = "f1"
    [[tasks.frieda.assets]]
    env = "FRIEDA_DIR"
    relative = "images"
    source = "hf://x/y/images.tar"
    extract = "tar"
    min_files = 3
    [tasks.viewspatial]
    framework = "VLMEvalKit"
    harness_task = "ViewSpatialBench"
    category = "Spatial & Embodied"
    headline = "acc"
    max_model_len = 262144
''')

def _setup(tmp_path):
    (tmp_path / "tasks.toml").write_text(REG)
    m = tmp_path / "model"; m.mkdir(); (m / "config.json").write_text("{}")
    (m / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 8)
    t = tmp_path / "tok"; t.mkdir(); (t / "tokenizer.json").write_text("{}"); (t / "chat_template.jinja").write_text("x")
    v = tmp_path / "vq"; v.mkdir(); (v / "config.yaml").write_text("a"); (v / "model.ckpt").write_bytes(b"\0")
    img = tmp_path / "img"; img.mkdir(); (img / "photo.jpg").write_bytes(b"\0")
    fr = tmp_path / "frieda"; (fr / "images").mkdir(parents=True)
    for i in range(3): (fr / "images" / f"{i}.png").write_bytes(b"\0")
    return load_registry(tmp_path / "tasks.toml"), m, t, v, img, fr

def _failed(checks):
    return sorted(c.name for c in checks if not c.ok)

def test_all_good(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    checks = run_checks(registry=reg, framework="lmms-eval", model_path=m, tasks=["pope", "frieda"], thinking=False,
                        tokenizer_path=t, vision_tokenizer_dir=v, container_image=img / "photo.jpg",
                        env={"FRIEDA_DIR": str(fr)}, max_model_len=131072)
    assert _failed(checks) == []

def test_dangling_weight_symlink(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    (m / "model-00001-of-00001.safetensors").unlink(); os.symlink(tmp_path / "gone", m / "model-00001-of-00001.safetensors")
    checks = run_checks(registry=reg, framework="lmms-eval", model_path=m, tasks=["pope"], thinking=False,
                        tokenizer_path=t, vision_tokenizer_dir=v, container_image=img / "photo.jpg", env={}, max_model_len=131072)
    assert "model:weights" in _failed(checks)

def test_missing_asset_and_judge_and_context(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    checks = run_checks(registry=reg, framework="lmms-eval", model_path=m, tasks=["frieda"], thinking=False,
                        tokenizer_path=t, vision_tokenizer_dir=v, container_image=img / "photo.jpg",
                        env={"FRIEDA_DIR": str(tmp_path / "nowhere")}, max_model_len=131072)
    assert "task:frieda:assets" in _failed(checks)
    checks = run_checks(registry=reg, framework="VLMEvalKit", model_path=m, tasks=["mathvista_mini", "viewspatial"], thinking=False,
                        tokenizer_path=t, vision_tokenizer_dir=v, container_image=img / "photo.jpg", env={}, max_model_len=131072)
    assert "task:mathvista_mini:judge" in _failed(checks) and "task:viewspatial:context" in _failed(checks)
    checks = run_checks(registry=reg, framework="VLMEvalKit", model_path=m, tasks=["mathvista_mini"], thinking=False,
                        tokenizer_path=t, vision_tokenizer_dir=v, container_image=img / "photo.jpg", env={"OPENAI_API_KEY": "k"}, max_model_len=131072)
    assert "task:mathvista_mini:judge" not in _failed(checks)

def test_unknown_task_and_wrong_framework(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    checks = run_checks(registry=reg, framework="lmms-eval", model_path=m, tasks=["nope", "mathvista_mini"], thinking=False,
                        tokenizer_path=t, vision_tokenizer_dir=v, container_image=img / "photo.jpg", env={}, max_model_len=131072)
    assert "task:nope:registry" in _failed(checks) and "task:mathvista_mini:framework" in _failed(checks)

def test_cli_exit_code(tmp_path):
    from suite.preflight import main
    reg, m, t, v, img, fr = _setup(tmp_path)
    rc = main(["--registry", str(tmp_path / "tasks.toml"), "--framework", "lmms-eval", "--model", str(m), "--tasks", "pope",
               "--tokenizer", str(t), "--vision-tokenizer", str(v), "--container-image", str(img / "photo.jpg"), "--max-model-len", "131072"])
    assert rc == 0
    rc = main(["--registry", str(tmp_path / "tasks.toml"), "--framework", "lmms-eval", "--model", str(tmp_path / "missing"), "--tasks", "pope",
               "--tokenizer", str(t), "--vision-tokenizer", str(v), "--container-image", str(img / "photo.jpg"), "--max-model-len", "131072"])
    assert rc == EXIT_PREFLIGHT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_preflight.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'suite.preflight'`

- [ ] **Step 3: Write preflight**

```python
"""Refuse work that cannot succeed, before it costs a node."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from suite.tasks import Registry, load_registry

EXIT_PREFLIGHT = 2


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


def _readable_file(path: Path) -> bool:
    try:
        return path.resolve().is_file() and os.access(path.resolve(), os.R_OK)
    except OSError:
        return False


def _count_files(path: Path, limit: int) -> int:
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
        if n >= limit:
            return n
    return n


def check_model(model_path: Path) -> list[Check]:
    model_path = Path(model_path)
    out = [Check("model:dir", model_path.is_dir(), str(model_path))]
    if not model_path.is_dir():
        return out
    out.append(Check("model:config", _readable_file(model_path / "config.json"), "config.json"))
    shards = sorted(model_path.glob("*.safetensors"))
    bad = [s.name for s in shards if not _readable_file(s) or s.resolve().stat().st_size == 0]
    out.append(Check("model:weights", bool(shards) and not bad, f"{len(shards)} shards" + (f", unreadable: {bad}" if bad else "")))
    return out


def check_tokenizer(tokenizer_path: Path | None) -> list[Check]:
    if tokenizer_path is None:
        return [Check("tokenizer", False, "no tokenizer path")]
    tokenizer_path = Path(tokenizer_path)
    return [Check("tokenizer:json", _readable_file(tokenizer_path / "tokenizer.json"), str(tokenizer_path)),
            Check("tokenizer:template", _readable_file(tokenizer_path / "chat_template.jinja"), "chat_template.jinja")]


def check_vision_tokenizer(vq_dir: Path | None) -> list[Check]:
    if vq_dir is None:
        return [Check("vision_tokenizer", False, "no vision tokenizer dir")]
    vq_dir = Path(vq_dir)
    ok = _readable_file(vq_dir / "config.yaml") and _readable_file(vq_dir / "model.ckpt")
    return [Check("vision_tokenizer", ok, str(vq_dir))]


def check_container(image: Path | None) -> list[Check]:
    return [Check("container:image", image is not None and _readable_file(Path(image)), str(image))]


def check_tasks(registry: Registry, framework: str, tasks: list[str], env: dict, max_model_len: int) -> list[Check]:
    out = []
    for name in tasks:
        task = registry.tasks.get(name)
        if task is None:
            out.append(Check(f"task:{name}:registry", False, "not in suite/tasks.toml"))
            continue
        if task.framework != framework:
            out.append(Check(f"task:{name}:framework", False, f"registered for {task.framework}, launched on {framework}"))
        for asset in task.assets:
            base = env.get(asset.env, "")
            path = Path(base) / asset.relative if base else None
            n = _count_files(path, asset.min_files) if path and path.is_dir() else 0
            out.append(Check(f"task:{name}:assets", n >= asset.min_files,
                             f"{asset.env}={base or '<unset>'} {asset.relative}: {n} files, need {asset.min_files}; restore with `python3 -m suite.stage_datasets --task {name}`"))
        if task.judge:
            out.append(Check(f"task:{name}:judge", bool(env.get(task.judge_env or "")), f"{task.judge} judge needs {task.judge_env}"))
        if task.max_model_len > max_model_len:
            out.append(Check(f"task:{name}:context", False, f"needs max_model_len {task.max_model_len}, launch has {max_model_len}"))
    return out


def run_checks(*, registry: Registry, framework: str, model_path: Path, tasks: list[str], thinking: bool,
               tokenizer_path: Path | None, vision_tokenizer_dir: Path | None, container_image: Path | None,
               env: dict, max_model_len: int) -> list[Check]:
    checks = check_model(model_path) + check_tokenizer(tokenizer_path) + check_container(container_image)
    if framework in ("lmms-eval", "VLMEvalKit"):
        checks += check_vision_tokenizer(vision_tokenizer_dir)
    checks += check_tasks(registry, framework, tasks, env, max_model_len)
    return checks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Preflight checks for an evaluation launch")
    p.add_argument("--registry"); p.add_argument("--framework", required=True); p.add_argument("--model", required=True)
    p.add_argument("--tasks", required=True, help="comma-separated dashboard task names")
    p.add_argument("--thinking", action="store_true"); p.add_argument("--tokenizer"); p.add_argument("--vision-tokenizer")
    p.add_argument("--container-image"); p.add_argument("--max-model-len", type=int, default=131072)
    a = p.parse_args(argv)
    registry = load_registry(a.registry) if a.registry else load_registry()
    checks = run_checks(registry=registry, framework=a.framework, model_path=Path(a.model), tasks=[t for t in a.tasks.split(",") if t],
                        thinking=a.thinking, tokenizer_path=Path(a.tokenizer) if a.tokenizer else None,
                        vision_tokenizer_dir=Path(a.vision_tokenizer) if a.vision_tokenizer else None,
                        container_image=Path(a.container_image) if a.container_image else None, env=dict(os.environ), max_model_len=a.max_model_len)
    failed = [c for c in checks if not c.ok]
    for c in checks:
        print(f"{'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}")
    print(f"preflight: {len(checks) - len(failed)} ok, {len(failed)} failed")
    return EXIT_PREFLIGHT if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_preflight.py -q`
Expected: 5 passed

- [ ] **Step 5: Wire the launchers and jobs**

In `launchers/lmms-eval/eval.sh`, after tasks are resolved (after the judge guard block) and before the submission loop, add:
```bash
PREFLIGHT_TASKS="$(echo "$TASKS" | tr ' ' ',')"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 -m suite.preflight --framework lmms-eval --model "$MODEL_PATH" --tasks "$PREFLIGHT_TASKS" \
  ${ENABLE_THINKING:+--thinking} --tokenizer "$TOKENIZER_PATH" --vision-tokenizer "${MODELS_CACHE_ROOT}/BAAI/Emu3.5-VisionTokenizer" \
  --container-image "$SUITE_CONTAINER_IMAGE" --max-model-len "${MAX_MODEL_LEN:-131072}" || { echo "preflight failed; not submitting" >&2; exit 2; }
```
where the task names passed are dashboard names: for lmms-eval they are the harness task ids already (registry keys equal harness ids for lmms-eval tasks). In `launchers/VLMEvalKit/eval.sh` do the same with `--framework VLMEvalKit` and map dataset names to registry keys through `python3 -m suite.tasks --list all --framework VLMEvalKit` by harness id (add a `--harness-to-name` mode to `suite.tasks` that prints `harness_task<TAB>name`). In both job scripts, call preflight again right before the harness command with the same arguments (fast fail inside the allocation) and `exit 2` on failure.

- [ ] **Step 6: Verify and commit**

Run: `bash -n launchers/lmms-eval/eval.sh launchers/VLMEvalKit/eval.sh slurm/lmms-eval/eval_job.slurm slurm/VLMEvalKit/eval_job.slurm && python3 -m pytest tests -q`
```bash
git add suite/preflight.py tests/test_preflight.py launchers slurm suite/tasks.py && git commit -m "suite: preflight checks before submission and before model load"
```

---

### Task 6: Dataset restorer

**Files:**
- Create: `suite/stage_datasets.py`
- Test: `tests/test_stage_datasets.py`
- Delete: `cache/claude_probe/stage_rs.sh` is scratch-only and stays out of the repo.

**Interfaces:**
- Consumes: `Task.assets`, `load_registry`.
- Produces: `restore(task: Task, roots: dict[str, Path], downloader: Callable[[str, Path], Path]) -> list[Path]`; `default_downloader(source, dest_dir) -> Path` (uses `huggingface_hub.hf_hub_download` for `hf://datasets/<repo>/<file>` sources and `shutil.copy` for `file://` sources); CLI `python3 -m suite.stage_datasets --task frieda [--root cache/rs_datasets]` which sets the env var root to `<root>/<task dir>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage_datasets.py
import io, tarfile, zipfile
from pathlib import Path
from suite.tasks import Task, Asset
from suite.stage_datasets import restore, parse_source

def _tar(dest: Path, names):
    with tarfile.open(dest, "w") as tf:
        for n in names:
            data = b"x"; info = tarfile.TarInfo(n); info.size = 1; tf.addfile(info, io.BytesIO(data))

def test_parse_source():
    assert parse_source("hf://datasets/knowledge-computing/FRIEDA/images.tar") == ("hf", "knowledge-computing/FRIEDA", "images.tar")
    assert parse_source("file:///tmp/x.zip") == ("file", "", "/tmp/x.zip")

def test_restore_extracts_tar_into_env_root(tmp_path):
    archive = tmp_path / "images.tar"; _tar(archive, ["images/a.png", "images/b.png"])
    task = Task(name="frieda", framework="lmms-eval", harness_task="frieda", category="c", headline="f1",
                assets=(Asset(env="FRIEDA_DIR", relative="images", source=f"file://{archive}", extract="tar", min_files=2),))
    root = tmp_path / "rs" / "frieda"
    done = restore(task, {"FRIEDA_DIR": root}, downloader=lambda src, dest: Path(src.removeprefix("file://")))
    assert done == [root / "images"] and sorted(p.name for p in (root / "images").iterdir()) == ["a.png", "b.png"]

def test_restore_zip_and_skips_when_present(tmp_path):
    archive = tmp_path / "Images_val.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Images_val/P0001.png", "x")
    task = Task(name="vrsbench_vqa", framework="lmms-eval", harness_task="vrsbench_vqa", category="c", headline="acc",
                assets=(Asset(env="VRSBENCH_DIR", relative="Images_val", source=f"file://{archive}", extract="zip", min_files=1),))
    root = tmp_path / "vrs"
    calls = []
    def dl(src, dest):
        calls.append(src); return Path(src.removeprefix("file://"))
    assert restore(task, {"VRSBENCH_DIR": root}, downloader=dl) == [root / "Images_val"]
    assert restore(task, {"VRSBENCH_DIR": root}, downloader=dl) == []
    assert len(calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_datasets.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the restorer**

```python
"""Restore the local image trees that registry tasks declare, from their sources."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable

from suite import REPO_ROOT
from suite.tasks import Task, load_registry

DEFAULT_ROOT = REPO_ROOT / "cache" / "rs_datasets"
ASSET_DIRS = {"FRIEDA_DIR": "frieda", "VRSBENCH_DIR": "vrsbench", "GEOBENCH_DIR": "geobench", "BIGEARTH_S2_DIR": "bigearth/BigEarthNet-S2"}


def parse_source(source: str) -> tuple[str, str, str]:
    if source.startswith("hf://datasets/"):
        rest = source[len("hf://datasets/"):]
        repo, _, filename = rest.partition("/")
        owner_repo = repo
        if "/" not in owner_repo:
            owner_repo, _, filename2 = filename.partition("/")
            owner_repo, filename = f"{repo}/{owner_repo}", filename2
        return "hf", owner_repo, filename
    if source.startswith("file://"):
        return "file", "", source[len("file://"):]
    raise ValueError(f"unsupported source {source!r}")


def default_downloader(source: str, dest_dir: Path) -> Path:
    kind, repo, filename = parse_source(source)
    if kind == "file":
        return Path(filename)
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=str(dest_dir)))


def _count(path: Path, limit: int) -> int:
    n = 0
    for _r, _d, files in os.walk(path):
        n += len(files)
        if n >= limit:
            break
    return n


def _extract(archive: Path, kind: str, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if kind == "tar":
        with tarfile.open(archive) as tf:
            tf.extractall(into, filter="data")
    elif kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    elif kind == "none":
        shutil.copy2(archive, into / archive.name)
    else:
        raise ValueError(f"unknown extract kind {kind!r}")


def restore(task: Task, roots: dict[str, Path], downloader: Callable[[str, Path], Path] = default_downloader) -> list[Path]:
    restored = []
    for asset in task.assets:
        root = Path(roots[asset.env])
        target = root / asset.relative
        if target.is_dir() and _count(target, asset.min_files) >= asset.min_files:
            continue
        downloads = root / "_downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        archive = downloader(asset.source, downloads)
        _extract(archive, asset.extract, root)
        if _count(target, asset.min_files) < asset.min_files:
            raise RuntimeError(f"{task.name}: {target} has fewer than {asset.min_files} files after extracting {archive}")
        restored.append(target)
    return restored


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Restore declared dataset assets")
    p.add_argument("--task", action="append", required=True)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    reg = load_registry()
    for name in a.task:
        task = reg.tasks[name]
        roots = {asset.env: Path(os.environ.get(asset.env) or Path(a.root) / ASSET_DIRS.get(asset.env, name)) for asset in task.assets}
        for path in restore(task, roots):
            print(f"restored {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, commit**

Run: `python3 -m pytest tests/test_stage_datasets.py -q` → 3 passed
```bash
git add suite/stage_datasets.py tests/test_stage_datasets.py && git commit -m "suite: restore declared dataset assets from their registry sources"
```

---

### Task 7: Harness fork changes (thinking canary, cache-safe filters)

**Files:**
- Modify: `third_party/lmms-eval/lmms_eval/models/chat/apertus_1p5_vllm.py` (constructor already fixed; add canary in `generate_until`), `third_party/lmms-eval/lmms_eval/tasks/geobench/utils.py` (already patched: `load_from_cache_file=False`).
- Modify: `third_party/VLMEvalKit/vlmeval/vlm/apertus_1p5.py` (canary at first generate when thinking is on).

**Interfaces:**
- Produces: log lines `apertus_1p5_vllm: thinking canary passed (<n> tokens)` or a raised `RuntimeError("thinking canary failed: ...")`; VLMEvalKit logs `apertus_1p5: thinking canary passed (<n> tokens)`. Env `APERTUS_SKIP_THINKING_CANARY=1` bypasses.

- [ ] **Step 1: Add the canary to the lmms-eval wrapper**

In `Apertus1p5VLLM`, add a method and call it at the top of `generate_until` when `self.enable_thinking` and not yet run:
```python
    _CANARY_PROMPT = "What is 17 multiplied by 23? Think it through before answering."

    def _run_thinking_canary(self):
        if os.environ.get("APERTUS_SKIP_THINKING_CANARY", "").lower() in ("1", "true", "yes"):
            eval_logger.warning("apertus_1p5_vllm: thinking canary skipped by APERTUS_SKIP_THINKING_CANARY")
            return
        from vllm import SamplingParams

        prompt = self._ap_tokenizer.apply_chat_template(
            [{"role": "user", "content": {"parts": [{"type": "text", "text": self._CANARY_PROMPT}]}}],
            add_generation_prompt=True, tokenize=False, chat_template=self._ap_chat_template, enable_thinking=True,
        )
        token_ids = self._ap_tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        params = self._build_sampling_params_dict({"max_new_tokens": 512, "temperature": 0.6, "top_p": 0.95})
        out = self.client.generate(prompts=[{"prompt_token_ids": token_ids}], sampling_params=[SamplingParams(**params)])
        text, n_tokens = out[0].outputs[0].text, len(out[0].outputs[0].token_ids)
        if _INNER_PREFIX not in text:
            raise RuntimeError(f"thinking canary failed: enable_thinking={self.enable_thinking} but no {_INNER_PREFIX} in {n_tokens} output tokens: {text[:200]!r}")
        eval_logger.info(f"apertus_1p5_vllm: thinking canary passed ({n_tokens} tokens)")
```
Set `self._canary_done = False` in `__init__`; in `generate_until`: `if self.enable_thinking and not self._canary_done: self._run_thinking_canary(); self._canary_done = True`. The canary runs on every DP rank (each rank owns an engine); that is intended.

- [ ] **Step 2: Add the same canary to the VLMEvalKit wrapper**

In `vlmeval/vlm/apertus_1p5.py`, in the method that renders and generates (line ~181 `apply_chat_template`), add `_run_thinking_canary` using the class's own tokenizer, template, and engine handle, logging with the module's logger, raising `RuntimeError` on failure, gated by the same env var, and invoked once before the first generation when `self.enable_thinking`.

- [ ] **Step 3: Verify syntax and commit on fork branches**

```bash
cd third_party/lmms-eval && python3 -m py_compile lmms_eval/models/chat/apertus_1p5_vllm.py lmms_eval/tasks/geobench/utils.py
git checkout -b yxu/thinking-flag-canary-geobench && git add -A lmms_eval/models/chat/apertus_1p5_vllm.py lmms_eval/tasks/geobench/utils.py
git commit -m "apertus_1p5_vllm: keep enable_thinking across the base constructor and prove deliberation with a canary; geobench presence filters bypass the datasets cache"
git push -u origin yxu/thinking-flag-canary-geobench
cd ../VLMEvalKit && python3 -m py_compile vlmeval/vlm/apertus_1p5.py && git add vlmeval/vlm/apertus_1p5.py && git commit -m "apertus_1p5: thinking canary before the first generation" && git push
cd ../.. && git add third_party/lmms-eval third_party/VLMEvalKit && git commit -m "third_party: pin lmms-eval thinking fix and VLMEvalKit torchcodec+canary branches"
```

---

### Task 8: Tokenize-only cache fill and read-only 70B profile

**Files:**
- Create: `suite/tokenize_cache.py`, `slurm/lmms-eval/tokenize_job.slurm`
- Modify: `launchers/lmms-eval/eval.sh` (`--mode tokenize`, `--allow-encode`, 70b readonly-strict default), `slurm/shared/image_token_cache_env.sh` (no change if `readonly` already implies strict; verify).

**Interfaces:**
- Produces: `python -m suite.tokenize_cache --tasks a,b --shard i/n` (runs in the container; iterates docs via lmms-eval's `TaskManager`/`get_task_dict`, calls `doc_to_visual`, and encodes through `shared.apertus_image_tokenizer.ApertusImageTokenizer.encode_images` with the cache env in fill mode; prints `tokenize_cache: task=<t> docs=<n> images=<m> misses=<k>`); Slurm template runs four shards, one per GPU; launcher submits one tokenize job per task with `--mode tokenize`.

- [ ] **Step 1: Write the tokenize pass**

```python
"""Fill the image-token cache for tasks without loading the language model."""
from __future__ import annotations

import argparse
import os
import sys

from suite import REPO_ROOT


def iter_task_images(task_name: str, shard: int, nshards: int):
    from lmms_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager()
    task = get_task_dict([task_name], tm)[task_name]
    docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
    for i, doc in enumerate(docs):
        if i % nshards != shard:
            continue
        yield i, task.doc_to_visual(doc)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tokenize-only image cache fill")
    p.add_argument("--tasks", required=True); p.add_argument("--shard", default="0/1"); p.add_argument("--tokenizer", required=True)
    a = p.parse_args(argv)
    shard, nshards = (int(x) for x in a.shard.split("/"))
    sys.path.insert(0, str(REPO_ROOT / "shared"))
    from apertus_image_tokenizer import _shared_state
    from transformers import AutoTokenizer
    image_tokenizer, mm_kwargs = _shared_state()
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=False)
    for task_name in a.tasks.split(","):
        n_docs = n_images = 0
        for _i, visuals in iter_task_images(task_name, shard, nshards):
            images = [v for v in visuals if hasattr(v, "size")]
            if images:
                image_tokenizer.encode_images(images, tokenizer=tokenizer, mm_processor_kwargs=mm_kwargs)
            n_docs += 1; n_images += len(images)
        print(f"tokenize_cache: task={task_name} shard={a.shard} docs={n_docs} images={n_images}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
Check the exact names `get_task_dict`, `has_test_docs`, `test_docs`, `validation_docs`, and `doc_to_visual` in `third_party/lmms-eval/lmms_eval/tasks/__init__.py` and `lmms_eval/api/task.py` before committing; adjust to the fork's API.

- [ ] **Step 2: Slurm template and launcher mode**

`slurm/lmms-eval/tokenize_job.slurm`: copy the `#SBATCH` header and the environment setup (cache env, HF_HOME, PYTHONPATH with `third_party/lmms-eval` and `shared`) from `eval_job.slurm`, then run four shards: `for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g /opt/venv/bin/python -m suite.tokenize_cache --tasks "$TASKS" --shard $g/4 --tokenizer "$TOKENIZER_PATH" & done; wait`. Cache env must be `IMAGE_TOKEN_CACHE_MODE=fill`.

In `launchers/lmms-eval/eval.sh`: add `--mode tokenize` (submits `tokenize_job.slurm` per task instead of `eval_job.slurm`), `--allow-encode` flag, and in the `70b` profile set `MODE=readonly` unless `--allow-encode` was given, printing `70b profile: image-token cache readonly (strict); run --mode tokenize first or pass --allow-encode`.

- [ ] **Step 3: Verify and commit**

Run: `bash -n launchers/lmms-eval/eval.sh slurm/lmms-eval/tokenize_job.slurm && python3 -m py_compile suite/tokenize_cache.py`
```bash
git add suite/tokenize_cache.py slurm/lmms-eval/tokenize_job.slurm launchers/lmms-eval/eval.sh && git commit -m "suite: tokenize-only cache fill; 70B profile runs the image cache read-only"
```

---

### Task 9: Dashboard from manifests, coverage report, JSON export

**Files:**
- Create: `suite/coverage.py`
- Modify: `scripts/make_dashboard.py` (`BENCHMARKS` import, manifest provenance in cells, coverage and JSON outputs), `scripts/refresh_dashboard.sh`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: dashboard `table` rows (`{"task","metric","framework","cat","cells": {model: {"v","raw","run"}}}`), `models`, `Registry`.
- Produces: `coverage(table, models, registry, manifests: dict[tuple[str,str], dict]) -> dict` with `{"missing": [{"task","model","reason"}], "invalid": [...], "counts": {...}}` where reason is one of `no-run`, `run-failed:<error>`, `run-invalid:<error>`, `no-parsable-result`; `find_manifest(path) -> dict|None` walks up from a results file to the nearest `run_meta.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage.py
import json
from pathlib import Path
from suite.coverage import coverage, find_manifest
from suite.tasks import Registry, Task

def _reg():
    return Registry(tasks={"gqa": Task("gqa", "lmms-eval", "gqa", "General VQA & Perception", "exact_match", card=True),
                           "pope": Task("pope", "lmms-eval", "pope", "Alignment", "pope_accuracy", card=True)})

def test_coverage_reasons():
    table = [{"task": "gqa", "metric": "exact_match", "framework": "lmms-eval", "cat": "x", "cells": {"m1": {"v": 59.2, "run": "r"}}}]
    manifests = {("pope", "m1"): {"status": "failed", "error": "OOM"}, ("gqa", "m2"): {"status": "invalid", "error": "no thinking"}}
    cov = coverage(table, ["m1", "m2"], _reg(), manifests)
    reasons = {(m["task"], m["model"]): m["reason"] for m in cov["missing"]}
    assert reasons[("pope", "m1")] == "run-failed:OOM" and reasons[("gqa", "m2")] == "run-invalid:no thinking" and reasons[("pope", "m2")] == "no-run"
    assert cov["counts"] == {"cells": 4, "present": 1, "missing": 3}

def test_find_manifest_walks_up(tmp_path):
    run = tmp_path / "results" / "lmms-eval" / "m" / "r1" / "gqa"; leaf = run / "textview__m"; leaf.mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"status": "ok", "run_id": "r1"}))
    f = leaf / "x_results.json"; f.write_text("{}")
    assert find_manifest(f)["run_id"] == "r1" and find_manifest(tmp_path / "nowhere.json") is None
```

- [ ] **Step 2: Run test to verify it fails, then write `suite/coverage.py`**

```python
"""Which dashboard cells are missing, and why."""
from __future__ import annotations

import json
from pathlib import Path

from suite.tasks import Registry


def find_manifest(path: Path) -> dict | None:
    for parent in Path(path).parents:
        candidate = parent / "run_meta.json"
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except json.JSONDecodeError:
                return None
    return None


def coverage(table: list[dict], models: list[str], registry: Registry, manifests: dict[tuple[str, str], dict]) -> dict:
    present = {(row["task"], m) for row in table for m in row["cells"]}
    tasks = [t for t in registry.tasks]
    missing, cells = [], 0
    for task in tasks:
        for model in models:
            cells += 1
            if (task, model) in present:
                continue
            man = manifests.get((task, model))
            if man is None:
                reason = "no-run"
            elif man.get("status") == "failed":
                reason = f"run-failed:{man.get('error')}"
            elif man.get("status") == "invalid":
                reason = f"run-invalid:{man.get('error')}"
            else:
                reason = "no-parsable-result"
            missing.append({"task": task, "model": model, "reason": reason})
    return {"missing": missing, "counts": {"cells": cells, "present": cells - len(missing), "missing": len(missing)}}
```

- [ ] **Step 3: Integrate into `make_dashboard.py`**

At the top: `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`, `from suite.tasks import load_registry, benchmarks_dict`, `from suite.coverage import coverage, find_manifest`; replace the literal `BENCHMARKS = {...}` with `REGISTRY = load_registry(); BENCHMARKS = benchmarks_dict(REGISTRY)` and keep `VK_HEADLINE_BY_TASK` built from `REGISTRY.tasks[t].headline` for VLMEvalKit tasks (as a one-tuple) so headline selection is unchanged. Where cells are built from a results file (in `collect`, `collect_vlmeval`, `collect_lm_eval`), attach `cell["manifest"] = {"run_id": man["run_id"], "status": man["status"], "thinking": man["thinking"], "model_sha": man["model"]["config_sha256"], "harness": man["harness"]["commit"]}` when `find_manifest(path)` returns one, and record `manifests[(task, canonical_key)] = man` for every manifest seen, including failed and invalid runs whose results are absent (walk `runs_root/**/run_meta.json`). After the table is assembled and before rendering: `cov = coverage(table, models, REGISTRY, manifests)`; write `docs/coverage.json` and `docs/dashboard.json` next to the output; print `coverage: <present>/<cells> cells; <n> missing (run-failed: a, run-invalid: b, no-run: c)`. Registry validation: every column key in `dashboard_models.txt` must match at least one model key seen in results or manifests; otherwise print `registry: column <key> has no runs` (warning, not failure).

- [ ] **Step 4: Verify with the mirrored results and commit**

Run: `python3 -m pytest tests -q && python3 scripts/make_dashboard.py --runs-root results/lmms-eval --vlmeval-root cache/vlmeval_bridge --lm-eval-root results/lm-eval --models-file scripts/dashboard_models.txt -o /tmp/claude-30214/-iopsstor-scratch-cscs-xyixuan-apertus-apertus-1-5-report/de764314-2ef9-44b9-865f-d8f5c56a3349/scratchpad/dash_check.html` (build the bridge first with the `link_model` loop from `scripts/refresh_dashboard.sh` pointing at the clone's `results/VLMEvalKit`).
Expected: the page builds, `coverage:` line printed, `dashboard.json` and `coverage.json` written.
```bash
git add suite/coverage.py tests/test_coverage.py scripts/make_dashboard.py scripts/refresh_dashboard.sh && git commit -m "dashboard: benchmarks from the registry, manifest provenance per cell, coverage report, JSON export"
```

---

### Task 10: Status command

**Files:**
- Create: `suite/status.py`
- Test: `tests/test_status.py`

**Interfaces:**
- Produces: `scan(results_root: Path) -> list[dict]` (one dict per manifest: run_id, framework, task, model, status, error, started_at, finished_at, job_id); CLI `python3 -m suite.status [--root results] [--run-id X] [--slurm]` printing a table, `--slurm` annotating `running` manifests with `sacct` state.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_status.py
import json
from suite.status import scan, format_table

def test_scan_and_format(tmp_path):
    a = tmp_path / "lmms-eval" / "m" / "r1" / "gqa"; a.mkdir(parents=True)
    (a / "run_meta.json").write_text(json.dumps({"run_id": "r1", "framework": "lmms-eval", "task": "gqa", "model": {"name": "m"}, "status": "ok", "error": None, "started_at": 1, "finished_at": 2, "slurm": {"job_id": "9"}}))
    b = tmp_path / "VLMEvalKit" / "r1" / "m" / "BLINK"; b.mkdir(parents=True)
    (b / "run_meta.json").write_text(json.dumps({"run_id": "r1", "framework": "VLMEvalKit", "task": "BLINK", "model": {"name": "m"}, "status": "failed", "error": "OOM", "started_at": 1, "finished_at": 3, "slurm": {"job_id": "10"}}))
    rows = scan(tmp_path)
    assert [(r["task"], r["status"]) for r in rows] == [("BLINK", "failed"), ("gqa", "ok")]
    text = format_table(rows)
    assert "OOM" in text and "gqa" in text
```

- [ ] **Step 2: Write `suite/status.py`**

```python
"""Run states from manifests, optionally annotated with Slurm accounting."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from suite import REPO_ROOT


def scan(results_root: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(results_root).rglob("run_meta.json")):
        try:
            m = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rows.append({"run_id": m.get("run_id"), "framework": m.get("framework"), "task": m.get("task"),
                     "model": (m.get("model") or {}).get("name"), "status": m.get("status"), "error": m.get("error"),
                     "started_at": m.get("started_at"), "finished_at": m.get("finished_at"),
                     "job_id": (m.get("slurm") or {}).get("job_id"), "path": str(path)})
    rows.sort(key=lambda r: (r["run_id"] or "", r["framework"] or "", r["task"] or ""))
    return rows


def annotate_slurm(rows: list[dict]) -> None:
    ids = [r["job_id"] for r in rows if r["status"] == "running" and r["job_id"]]
    if not ids:
        return
    out = subprocess.run(["sacct", "-j", ",".join(ids), "-X", "-n", "-o", "JobID,State"], capture_output=True, text=True).stdout
    state = {l.split()[0]: l.split()[1] for l in out.splitlines() if l.strip()}
    for r in rows:
        if r["status"] == "running" and r["job_id"] in state:
            r["status"] = f"running/{state[r['job_id']]}"


def format_table(rows: list[dict]) -> str:
    head = f"{'run_id':28} {'fw':10} {'task':24} {'model':30} {'status':16} error"
    lines = [head]
    for r in rows:
        lines.append(f"{(r['run_id'] or '')[:28]:28} {(r['framework'] or '')[:10]:10} {(r['task'] or '')[:24]:24} {(r['model'] or '')[:30]:30} {(r['status'] or '')[:16]:16} {(r['error'] or '')[:80]}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluation run status from manifests")
    p.add_argument("--root", default=str(REPO_ROOT / "results")); p.add_argument("--run-id"); p.add_argument("--slurm", action="store_true")
    a = p.parse_args(argv)
    rows = [r for r in scan(Path(a.root)) if not a.run_id or r["run_id"] == a.run_id]
    if a.slurm:
        annotate_slurm(rows)
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Test and commit**

Run: `python3 -m pytest tests/test_status.py -q` → 1 passed
```bash
git add suite/status.py tests/test_status.py && git commit -m "suite: status command over run manifests"
```

---

### Task 11: Documentation and branch push

**Files:**
- Modify: `README.md` (new section "Suite contracts": registry, preflight, manifests, tokenize mode, status, coverage), `docs/HANDOVER_VLLM_IMAGE_2026-09-08.md` (append a dated section pointing to the spec and the new state), `scripts/refresh_dashboard.sh` (run `python3 -m suite.tasks --check` first and fail on drift).

- [ ] **Step 1: Write the README section** describing each command with its exact invocation and exit codes from the Global Constraints, and the rule that a run without `run_meta.json` shows as legacy.

- [ ] **Step 2: Full test run and push**

Run: `python3 -m pytest tests -q && bash -n launchers/*.sh launchers/*/eval.sh slurm/*/eval_job.slurm slurm/lmms-eval/tokenize_job.slurm && python3 -m suite.tasks --check`
```bash
git add README.md docs scripts/refresh_dashboard.sh && git commit -m "docs: suite contracts and commands; refresh checks the registry"
git push -u origin yxu/hardening-2026-09-08
```

---

### Task 12: Integration validation (only after the owner lifts the submission pause)

**Files:** none new; uses the launchers from the capstor checkout.

- [ ] **Step 1: Launch path and exit codes.** From the capstor checkout: `bash launchers/eval.sh --eval-framework lmms-eval --model cache/models/textview/8B-Final-correct-rope --tasks pope,mmvp,mmstar --run-id val_8b_launch --size 8b`. Expected: preflight prints all `ok`, three jobs; `python3 -m suite.status --run-id val_8b_launch` shows `ok`; values within 0.3 of POPE 86.1, MMVP 69.0, MMStar 45.2; `sacct` shows COMPLETED with exit 0.
- [ ] **Step 2: Thinking canary and validity.** Same with `--thinking --tasks mmvp --run-id val_8b_think`. Expected: log contains `thinking canary passed`; manifest `thinking.effective=true`, `output_tokens.mean` in the hundreds; MMVP near 76.0.
- [ ] **Step 3: Negative thinking test.** Run once with `APERTUS_SKIP_THINKING_CANARY=1` and `--thinking` against a model dir whose `chat_template.jinja` has the `Deliberation` block removed (a copied temp dir). Expected: manifest status `invalid`, job exit 4.
- [ ] **Step 4: Tokenize-only then read-only 70B.** `bash launchers/eval.sh --eval-framework lmms-eval --mode tokenize --tasks mmvp --run-id val_tok` then `--model cache/models/textview/70B-Final-fast --tasks mmvp --size 70b --run-id val_70b_ro`. Expected: the 70B log shows zero VQ encodes and no OOM; MMVP near 68.3.
- [ ] **Step 5: Broken preflight.** `FRIEDA_DIR=/nonexistent bash launchers/eval.sh ... --tasks frieda`. Expected: `preflight failed; not submitting`, exit 2, no job submitted.
- [ ] **Step 6: Record results** in `docs/superpowers/specs/2026-09-08-eval-suite-hardening-design.md` under a new "Validation record" section with job ids and numbers; commit; open the pull requests with `gh pr create` on the suite branch and both fork branches.

---

### Task 13: Harness sync (separate plan)

Write `docs/superpowers/plans/<date>-harness-sync.md` after Task 12 passes: fetch upstream `EvolvingLMMs-Lab/lmms-eval` and `open-compass/VLMEvalKit`, merge into fork branches, resolve conflicts in the Apertus wrappers and task additions, pin `lm-eval-harness` to its latest tag, run the harness unit tests where they exist, then repeat Task 12 steps 1, 2, and 4. A production number moving more than 0.5 points blocks the sync until explained.
