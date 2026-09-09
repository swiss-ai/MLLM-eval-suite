#!/usr/bin/env python3
"""Generate a self-contained static HTML benchmark matrix from the runs tree.

One file, no server, no CDN — works over file://, scp, or VS Code preview.
Rows are canonical headline metrics (via metric_selection, including the
extra canonical rows like mmvp pair accuracy); columns are models; cell value
is the newest result per task across that model's runs.

Usage:
  python3 make_dashboard.py --runs-root /capstor/.../apertus-1p5-eval/runs \
      [--models substr ...] [-o dashboard.html]
"""

import argparse
import csv
import datetime
import glob
import json
import math
import sys
import re
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suite.coverage import Manifests, cell_provenance  # noqa: E402
from suite.coverage import coverage as coverage_report  # noqa: E402
from suite.tasks import benchmarks_dict, load_registry  # noqa: E402
from suite.result_selection import artifact_metadata, ineligible_reason, merge_cells  # noqa: E402

from gather_results import newest_per_task
from metric_selection import iter_headline_metrics, normalize_score

# Spatial-intelligence benchmarks are tracked on EASI via VLMEvalKit
# (https://easi.lmms-lab.com/leaderboard/), not here.
# ---------------------------------------------------------------------------
# The benchmark registry: one record per dashboard task key. Everything the
# dashboard needs to know about a benchmark lives here — its category, the
# VLMEvalKit dataset name(s) when that harness owns it, a headline-metric
# override, and whether the key/category match by prefix. The legacy
# structures below are derived views of this table; edit the table, not them.
# ---------------------------------------------------------------------------
# Benchmark families, VLMEvalKit dataset names, headline metrics, and lm-eval
# metric names live in suite/tasks.toml; this dict is that table verbatim.
REGISTRY = load_registry()
BENCHMARKS = benchmarks_dict(REGISTRY)


# Categories in display order, with modality.
CATEGORIES = [('General VQA & Perception', 'vision'), ('Robustness & Bias', 'vision'), ('Spatial & Embodied', 'vision'), ('Multi-Image', 'vision'), ('Instruction Following', 'vision'), ('Counting & Grounding', 'vision'), ('Docs, Charts & OCR', 'vision'), ('Math & Logic', 'vision'), ('STEM & Knowledge', 'vision'), ('Remote Sensing', 'vision'), ('Alignment', 'vision'), ('Medical VQA', 'vision'), ('Medical', 'text'), ('Math (Text)', 'text'), ('Knowledge & Reasoning (Text)', 'text'), ('Instruction Following (Text)', 'text')]

EXTRA_VK_PREFIXES = ('muirbench', 'mm_ifeval', 'mia_bench')

EASI_SPATIAL_PREFIXES = tuple(k for k, b in BENCHMARKS.items() if b.get("vk_prefix"))

# Spatial-intelligence + multi-image benchmarks are owned by VLMEvalKit/EASI
# (correct interleaving + EASI protocol); everything else by lmms-eval. The
# badge shows which harness produced each benchmark's number, EASI-style.
VLMEVALKIT_PREFIXES = EASI_SPATIAL_PREFIXES + EXTRA_VK_PREFIXES


LM_EVAL_TASKS = tuple(k for k, b in BENCHMARKS.items() if b.get("lm_metric"))


def framework_for(task: str) -> str:
    """The harness whose number is authoritative for a task: the registry's owner, else the legacy prefix rule."""
    registered = REGISTRY.lookup("lmms-eval", task)
    if registered is not None:
        return registered.framework
    if task in LM_EVAL_TASKS:
        return "lm-eval"
    return "VLMEvalKit" if task.lower().startswith(VLMEVALKIT_PREFIXES) else "lmms-eval"


# Benchmarks dropped from every harness: cmmmu (removed), the mmlu_flan
# generative-medical subjects (exact-match scorer is format-fragile),
# ok_vqa / simplevqa (low-signal, not widely reported).
# Valid zero scores, including RefSpatial, remain visible.
DROPPED_TASK_PREFIXES = ("cmmmu", "mmlu_flan", "ok_vqa", "simplevqa", "mathvista_testmini", "logicvista_reasoning",
                         # lmms duplicate of the VLMEvalKit-owned HallusionBench row
                         # (VK runs the judge; the lmms copy scored 0.0 keyless).
                         "hallusion_bench_image",
                         # n-gram captioning metrics (CIDEr/BLEU) measure prompt-style overlap, not
                         # caption quality, on free-form RS output; results stay on disk.
                         # Run/schedule policy lives in task_suites/*; entries here only
                         # suppress artifacts already on disk.
                         "vrsbench_cap", "geobench_cap", "bigearth_cap")

_TRUNC_CACHE: dict | None = None


def _truncation_for(model_dir_name: str) -> dict:
    """Per-task truncation rates for a thinking run, precomputed by
    truncation_report.py --json into cache/truncation/ so we don't re-scan the
    (huge) samples files on every dashboard regen."""
    global _TRUNC_CACHE
    if _TRUNC_CACHE is None:
        _TRUNC_CACHE = {}
        tdir = Path(__file__).resolve().parent.parent / "cache" / "truncation"
        if tdir.is_dir():
            for f in tdir.glob("*.json"):
                try:
                    _TRUNC_CACHE[f.stem] = json.loads(f.read_text()).get("rates", {})
                except (OSError, ValueError):
                    pass
    return _TRUNC_CACHE.get(model_dir_name, {})

_FAMILY = re.compile(r"^(?:apertus[-_]?1[.p]5[-_]?8b|ap1p5[-_]?8b)[-_]?", re.I)
# Eval-mode / sampling suffixes — different runs of the SAME checkpoint, kept
# distinct so a CoT run never merges with its direct-mode sibling.
_MODE_SUFFIXES = [
    ("-cot-t6", "cot-t6"), ("-cot-rp105", "cot-rp105"), ("-cot-rp110", "cot-rp110"),
    ("-cot", "cot"), ("-rp105", "rp105"), ("-rp110", "rp110"),
]
# Thinking mode carries an optional token-budget suffix (-thinking, -thinking-32k).
# Both must tag a distinct thinking mode; otherwise the step-only sft-256k key
# rebuild drops the suffix and the run collapses onto its direct sibling.
_THINKING_RE = re.compile(r"-thinking(-\d+k)?$")
_MODEL_ALIAS = {
    "sft-capfilter-lr6e-5-constant-innovator-fix-it23409": "sft-capfilter-innovator-it23409",
}


def canonical_model_key(name: str) -> str:
    """Map a run/output dir name to a checkpoint identity shared across harnesses.

    lmms-eval and VLMEvalKit name the same checkpoint differently
    (ap1p5-...-128n_4200 vs Apertus-1p5-8B-sft-256k-4200); both collapse to
    sft-256k-4200 so their numbers land in one column. Hyperparameter noise is
    dropped; RL stage + step + image range are kept so distinct checkpoints
    (online vs mixed vs base) never merge.
    """
    s = name.lower()
    # VLMEval runs launched by checkpoint path carry a slugified absolute path
    # (capstor_store_..._<ckpt>); reduce to the final checkpoint name so they
    # canonicalize like the clean run-name launches.
    if s.startswith("capstor_store_"):
        s = re.split(r"(?:hf_checkpoints|hf-checkpoints|rleval|final_8b)_", s)[-1]
    mode = None
    if (t := _THINKING_RE.search(s)):
        s, mode = s[: t.start()], "thinking" + (t.group(1) or "")
    else:
        for suffix, tag in _MODE_SUFFIXES:
            if s.endswith(suffix):
                s, mode = s[: -len(suffix)], tag
                break
    body = _FAMILY.sub("", s)
    step_match = re.search(r"sft-256k.*?[_-](\d{3,4})\b", body)
    if step_match:
        step = step_match.group(1)
        rl = re.search(r"(online|mixed)\b.*?images?-(\d+)-(\d+)", body)
        if rl:
            key = f"sft-256k-{step}-{rl.group(1)}-{rl.group(2)}-{rl.group(3)}"
        elif "dpo" in body:
            stage = re.search(r"(maxmin|mixed)", body)
            key = f"sft-256k-{step}-{stage.group(1) if stage else 'dpo'}"
        else:
            key = f"sft-256k-{step}"
    elif (m := re.search(r"sft-16k.*?it(\d+)", body)):
        key = f"sft-16k-it{m.group(1)}"
    elif (m := re.search(r"sft-8k.*?it(\d+)", body)):
        key = f"sft-8k-it{m.group(1)}"
    elif re.fullmatch(r"sft-rl-dpo", body):
        key = "sft-rl-dpo"
    else:
        key = _MODEL_ALIAS.get(body, body or "base")
    return f"{key} [{mode}]" if mode else key


def parse_models_manifest(path: Path) -> tuple[list, list, dict, dict]:
    """Parse dashboard_models.txt: 'key[|alias...]=Label' lines, # comments.

    A '# group: Name' comment starts a selector section; subsequent keys
    belong to it. Returns (only_keys, label_specs, alias_map, group_map)
    where alias_map sends each alias key to its primary (foreign models whose
    lmms label and VLMEvalKit registry name canonicalize differently, e.g.
    gemma-3-27b-it vs gemma3-27b) and group_map sends each key to its section.
    """
    only, labels, aliases, groups = [], [], {}, {}
    group = "Models"
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if stripped.lower().startswith("# group:"):
            group = stripped.split(":", 1)[1].strip() or group
            continue
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        keypart, label = line.split("=", 1)
        keys = [k.strip() for k in keypart.split("|")]
        only.append(keys[0])
        labels.append(f"{keys[0]}={label.strip()}")
        groups[keys[0]] = group
        for alias in keys[1:]:
            aliases[alias] = keys[0]
    return only, labels, aliases, groups


# VLMEvalKit dataset dir -> canonical task name, restricted to the benchmarks
# VLMEvalKit owns (spatial + multi-image + UI grounding). Everything else in a
# VLMEval_Outputs tree is lmms-eval-owned and ignored here.
VK_OWNED_TASKS = {
    vk: key
    for key, b in BENCHMARKS.items()
    for vk in ([b["vk"]] if isinstance(b.get("vk"), str) else b.get("vk", []))
}
_VK_HEADLINE = ("overall", "overall_accuracy", "acc", "accuracy")
_VK_AGG_LABELS = ("all", "overall", "none")
# Per-benchmark headline override where the EASI-canonical metric is not plain
# accuracy. site_bench reports chance-adjusted accuracy (overall_caa); raw
# accuracy ~2x inflates it relative to the EASI leaderboard.
VK_HEADLINE_BY_TASK = {k: b["headline"] for k, b in BENCHMARKS.items() if "headline" in b and k in VK_OWNED_TASKS.values()}


def _vk_norm(name: str) -> str:
    return name.strip().lower().replace("(%)", "").strip()


def parse_vk_acc(path: Path, headline: tuple[str, ...] = _VK_HEADLINE) -> float | None:
    """Headline score from a VLMEvalKit *_acc.csv, normalized to 0-100.

    ``headline`` is a priority-ordered list of metric names; the first present
    wins, so a benchmark whose canonical metric is chance-adjusted gets it ahead
    of plain accuracy. Handles wide single-row, long/melted (metric,value), and
    multi-row-by-category layouts; metric names are normalized (a trailing
    ``(%)`` is stripped) so e.g. ``accuracy (%)`` still matches.
    """
    lines = path.read_text().splitlines()
    delim = "\t" if "\t" in lines[0] else ","
    rows = [r for r in csv.reader(lines, delimiter=delim) if r]
    if len(rows) < 2:
        return None
    header = [_vk_norm(h) for h in rows[0]]
    data = rows[1:]

    def scale(text: str, label: str) -> float | None:
        value = float(text.strip().rstrip("%"))
        if not math.isfinite(value):
            return None
        # An explicit unit takes precedence over the legacy fraction heuristic.
        percent = "%" in label or text.strip().endswith("%")
        return value * 100 if not percent and 0 <= value <= 1.0 else value

    if len(header) == 2 and header[1] == "value":
        cells = {_vk_norm(r[0]): (r[1], r[0]) for r in data if len(r) >= 2}
        for want in headline:
            if want in cells:
                return scale(*cells[want])
        return None
    for want in headline:
        if want not in header:
            continue
        col = header.index(want)
        if len(data) > 1:
            for row in data:
                if any(_vk_norm(c) in _VK_AGG_LABELS for c in row):
                    return scale(row[col], rows[0][col])
        return scale(data[0][col], rows[0][col])
    return None


def collect_vlmeval(vk_root: Path, model_filters: list[str] | None, manifests: Manifests,
                    *, model_key=canonical_model_key):
    model_dirs = sorted(d for d in vk_root.iterdir() if d.is_dir())
    if model_filters:
        model_dirs = [d for d in model_dirs
                      if any(s in d.name or s in canonical_model_key(d.name) for s in model_filters)]

    rows: dict[tuple[str, str], dict[str, dict]] = {}
    models: set[str] = set()
    skipped: list[str] = []
    for mdir in model_dirs:
        canon = model_key(mdir.name)
        for vk_name, task in VK_OWNED_TASKS.items():
            if task.lower().startswith(DROPPED_TASK_PREFIXES):
                continue
            bench_dirs = [d for d in ([mdir / vk_name] + sorted(mdir.glob(f"{vk_name}__*")))
                          if d.is_dir()]
            if not bench_dirs:
                continue
            shadows = [k for k in VK_OWNED_TASKS if k != vk_name and k.startswith(vk_name)]

            def owned(name: str) -> bool:
                return vk_name in name and not any(s in name for s in shadows)

            def by_mtime(paths):
                # transient judge artifacts in the shared outputs tree can
                # vanish between glob and stat
                stamped = []
                for p in paths:
                    try:
                        stamped.append((Path(p).stat().st_mtime, p))
                    except OSError:
                        continue
                return [p for _, p in sorted(stamped)]

            all_accs, all_scores = [], []
            for bench_dir in bench_dirs:
                all_accs += glob.glob(f"{bench_dir}/**/*acc*.csv", recursive=True)
                all_scores += glob.glob(f"{bench_dir}/**/*_score.csv", recursive=True)
            real = by_mtime(a for a in all_accs if owned(Path(a).name))
            scores = by_mtime(a for a in all_scores if owned(Path(a).name))
            derived = by_mtime(a for a in all_accs if Path(a).name == "derived_acc.csv")

            # Newest real judge acc wins; score.csv next; broken-era derived files
            # only when nothing better parses.
            acc, value = None, None
            for candidate in real[::-1] + scores[::-1] + derived[::-1]:
                candidate_path = Path(candidate)
                manifest = manifests.for_result(candidate_path)
                run_id = (manifest or {}).get("run_id") or candidate_path.parent.name
                if ineligible_reason(manifest):
                    manifests.reject_result(candidate_path, task, canon, run_id)
                    continue
                try:
                    value = parse_vk_acc(Path(candidate), VK_HEADLINE_BY_TASK.get(task, _VK_HEADLINE))
                except (OSError, ValueError, IndexError):
                    value = None
                if value is not None:
                    acc = Path(candidate)
                    break
                manifests.reject_result(candidate_path, task, canon, run_id)
            if value is None:
                skipped.append(f"{mdir.name}/{vk_name}")
                continue
            models.add(canon)
            cell = {"v": round(value, 2), "raw": value, "run": (manifest or {}).get("run_id") or acc.parent.name,
                    **artifact_metadata(acc, manifest)}
            prov = cell_provenance(manifest)
            if prov:
                cell["prov"] = prov
            # mm_safetybench is direction-normalized to safety_rate at derivation
            # (attack_rate is lower-better); label it so readers see which it is.
            metric = "safety_rate" if task == "mm_safetybench" else "acc"
            merge_cells(rows.setdefault((task, metric), {}), {canon: cell})

    if skipped:
        print(f"VLMEval: skipped {len(skipped)} benchmark(s) with no parseable acc.csv "
              f"(e.g. {', '.join(skipped[:4])})")
    table = [
        {"task": task, "metric": metric, "framework": "VLMEvalKit", "cells": cells}
        for (task, metric), cells in sorted(rows.items())
    ]
    return sorted(models), table


def short_labels(names: list[str]) -> dict[str, str]:
    """Strip the longest common prefix so column headers stay readable."""
    if len(names) < 2:
        return {n: n for n in names}
    prefix = names[0]
    for n in names[1:]:
        while not n.startswith(prefix):
            prefix = prefix[:-1]
    cut = prefix.rfind("-") + 1 if "-" in prefix else len(prefix)
    return {n: (n[cut:] or n) for n in names}


def lmms_result_rows(task: str, data: dict, include_spatial: bool = False):
    """Usable canonical rows, shared by candidate selection and verification."""
    if task.lower().startswith(DROPPED_TASK_PREFIXES):
        return []
    if not include_spatial and framework_for(task) == "VLMEvalKit":
        return []
    metrics = data.get("results", {}).get(task, {})
    if not isinstance(metrics, dict):
        return []
    rows = []
    for row_task, metric, value in iter_headline_metrics(task, metrics, *REGISTRY.headline_for("lmms-eval", task)):
        registered = REGISTRY.resolve("lmms-eval", row_task)
        if registered is not None:
            row_task = registered.name
        if task.lower().startswith("mathvista") != ("llm_as_judge" in metric.lower()):
            continue
        norm = normalize_score(metric, value)
        if norm is not None:
            rows.append((row_task, metric, value, norm))
    return rows


def collect(runs_root: Path, model_filters: list[str] | None, manifests: Manifests, include_spatial: bool = False,
            *, model_key=canonical_model_key):
    model_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    if model_filters:
        model_dirs = [d for d in model_dirs if any(s in d.name for s in model_filters)]
    if not model_dirs:
        print(f"no model dirs under {runs_root}; skipping")
        return [], []

    rows: dict[tuple[str, str], dict[str, dict]] = {}
    for mdir in model_dirs:
        canon = model_key(mdir.name)
        trunc = _truncation_for(mdir.name)
        # All tasks in one parsed artifact share its provenance check. Keep it
        # only for this collection; verified publication checks the bytes again.
        result_manifests = {}
        def invalid(path):
            manifest = manifests.for_result(path)
            relative = path.relative_to(mdir).parts
            task = (manifest or {}).get("task") or (relative[1] if len(relative) > 2 else "")
            registered = REGISTRY.resolve("lmms-eval", task)
            row_task = registered.name if registered else task
            run_id = (manifest or {}).get("run_id") or relative[0]
            manifests.reject_result(path, row_task, canon, run_id)

        def eligible(task, path, data):
            if path not in result_manifests:
                result_manifests[path] = manifests.for_result(path)
            manifest = result_manifests[path]
            if ineligible_reason(manifest, data, task) or not lmms_result_rows(task, data, include_spatial):
                registered = REGISTRY.resolve("lmms-eval", task)
                row_task = registered.name if registered else task
                run_id = (manifest or {}).get("run_id") or path.relative_to(mdir).parts[0]
                manifests.reject_result(path, row_task, canon, run_id)
                return False
            return True

        for task, (path, data) in newest_per_task(mdir, eligible, invalid).items():
            manifest = result_manifests[path]
            run_id = (manifest or {}).get("run_id") or path.relative_to(mdir).parts[0]
            prov = cell_provenance(manifest)
            for row_task, metric, value, norm in lmms_result_rows(task, data, include_spatial):
                cell = {"v": round(norm * 100, 2), "raw": value, "run": run_id,
                        **artifact_metadata(path, manifest)}
                if task in trunc:
                    cell["t"] = round(trunc[task] * 100, 1)
                if prov:
                    cell["prov"] = prov
                # Two result dirs can canonicalize to one column (label case,
                # path-slug variants); the newest artifact wins, not dir order.
                merge_cells(rows.setdefault((row_task, metric), {}), {canon: cell})

    models = sorted({model_key(d.name) for d in model_dirs})
    table = [
        {"task": task, "metric": metric.replace(",none", ""), "framework": "lmms-eval", "cells": cells}
        for (task, metric), cells in sorted(rows.items())
    ]
    return models, table


def collect_lm_eval(lm_root: Path, model_filters: list[str] | None, manifests: Manifests,
                    *, model_key=canonical_model_key):
    """results/lm-eval/<model>/<run-id>/<task>/.../results_*.json → cells.

    lm_eval nests its json under a sanitized model dir, so rglob; only tasks
    registered with an lm_metric are ingested (this also drops per-subject
    subtask rows that share the results dict with their aggregate)."""
    if not lm_root.is_dir():
        return [], []
    model_dirs = sorted(d for d in lm_root.iterdir() if d.is_dir())
    if model_filters:
        model_dirs = [d for d in model_dirs if any(s in d.name for s in model_filters)]
    rows: dict[tuple[str, str], dict[str, dict]] = {}
    for mdir in model_dirs:
        canon = model_key(mdir.name)
        for path in mdir.rglob("results_*.json"):
            manifest = manifests.for_result(path)
            relative = path.relative_to(mdir).parts
            run_id = (manifest or {}).get("run_id") or relative[0]
            def invalid():
                task = (manifest or {}).get("task") or (relative[1] if len(relative) > 2 else "")
                registered = REGISTRY.resolve("lm-eval", task)
                manifests.reject_result(path, registered.name if registered else task, canon, run_id)
            try:
                data = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                invalid()
                continue
            if not isinstance(data, dict) or not isinstance(data.get("results"), dict) or not data["results"]:
                invalid()
                continue
            for task, metrics in data.get("results", {}).items():
                if ineligible_reason(manifest, data, task):
                    manifests.reject_result(path, task, canon, run_id)
                    continue
                rec = BENCHMARKS.get(task, {})
                metric = rec.get("lm_metric")
                if not metric:
                    continue
                value = metrics.get(metric) if isinstance(metrics, dict) else None
                if not isinstance(value, (int, float)):
                    manifests.reject_result(path, task, canon, run_id)
                    continue
                score = normalize_score(metric, float(value))
                if score is None:
                    manifests.reject_result(path, task, canon, run_id)
                    continue
                cell = {"v": round(score * 100, 2), "raw": score, "run": run_id,
                        **artifact_metadata(path, manifest)}
                prov = cell_provenance(manifest)
                if prov:
                    cell["prov"] = prov
                merge_cells(rows.setdefault((task, metric), {}), {canon: cell})
    models = sorted({model_key(d.name) for d in model_dirs})
    table = [
        {"task": task, "metric": metric.replace(",none", ""), "framework": "lm-eval", "cells": cells}
        for (task, metric), cells in sorted(rows.items())
    ]
    return models, table


def collect_inventory(roots, *, vlmeval_roots=(), lm_eval_roots=(), manifest_roots=(),
                      model_filters=None, include_spatial=False):
    """Collect once, retaining source directories until collision verification.

    Each collection is (root, raw model names, rows). Rendering canonicalizes
    these same cells only after the optional audit has examined every directory.
    """
    lanes = [(collect, roots), (collect_vlmeval, vlmeval_roots), (collect_lm_eval, lm_eval_roots)]
    all_roots = [root for _collector, lane_roots in lanes for root in lane_roots]
    manifests = Manifests(all_roots + list(manifest_roots))
    collections = []
    for collector, lane_roots in lanes:
        for root in sorted({Path(root).resolve() for root in lane_roots}):
            kwargs = {"include_spatial": include_spatial} if collector is collect else {}
            models, rows = collector(root, model_filters, manifests, model_key=lambda name: name, **kwargs)
            collections.append((root, models, rows))
    # Rejection evidence is consumed by legacy import under canonical identities,
    # even though the collected cells still use raw names for collision checking.
    manifests.rejected_results = {(task, canonical_model_key(model), run, source)
                                  for task, model, run, source in manifests.rejected_results}
    return manifests, collections


def audit_inventory(collections, *, aliases=None, only_keys=None) -> int:
    """Check unmerged collected cells for conflicting source directories."""
    keep = set(only_keys) if only_keys else None
    by_key = defaultdict(lambda: defaultdict(dict))
    for root, _models, rows in collections:
        for row in rows:
            for model, cell in row["cells"].items():
                key = canonical_model_key(model)
                key = (aliases or {}).get(key, key)
                if keep is not None and key not in keep:
                    continue
                by_key[key][str((root / model).resolve())][(row["task"], row["metric"])] = cell["v"]

    bad = 0
    for key, vals in sorted(by_key.items()):
        if len(vals) < 2:
            continue
        conflicts = []
        for t in sorted(set().union(*(v.keys() for v in vals.values()))):
            present = {n: v[t] for n, v in vals.items() if t in v}
            if len({round(x, 3) for x in present.values()}) > 1:
                conflicts.append((t, present))
        if conflicts:
            bad += 1
            print(f"COLLISION  key '{key}'  <-  {len(vals)} dirs:")
            for directory in vals:
                print(f"             {directory}")
            print(f"           {len(conflicts)} conflicting benchmark(s), e.g.:")
            for (task, metric), present in conflicts[:3]:
                shown = ", ".join(f"{n[-28:]}={x:.1f}" for n, x in present.items())
                print(f"             {task}: {shown}")
    print(f"\n{'PASS — no contaminating collisions' if not bad else f'FAIL — {bad} colliding key(s)'}")
    return bad


def _source_signature(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


# Benchmark taxonomy for the matrix band grouping. Bands follow the category
# conventions of recent VLM reports (Qwen3-VL, InternVL3.5): pure-text evals get
# their own top-level band instead of mixing into multimodal domains. One
# declarative structure — dict order is display order, modality defaults to
# "vision", tasks match by exact name or prefix.
TAXONOMY = {
    cat: {
        **({"modality": modality} if modality != "vision" else {}),
        "exact": [k for k, b in BENCHMARKS.items() if b.get("cat") == cat and not b.get("cat_prefix")],
        "prefix": [k for k, b in BENCHMARKS.items() if b.get("cat") == cat and b.get("cat_prefix")],
    }
    for cat, modality in CATEGORIES
}

CATEGORY_ORDER = list(TAXONOMY)
CATEGORY_MODALITY = {cat: spec.get("modality", "vision") for cat, spec in TAXONOMY.items()}
_TASK_TO_CAT = {t: cat for cat, spec in TAXONOMY.items() for t in spec.get("exact", ())}
_CAT_PREFIX = [(p, cat) for cat, spec in TAXONOMY.items() for p in spec.get("prefix", ())]


def category_for(task: str) -> str | None:
    lowered = task.lower()
    if lowered in _TASK_TO_CAT:
        return _TASK_TO_CAT[lowered]
    for prefix, cat in _CAT_PREFIX:
        if lowered.startswith(prefix):
            return cat
    return None


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apertus · Eval Matrix</title>
<style>
:root {
  --paper: #f7f3ec;
  --paper-2: #efe9de;
  --ink: #1d1915;
  --muted: #8a7f6e;
  --hair: #d9d1c2;
  --red: #c8321e;
  --green: #2c6e49;
}
* { box-sizing: border-box; }
html { background: var(--paper); }
body {
  margin: 0;
  color: var(--ink);
  font: 14px/1.45 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  background:
    radial-gradient(1200px 400px at 80% -10%, rgba(200,50,30,.045), transparent 60%),
    var(--paper);
  padding: 0 clamp(16px, 3vw, 48px) 80px;
}
.mono { font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
.wrap { max-width: 1560px; margin: 0 auto; }

header { padding: 40px 0 0; }
.kicker {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 11px; letter-spacing: .28em; text-transform: uppercase;
  color: var(--red);
}
.kicker::after { content: ""; display: block; width: 64px; height: 2px; background: var(--red); margin-top: 10px; }
h1 { font-size: clamp(26px, 3.5vw, 40px); font-weight: 500; letter-spacing: -.015em; margin: 16px 0 6px; }
.meta { color: var(--muted); font-size: 13px; }
.meta b { color: var(--ink); font-weight: 600; }

/* model picker */
.picker { margin-top: 26px; border: 1px solid var(--hair); background: var(--paper); }
.picker-head {
  display: flex; align-items: baseline; gap: 14px; padding: 10px 14px;
  border-bottom: 1px solid var(--hair); background: var(--paper-2);
}
.picker-head .lbl { font-family: ui-monospace, Menlo, monospace; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.picker-head button, .chip-group-head button {
  font: 11px ui-monospace, Menlo, monospace; color: var(--ink); background: none;
  border: 1px solid var(--hair); padding: 3px 10px; cursor: pointer;
}
.picker-head button:hover, .chip-group-head button:hover { border-color: var(--red); color: var(--red); }
.chip-group-head button { padding: 1px 8px; font-size: 10px; }
.chips { padding: 6px 14px 12px; max-height: 230px; overflow: auto; }
.chip-group { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding-top: 6px; }
.chip-group-head { flex-basis: 100%; display: flex; align-items: center; gap: 8px; margin-top: 4px; }
.chip-group-head .lbl { font-size: 11px; }
.chip {
  display: inline-flex; align-items: center; gap: 7px;
  font: 11.5px ui-monospace, Menlo, monospace;
  border: 1px solid var(--hair); padding: 4px 10px; cursor: pointer;
  background: var(--paper); color: var(--muted); user-select: none;
}
.chip .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--hair); flex: none; }
.chip.on { color: var(--ink); border-color: currentColor; }
.chip.on .dot { background: var(--c); }
.chip .cov { color: var(--muted); font-size: 10px; }

/* summary cards */
.cards { display: flex; flex-wrap: wrap; gap: 1px; background: var(--hair); border: 1px solid var(--hair); margin-top: 18px; }
.card { flex: 1 1 190px; background: var(--paper); padding: 13px 16px 14px; border-top: 3px solid var(--c, var(--hair)); }
.card .name { font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.card .big { font-size: 30px; font-weight: 500; margin-top: 4px; letter-spacing: -.02em; }
.card .big small { font-size: 13px; color: var(--muted); font-weight: 400; }
.card .cov { font-size: 11.5px; color: var(--muted); margin-top: 1px; }
.card.best .big { color: var(--red); }
.cards-note { font-size: 12px; color: var(--muted); margin: 8px 2px 0; }

/* controls */
.controls { display: flex; gap: 22px; align-items: baseline; flex-wrap: wrap; margin: 26px 0 14px; }
.controls label { font-size: 11px; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; font-family: ui-monospace, Menlo, monospace; }
input[type=search], select {
  font: 13px ui-monospace, Menlo, monospace; color: var(--ink);
  background: transparent; border: 0; border-bottom: 1px solid var(--hair);
  padding: 5px 2px; outline: none; min-width: 200px;
}
input[type=search]:focus, select:focus { border-bottom-color: var(--red); }
.viewtab { display: inline-flex; border: 1px solid var(--hair); }
.viewtab button {
  font: 11.5px ui-monospace, Menlo, monospace; letter-spacing: .08em; text-transform: uppercase;
  background: none; border: 0; padding: 6px 16px; cursor: pointer; color: var(--muted);
}
.viewtab button.on { background: var(--ink); color: var(--paper); }
.nodata { color: var(--muted); font-size: 12px; padding: 18px 14px; }

/* harness badge — which eval framework produced this benchmark (EASI-style) */
.hbadge { display: inline-block; font-family: ui-monospace, Menlo, monospace; font-size: 9.5px; letter-spacing: .03em; font-weight: 600; padding: 1px 5px; border-radius: 2px; vertical-align: middle; margin-left: 7px; }
.hbadge.lmms { color: #2c5f7a; background: #dceaf1; }
.hbadge.vlme { color: #7a3d1e; background: #f1e2d6; }
.hbadge.lme  { color: #3e6b2e; background: #e2eed9; }

/* matrix */
.matrix-wrap { overflow: auto; max-height: 76vh; border: 1px solid var(--hair); background: var(--paper); }
table { border-collapse: separate; border-spacing: 0; min-width: 100%; width: max-content; }
thead th {
  position: sticky; top: 0; z-index: 5; background: var(--paper-2);
  font-family: ui-monospace, Menlo, monospace; font-size: 11px; font-weight: 600;
  text-align: right; padding: 9px 13px; border-bottom: 1px solid var(--ink);
  cursor: pointer; white-space: nowrap; user-select: none;
}
thead th.task-h { text-align: left; left: 0; z-index: 6; }
thead th .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--c); margin-right: 6px; vertical-align: baseline; }
thead th .dir { color: var(--red); }
tbody td, tbody th { padding: 6px 13px; border-bottom: 1px solid var(--hair); font-size: 13px; }
tbody tr.catrow th, tbody tr.catrow td {
  background: var(--paper-2); border-bottom: 1px solid var(--ink); padding: 12px 13px 5px;
  position: sticky; top: var(--thead-h, 34px); z-index: 3;
}
tbody tr.catrow th {
  left: 0; z-index: 4; text-align: left;
  font: 600 10.5px ui-monospace, Menlo, monospace; letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted);
}
tbody tr.catrow td.cmean {
  text-align: right; white-space: nowrap;
  font-size: 11.5px; color: var(--muted);
}
tbody tr.catrow td.cmean.best { color: var(--red); font-weight: 600; }
tbody th.task { position: sticky; left: 0; background: var(--paper); text-align: left; font-weight: 400; z-index: 2; max-width: 330px; }
tbody tr:hover td, tbody tr:hover th.task { background: var(--paper-2); }
.task .t { font-family: ui-monospace, Menlo, monospace; font-size: 12.5px; }
.task .m { font-size: 11px; color: var(--muted); }
td.cell { text-align: right; white-space: nowrap; min-width: 92px; position: relative; }
thead th:not(.task-h) { min-width: 92px; }
sup.tr { color: #c87f0a; font-size: 0.62em; font-weight: 600; cursor: help; position: absolute; top: 3px; right: 3px; }
.cell.best { color: var(--red); font-weight: 600; }
.delta-slot { display: inline-block; min-width: 6ch; text-align: left; margin-left: 7px; }
.delta { font-size: 11px; font-weight: 400; }
.delta.up { color: var(--green); } .delta.down { color: var(--red); }
.missing { color: var(--hair); text-align: center; }

footer { margin-top: 26px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--hair); padding-top: 12px; }
footer code { font-family: ui-monospace, Menlo, monospace; font-size: 11px; }
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="kicker">Apertus · Multimodal Evaluation</div>
  <h1>Benchmark Report</h1>
  <div class="meta">__META__</div>
</header>

<div class="picker">
  <div class="picker-head">
    <span class="lbl">Models — <span id="selcount"></span></span>
    <button id="selall">all</button><button id="selnone">none</button>
    <span class="viewtab" style="margin-left:14px"><button id="avg-micro" class="on">Micro</button><button id="avg-macro">Macro</button></span>
    <span class="lbl" style="margin-left:auto">click to toggle</span>
  </div>
  <div class="chips" id="chips"></div>
</div>

<div class="cards" id="cards"></div>
<div class="cards-note" id="cards-note"></div>

<div class="controls">
  <span class="viewtab"><button id="tab-vision" class="on">Vision</button><button id="tab-audio">Audio</button><button id="tab-text">Text</button></span>
  <span><label>filter&nbsp;</label><input id="q" type="search" placeholder="benchmark substring…" spellcheck="false"></span>
  <span><label>category&nbsp;</label><select id="cfilter"><option value="">all</option></select></span>
  <span><label>harness&nbsp;</label><select id="hfilter"><option value="">all</option><option value="lmms-eval">lmms-eval</option><option value="VLMEvalKit">VLMEvalKit</option><option value="lm-eval">lm-eval</option></select></span>
  <span><label>baseline&nbsp;</label><select id="base"><option value="">none</option></select></span>
  <span class="hint" style="margin-left:auto"><span class="hbadge lmms">lmms-eval</span> <span class="hbadge vlme">VLMEvalKit</span> <span class="hbadge lme">lm-eval</span></span>
</div>

<div class="matrix-wrap" id="matrix-view"><table id="matrix"></table></div>

<footer>
  Scores share one absolute 0–100 scale · newest result per task across each model's runs ·
  canonical headline metrics via <code>metric_selection.py</code> · hover for raw value and source run ·
  generated by <a href="https://github.com/swiss-ai/MLLM-eval-suite/blob/yxu/bump-lmms-eval/scripts/make_dashboard.py" target="_blank" rel="noopener"><code>make_dashboard.py</code></a>
</footer>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const PALETTE = ["#c8321e","#1f5f8b","#2c6e49","#8a5a00","#6b4a8c","#b04a7a","#2a7f7f","#774f38","#55611f","#9c3b1e","#3d4f7c","#857000"];
const color = {}; D.models.forEach((m, i) => color[m] = PALETTE[i % PALETTE.length]);
const coverage = {}; D.models.forEach(m => coverage[m] = D.table.filter(r => r.cells[m]).length);

let state = {
  sel: new Set(D.defaultSelected),
  mod: "vision", avg: "micro", q: "", cat: "", sort: null, dir: -1, base: "", harness: "",
};

const fmt = v => v.toFixed(1);
const avg = vs => vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null;
const hbadge = fw => fw === "VLMEvalKit"
  ? '<span class="hbadge vlme">VLMEvalKit</span>'
  : fw === "lm-eval"
  ? '<span class="hbadge lme">lm-eval</span>'
  : '<span class="hbadge lmms">lmms-eval</span>';
const selModels = () => D.models.filter(m => state.sel.has(m));
const visRows = () => {
  const q = state.q.toLowerCase();
  return D.table.filter(r =>
    (D.modality[r.cat] || "vision") === state.mod &&
    (!q || r.task.toLowerCase().includes(q) || r.metric.toLowerCase().includes(q)) &&
    (!state.cat || r.cat === state.cat) &&
    (!state.harness || r.framework === state.harness) &&
    selModels().some(m => r.cells[m]));
};

function renderChips() {
  const groupNames = [];
  for (const m of D.models) {
    const g = D.groups[m] || "Models";
    if (!groupNames.includes(g)) groupNames.push(g);
  }
  const chipHtml = m =>
    `<span class="chip ${state.sel.has(m) ? "on" : ""}" data-m="${m}" style="--c:${color[m]}" title="${m}">` +
    `<span class="dot"></span>${D.labels[m]} <span class="cov">${coverage[m]}</span></span>`;
  document.getElementById("chips").innerHTML = groupNames.map(g => {
    const members = D.models.filter(m => (D.groups[m] || "Models") === g);
    return `<div class="chip-group"><div class="chip-group-head"><span class="lbl">${g}</span>` +
      `<button class="grp-all" data-g="${g}">all</button><button class="grp-none" data-g="${g}">none</button></div>` +
      members.map(chipHtml).join("") + `</div>`;
  }).join("");
  document.querySelectorAll(".chip").forEach(c => c.onclick = () => {
    const m = c.dataset.m;
    state.sel.has(m) ? state.sel.delete(m) : state.sel.add(m);
    renderAll();
  });
  const grpMembers = g => D.models.filter(m => (D.groups[m] || "Models") === g);
  document.querySelectorAll(".grp-all").forEach(b => b.onclick = () => {
    grpMembers(b.dataset.g).forEach(m => state.sel.add(m)); renderAll();
  });
  document.querySelectorAll(".grp-none").forEach(b => b.onclick = () => {
    grpMembers(b.dataset.g).forEach(m => state.sel.delete(m)); renderAll();
  });
  document.getElementById("selcount").textContent = `${state.sel.size}/${D.models.length} selected`;
}

function renderCards() {
  const sel = selModels();
  const inMod = D.table.filter(r => (D.modality[r.cat] || "vision") === state.mod);
  const common = inMod.filter(r => sel.length && sel.every(m => r.cells[m]));
  const nCats = new Set(common.map(r => r.cat)).size;
  const means = sel.map(m => {
    if (state.avg === "macro") {
      const byCat = {};
      for (const r of common) (byCat[r.cat] ??= []).push(r.cells[m].v);
      return { m, mean: avg(Object.values(byCat).map(avg)) };
    }
    return { m, mean: avg(common.map(r => r.cells[m].v)) };
  });
  const top = Math.max(...means.map(x => x.mean ?? -Infinity));
  document.getElementById("cards").innerHTML = means.map(x =>
    `<div class="card ${x.mean === top && means.length > 1 ? "best" : ""}" style="--c:${color[x.m]}" title="${x.m}">` +
    `<div class="name">${D.labels[x.m]}</div>` +
    `<div class="big mono">${x.mean == null ? "—" : fmt(x.mean)}<small> / 100</small></div>` +
    `<div class="cov">${coverage[x.m]} tasks covered</div></div>`).join("");
  document.getElementById("cards-note").textContent = sel.length
    ? (state.avg === "macro"
        ? `macro mean: equal weight per category, over ${nCats} ${state.mod} categories (${common.length} benchmarks covered by all ${sel.length} selected models)`
        : `micro mean: equal weight per benchmark, over the ${common.length} ${state.mod} benchmarks covered by all ${sel.length} selected models`)
    : "select models above";
}

function renderMatrix() {
  const sel = selModels();
  let rows = visRows();
  if (state.sort && state.sel.has(state.sort)) {
    rows = rows.slice().sort((a, b) => {
      const av = a.cells[state.sort]?.v, bv = b.cells[state.sort]?.v;
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return state.dir * (av - bv);
    });
  }
  let h = "<thead><tr><th class='task-h' data-k=''>benchmark / metric</th>";
  for (const m of sel) {
    const dir = state.sort === m ? (state.dir < 0 ? " <span class='dir'>↓</span>" : " <span class='dir'>↑</span>") : "";
    h += `<th data-k="${m}" title="${m}" style="--c:${color[m]}"><span class="dot"></span>${D.labels[m]}${dir}</th>`;
  }
  h += "</tr></thead><tbody>";
  const baseOn = state.base && state.sel.has(state.base);
  const deltaFor = (v, b) => {
    if (!baseOn || b == null || v == null) return "";
    const d = v - b;
    return `<span class="delta ${d >= 0 ? "up" : "down"}">${d >= 0 ? "+" : ""}${d.toFixed(1)}</span>`;
  };
  const slotFor = (m, v, bv) => baseOn ? `<span class="delta-slot">${state.base === m ? "" : deltaFor(v, bv)}</span>` : "";
  let lastCat = null;
  for (const r of rows) {
    if (!state.sort && r.cat !== lastCat) {
      lastCat = r.cat;
      const catRows = rows.filter(x => x.cat === r.cat);
      const covered = sel.filter(m => catRows.some(x => x.cells[m]));
      const commonCat = catRows.filter(x => covered.every(m => x.cells[m]));
      const cm = {};
      for (const m of sel) cm[m] = covered.includes(m) ? avg(commonCat.map(x => x.cells[m].v)) : null;
      const bestM = Math.max(...sel.map(m => cm[m] ?? -Infinity));
      h += `<tr class="catrow"><th>${r.cat}</th>` + sel.map(m => {
        const v = cm[m];
        if (v == null) return "<td class='cmean'>·</td>";
        const slot = slotFor(m, v, cm[state.base]);
        return `<td class="cmean mono ${v === bestM && sel.length > 1 ? "best" : ""}" ` +
               `title="mean over the ${commonCat.length} ${r.cat} benchmarks common to the ${covered.length} models with coverage">${fmt(v)}${slot}</td>`;
      }).join("") + "</tr>";
    }
    const present = sel.filter(m => r.cells[m]);
    const best = Math.max(...present.map(m => r.cells[m].v));
    h += `<tr><th class="task"><div class="t">${r.task}${hbadge(r.framework)}</div><div class="m">${r.metric}</div></th>`;
    for (const m of sel) {
      const c = r.cells[m];
      if (!c) { h += "<td class='missing'>·</td>"; continue; }
      const slot = slotFor(m, c.v, r.cells[state.base]?.v);
      const trtip = c.t != null ? `\\ntruncated: ${c.t}% hit the 32k cap` : "";
      const tr = c.t != null ? `<sup class="tr" title="${c.t}% of outputs hit the 32k token cap (non-terminating)">⌁${Math.round(c.t)}</sup>` : "";
      const evidence = c.legacy ? "legacy: no run manifest" : `manifest: ${c.prov?.status || "unknown"}; harness: ${c.prov?.harness || "unknown"}; image: ${c.prov?.image || "unknown"}`;
      const legacy = c.legacy ? '<sup title="Historical result without a run manifest">L</sup>' : "";
      h += `<td class="cell mono ${c.v === best && present.length > 1 ? "best" : ""}" ` +
           `title="${m}\\n${r.metric} = ${c.raw}\\nrun: ${c.run}\\n${evidence}${trtip}">${fmt(c.v)}${slot}${tr}${legacy}</td>`;
    }
    h += "</tr>";
  }
  document.getElementById("matrix").innerHTML = rows.length
    ? h + "</tbody>"
    : `<tbody><tr><td class="nodata">${state.sel.size
        ? `no ${state.mod} benchmarks yet`
        : "no models selected — pick Apertus checkpoints and baselines above"}</td></tr></tbody>`;
  const thd = document.querySelector("#matrix thead");
  if (thd) document.getElementById("matrix-view").style.setProperty("--thead-h", thd.offsetHeight + "px");
  document.querySelectorAll("thead th").forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    if (!k) { state.sort = null; renderMatrix(); return; }
    if (state.sort === k) state.dir *= -1; else { state.sort = k; state.dir = -1; }
    renderMatrix();
  });
}

function renderBaseSelect() {
  const baseSel = document.getElementById("base");
  const cur = state.base;
  baseSel.innerHTML = "<option value=''>none</option>" + selModels().map(m =>
    `<option value="${m}" ${m === cur ? "selected" : ""}>${D.labels[m]}</option>`).join("");
}

function renderAll() {
  renderChips(); renderCards(); renderBaseSelect();
  renderMatrix();
}

function renderTabs() {
  for (const t of ["vision", "audio", "text"])
    document.getElementById("tab-" + t).classList.toggle("on", state.mod === t);
  const cats = D.categories.filter(c => (D.modality[c] || "vision") === state.mod);
  document.getElementById("cfilter").innerHTML =
    "<option value=''>all</option>" + cats.map(c => `<option ${state.cat === c ? "selected" : ""}>${c}</option>`).join("");
}
for (const t of ["vision", "audio", "text"])
  document.getElementById("tab-" + t).onclick = () => { state.mod = t; state.cat = ""; renderTabs(); renderAll(); };
renderTabs();
for (const a of ["micro", "macro"])
  document.getElementById("avg-" + a).onclick = () => {
    state.avg = a;
    for (const x of ["micro", "macro"]) document.getElementById("avg-" + x).classList.toggle("on", x === a);
    renderCards();
  };
document.getElementById("selall").onclick = () => { state.sel = new Set(D.models); renderAll(); };
document.getElementById("selnone").onclick = () => { state.sel.clear(); renderAll(); };
document.getElementById("q").oninput = e => { state.q = e.target.value; renderAll(); };
document.getElementById("cfilter").onchange = e => { state.cat = e.target.value; renderAll(); };
document.getElementById("hfilter").onchange = e => { state.harness = e.target.value; renderAll(); };
document.getElementById("base").onchange = e => { state.base = e.target.value; renderMatrix(); };

renderAll();
</script>
</body>
</html>
"""


LEGACY_IMPORT_FRAMEWORKS = ("lmms-eval", "VLMEvalKit")


def import_legacy(merged: dict, models: list[str], legacy_paths: list[Path], rejected_runs=frozenset(),
                  rejected_sources=frozenset(), aliases=None, frameworks=LEGACY_IMPORT_FRAMEWORKS) -> int:
    """Fill (task, metric, model) slots no current result covers from earlier builds, marked legacy.

    Purged raw results survive only there; the mark lets the page and the coverage
    report tell a legacy number from a manifest-backed one. Only the harnesses whose
    raw results the purge destroyed are imported: every text result an earlier build
    saw is still on disk, so a legacy text cell could only re-add a number the current
    selection or column split rejected."""
    n_legacy = 0
    for legacy_path in legacy_paths:
        legacy = json.loads(legacy_path.read_text())
        for row in legacy.get("table", []):
            if row.get("framework") not in frameworks:
                continue
            key = (row["task"], row["metric"])
            target = merged.setdefault(key, {"task": row["task"], "metric": row["metric"], "framework": row["framework"], "cells": {}})
            for model, cell in row["cells"].items():
                model = (aliases or {}).get(model, model)
                run_ids = (cell.get("run"), (cell.get("prov") or {}).get("run_id"))
                source = (cell.get("source") or {}).get("path")
                if (any((row["task"], model, run_id) in rejected_runs for run_id in run_ids if run_id is not None)
                        or (source and (row["task"], str(Path(source).resolve())) in rejected_sources)):
                    continue
                if model not in target["cells"]:
                    target["cells"][model] = dict(cell, legacy=True)
                    n_legacy += 1
                    if model not in models:
                        models.append(model)
    return n_legacy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path, nargs="+",
                   help="one or more runs trees; each immediate child dir is a run identity, unioned across roots")
    p.add_argument("--models", nargs="*", help="filter model dirs by substring")
    p.add_argument("--only", nargs="*", help="curate to these exact canonical checkpoint keys (drops all others)")
    p.add_argument("--label", nargs="*", default=[], help="override column labels as 'canonical_key=Display Name'")
    p.add_argument("--vlmeval-root", type=Path, help="VLMEval_Outputs tree; ingests VLMEvalKit-owned (spatial/multi-image) benchmarks, merged by checkpoint identity")
    p.add_argument("--lm-eval-root", type=Path, help="results/lm-eval tree; ingests lm-eval-owned text benchmarks, merged by checkpoint identity")
    p.add_argument("--vlmeval-results-root", type=Path,
                   help="results/VLMEvalKit tree (run/<model>/<dataset>) whose run_meta.json manifests feed the coverage report")
    p.add_argument("--legacy-json", type=Path, action="append", default=[],
                   help="dashboard.json of an earlier build; its cells fill (task, metric, model) slots no current result covers, marked legacy")
    p.add_argument("--include-spatial", action="store_true", help="include EASI spatial benchmarks from lmms-eval data (tracked on VLMEvalKit by default)")
    p.add_argument("--models-file", type=Path,
                   help="column manifest (key[|alias]=Label per line); overrides --only/--label")
    p.add_argument("--verify", action="store_true",
                   help="refuse publication if collected source directories have conflicting canonical scores")
    p.add_argument("-o", "--output", type=Path, default=Path("dashboard.html"))
    args = p.parse_args()

    aliases: dict = {}
    model_groups: dict = {}
    if args.models_file:
        args.only, args.label, aliases, model_groups = parse_models_manifest(args.models_file)

    for root in args.runs_root:
        if not root.is_dir():
            print(f"runs-root not found, skipping: {root}")
    manifests, collections = collect_inventory(
        [root for root in args.runs_root if root.is_dir()],
        vlmeval_roots=[args.vlmeval_root] if args.vlmeval_root else [],
        lm_eval_roots=[args.lm_eval_root] if args.lm_eval_root else [],
        manifest_roots=[args.vlmeval_results_root] if args.vlmeval_results_root else [],
        model_filters=args.models, include_spatial=args.include_spatial)
    source_signatures = {}
    if args.verify:
        sources = {Path(cell["source"]["path"]) for _root, _models, rows in collections
                   for row in rows for cell in row["cells"].values()}
        source_signatures = {source: _source_signature(source) for source in sources}
        if audit_inventory(collections, aliases=aliases, only_keys=args.only):
            p.exit(1, "refusing to publish a dashboard with colliding model identities\n")

    def model_key(name):
        key = canonical_model_key(name)
        return aliases.get(key, key)

    models = sorted({model_key(name) for _root, names, _rows in collections for name in names})
    if args.only:
        present = set(models)
        models = [m for m in args.only if m in present]
    # Ownership partitions benchmarks, so no (task, metric) appears in both
    # harnesses; cells merge defensively if one ever does.
    merged: dict[tuple[str, str], dict] = {}
    for _root, _models, rows in collections:
        for row in rows:
            key = (row["task"], row["metric"])
            target = merged.setdefault(key, {**row, "cells": {}})
            for name, cell in row["cells"].items():
                merge_cells(target["cells"], {model_key(name): cell})
    n_legacy = import_legacy(merged, models, args.legacy_json,
                             manifests.rejected_runs(REGISTRY, canonical_model_key, aliases),
                             {(task, source) for task, _model, _run, source in manifests.rejected_results}, aliases)
    if n_legacy:
        print(f"legacy: imported {n_legacy} cells from {len(args.legacy_json)} earlier build(s)")
    table = [merged[key] for key in sorted(merged)]
    labels = short_labels(models)
    for pair in args.label:
        key, _, disp = pair.partition("=")
        if key in labels:
            labels[key] = disp
    keep = set(models)
    cells_rows = [
        {"task": r["task"], "metric": r["metric"], "framework": r["framework"],
         "cat": category_for(r["task"]) or "Uncategorized",
         "cells": {m: v for m, v in r["cells"].items() if m in keep}}
        for r in table
    ]
    cells_rows = [r for r in cells_rows if r["cells"]]
    stray = sorted({r["task"] for r in cells_rows if r["cat"] == "Uncategorized"})
    if stray:
        print(f"WARNING: {len(stray)} tasks lack a category (shown in a trailing band): {', '.join(stray)}")
    categories = CATEGORY_ORDER + (["Uncategorized"] if stray else [])
    cells_rows.sort(key=lambda r: (categories.index(r["cat"]), r["task"].lower()))
    # Pre-select the best-covered checkpoints so the page opens with a
    # meaningful comparison instead of every sparse column at once.
    coverage = {m: sum(1 for r in cells_rows if m in r["cells"]) for m in models}
    default_selected = [] if model_groups else sorted(models, key=lambda m: -coverage[m])[:5]
    data = {
        "models": models,
        "labels": labels,
        "table": cells_rows,
        "categories": categories,
        "modality": {c: m for c, m in CATEGORY_MODALITY.items() if m != "vision"},
        "defaultSelected": default_selected,
        "groups": {m: model_groups.get(m, "Models") for m in models},
    }
    all_cells = [cell for row in cells_rows for cell in row["cells"].values()]
    data["evidence"] = {"manifest": sum(bool(c.get("prov")) for c in all_cells),
                        "legacy": sum(bool(c.get("legacy")) for c in all_cells)}
    n_vk = sum(1 for r in cells_rows if r["framework"] == "VLMEvalKit")
    n_lm = sum(1 for r in cells_rows if r["framework"] == "lm-eval")
    sources = (f"lmms-eval ({len(cells_rows) - n_vk - n_lm} rows)"
               + (f" · VLMEvalKit ({n_vk} rows)" if n_vk else "")
               + (f" · lm-eval ({n_lm} rows)" if n_lm else ""))
    meta = (
        f"<b>{len(models)}</b> checkpoints · <b>{len(cells_rows)}</b> metric rows · "
        f"{sources} · {data['evidence']['manifest']} manifest-backed / {data['evidence']['legacy']} legacy cells"
        f" · generated {datetime.datetime.now():%Y-%m-%d %H:%M}"
    )
    manifest_cells = manifests.by_cell(REGISTRY, canonical_model_key)
    if aliases:
        manifest_cells = {(t, aliases.get(mk, mk)): man for (t, mk), man in manifest_cells.items()}
    cov = coverage_report(cells_rows, models, REGISTRY, manifest_cells,
                          tasks=[t.name for t in REGISTRY.tasks.values() if t.card or t.report])
    data["coverage"] = cov["counts"]
    for key in list(args.only or []):
        if not any(key in r["cells"] for r in cells_rows) and not any(mk == key for (_t, mk) in manifest_cells):
            print(f"registry: column {key!r} has no results and no manifests")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html_text = HTML_TEMPLATE.replace("__DATA__", payload).replace("__META__", meta)
    for source, signature in source_signatures.items():
        if (signature is None or _source_signature(source) != signature
                or ineligible_reason(manifests.for_result(source))):
            p.exit(1, f"refusing to publish: source changed after collection: {source}\n")
    args.output.write_text(html_text)
    print(f"wrote {args.output.resolve()}  ({args.output.stat().st_size / 1024:.0f} KB)")
    json_path = args.output.with_name("dashboard.json")
    json_path.write_text(json.dumps(data, indent=1))
    cov_path = args.output.with_name("coverage.json")
    cov_path.write_text(json.dumps(cov, indent=1))
    reasons = ", ".join(f"{k}: {v}" for k, v in sorted(cov["by_reason"].items()))
    print(f"coverage (card+report tasks x columns): {cov['counts']['present']}/{cov['counts']['cells']} cells present; "
          f"{cov['counts']['missing']} missing ({reasons or 'none'})")
    print(f"wrote {json_path.name} and {cov_path.name} next to the page")


if __name__ == "__main__":
    main()
