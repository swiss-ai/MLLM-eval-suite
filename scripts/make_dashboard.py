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
import re
from pathlib import Path

from gather_results import newest_per_task
from metric_selection import iter_headline_metrics, normalize_score

# Spatial-intelligence benchmarks are tracked on EASI via VLMEvalKit
# (https://easi.lmms-lab.com/leaderboard/), not here. Matches the EASI block
# in metric_selection.TASK_METRIC_PRIORITY.
EASI_SPATIAL_PREFIXES = (
    "3dsrbench",
    "site_bench",
    "mmsi_bench",
    "viewspatial",
    "embspatial",
    "mindcube",
    "sparbench",
    "omnispatial",
    "erqa",
    "blink",
    "cv_bench",
    "vsibench",
    "refspatial",
    "where2place",
    # UI grounding, also not reported from this harness.
    "screenspot",
    "osworld",
)

# Spatial-intelligence + multi-image benchmarks are owned by VLMEvalKit/EASI
# (correct interleaving + EASI protocol); everything else by lmms-eval. The
# badge shows which harness produced each benchmark's number, EASI-style.
VLMEVALKIT_PREFIXES = EASI_SPATIAL_PREFIXES + ("muirbench",)


def framework_for(task: str) -> str:
    return "VLMEvalKit" if task.lower().startswith(VLMEVALKIT_PREFIXES) else "lmms-eval"


# Benchmarks dropped from every harness: cmmmu (removed), the mmlu_flan
# generative-medical subjects (exact-match scorer is format-fragile),
# ok_vqa / simplevqa (low-signal, not widely reported), and refspatial (a true
# zero-shot floor — Apertus never trained on it; points parse but always ~0).
DROPPED_TASK_PREFIXES = ("cmmmu", "mmlu_flan", "ok_vqa", "simplevqa", "refspatial", "mathvista_testmini")

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


# VLMEvalKit dataset dir -> canonical task name, restricted to the benchmarks
# VLMEvalKit owns (spatial + multi-image + UI grounding). Everything else in a
# VLMEval_Outputs tree is lmms-eval-owned and ignored here.
VK_OWNED_TASKS = {
    "BLINK": "blink", "MUIRBench": "muirbench", "EmbSpatialBench": "embspatial",
    "MMSIBench_wo_circular": "mmsi_bench", "3DSRBench": "3dsrbench",
    "CV-Bench-2D": "cv_bench_2d", "CV-Bench-3D": "cv_bench_3d", "ERQA": "erqa",
    "MindCubeBench_tiny_raw_qa": "mindcube", "OmniSpatialBench_default": "omnispatial",
    "SparBench": "sparbench", "SiteBenchImage": "site_bench", "ViewSpatialBench": "viewspatial",
    "VSI-Bench-Debiased": "vsibench", "RefSpatial_wo_unseen": "refspatial",
    "RoboSpatialHome": "robospatial", "ScreenSpot": "screenspot",
    "ScreenSpot_v2": "screenspot_v2", "ScreenSpot_Pro": "screenspot_pro", "OSWorld_G": "osworld",
    "MathVista_MINI": "mathvista_mini", "HallusionBench": "hallusionbench", "MathVerse_MINI": "mathverse",
}
_VK_HEADLINE = ("overall", "overall_accuracy", "acc", "accuracy")
_VK_AGG_LABELS = ("all", "overall", "none")
# Per-benchmark headline override where the EASI-canonical metric is not plain
# accuracy. site_bench reports chance-adjusted accuracy (overall_caa); raw
# accuracy ~2x inflates it relative to the EASI leaderboard.
VK_HEADLINE_BY_TASK = {
    "site_bench": ("overall_caa", "overall_accuracy", "accuracy"),
}


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

    def scale(value: float) -> float:
        return value * 100 if value <= 1.0 else value

    if len(header) == 2 and header[1] == "value":
        cells = {_vk_norm(r[0]): r[1] for r in data if len(r) >= 2}
        for want in headline:
            if want in cells:
                return scale(float(cells[want]))
        return None
    for want in headline:
        if want not in header:
            continue
        col = header.index(want)
        if len(data) > 1:
            for row in data:
                if any(_vk_norm(c) in _VK_AGG_LABELS for c in row):
                    return scale(float(row[col]))
        return scale(float(data[0][col]))
    return None


def collect_vlmeval(vk_root: Path, model_filters: list[str] | None):
    model_dirs = sorted(d for d in vk_root.iterdir() if d.is_dir())
    if model_filters:
        model_dirs = [d for d in model_dirs if any(s in d.name for s in model_filters)]

    rows: dict[tuple[str, str], dict[str, dict]] = {}
    models: set[str] = set()
    skipped: list[str] = []
    for mdir in model_dirs:
        canon = canonical_model_key(mdir.name)
        for vk_name, task in VK_OWNED_TASKS.items():
            if task.lower().startswith(DROPPED_TASK_PREFIXES):
                continue
            bench_dir = mdir / vk_name
            if not bench_dir.is_dir():
                continue
            accs = sorted(glob.glob(f"{bench_dir}/**/*acc*.csv", recursive=True))
            if not accs:
                skipped.append(f"{mdir.name}/{vk_name}")
                continue
            acc = Path(accs[-1])
            try:
                value = parse_vk_acc(acc, VK_HEADLINE_BY_TASK.get(task, _VK_HEADLINE))
            except (OSError, ValueError, IndexError):
                value = None
            if value is None:
                skipped.append(f"{mdir.name}/{vk_name}")
                continue
            models.add(canon)
            cell = {"v": round(value, 2), "raw": value, "run": acc.parent.name}
            rows.setdefault((task, "acc"), {})[canon] = cell

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


def collect(runs_root: Path, model_filters: list[str] | None, include_spatial: bool = False):
    model_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    if model_filters:
        model_dirs = [d for d in model_dirs if any(s in d.name for s in model_filters)]
    if not model_dirs:
        print(f"no model dirs under {runs_root}; skipping")
        return [], []

    rows: dict[tuple[str, str], dict[str, dict]] = {}
    for mdir in model_dirs:
        canon = canonical_model_key(mdir.name)
        trunc = _truncation_for(mdir.name)
        for task, path in newest_per_task(mdir).items():
            if task.lower().startswith(DROPPED_TASK_PREFIXES):
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            metrics = data.get("results", {}).get(task, {})
            run_id = path.relative_to(mdir).parts[0] if path.is_relative_to(mdir) else ""
            # Benchmarks owned by VLMEvalKit (spatial + multi-image) are tracked
            # there, not here; lmms-eval's number for them is non-authoritative.
            if not include_spatial and framework_for(task) == "VLMEvalKit":
                continue
            for row_task, metric, value in iter_headline_metrics(task, metrics):
                # mathvista is judge-canonical now, so keep ONLY its judge metric
                # (drop the stale pre-switch extraction relic); every other task's
                # judge metric is dummy-prone, so drop that. (XOR.)
                if task.lower().startswith("mathvista") != ("llm_as_judge" in metric.lower()):
                    continue
                norm = normalize_score(metric, value)
                if norm is None:
                    continue
                cell = {"v": round(norm * 100, 2), "raw": value, "run": run_id}
                if task in trunc:
                    cell["t"] = round(trunc[task] * 100, 1)
                rows.setdefault((row_task, metric), {})[canon] = cell

    models = sorted({canonical_model_key(d.name) for d in model_dirs})
    table = [
        {"task": task, "metric": metric.replace(",none", ""), "framework": "lmms-eval", "cells": cells}
        for (task, metric), cells in sorted(rows.items())
    ]
    return models, table


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
.picker-head button {
  font: 11px ui-monospace, Menlo, monospace; color: var(--ink); background: none;
  border: 1px solid var(--hair); padding: 3px 10px; cursor: pointer;
}
.picker-head button:hover { border-color: var(--red); color: var(--red); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 12px 14px; max-height: 170px; overflow: auto; }
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

/* charts: one panel per benchmark */
.charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 14px; }
.panel { border: 1px solid var(--hair); background: var(--paper); padding: 12px 14px 12px; }
.panel h3 { margin: 0; font-family: ui-monospace, Menlo, monospace; font-size: 12.5px; font-weight: 600; }
.panel .pm { font-size: 11px; color: var(--muted); margin: 1px 0 10px; }
/* harness badge — which eval framework produced this benchmark (EASI-style) */
.hbadge { display: inline-block; font-family: ui-monospace, Menlo, monospace; font-size: 9.5px; letter-spacing: .03em; font-weight: 600; padding: 1px 5px; border-radius: 2px; vertical-align: middle; margin-left: 7px; }
.hbadge.lmms { color: #2c5f7a; background: #dceaf1; }
.hbadge.vlme { color: #7a3d1e; background: #f1e2d6; }
.panel h3 .hbadge { margin-left: 6px; }
.brow { display: grid; grid-template-columns: 1fr 44px; align-items: center; gap: 8px; margin: 3px 0; }
.btrack { position: relative; height: 13px; background: var(--paper-2); }
.bfill { position: absolute; inset: 0 auto 0 0; background: var(--c); opacity: .85; }
.brow .bv { font-family: ui-monospace, Menlo, monospace; font-size: 11.5px; text-align: right; }
.brow.best .bv { color: var(--red); font-weight: 600; }
.brow.best .bfill { opacity: 1; }
.panel .nodata { color: var(--muted); font-size: 12px; }

/* matrix */
.matrix-wrap { overflow: auto; max-height: 76vh; border: 1px solid var(--hair); background: var(--paper); }
table { border-collapse: separate; border-spacing: 0; min-width: 100%; width: max-content; }
thead th {
  position: sticky; top: 0; z-index: 3; background: var(--paper-2);
  font-family: ui-monospace, Menlo, monospace; font-size: 11px; font-weight: 600;
  text-align: right; padding: 9px 13px; border-bottom: 1px solid var(--ink);
  cursor: pointer; white-space: nowrap; user-select: none;
}
thead th.task-h { text-align: left; left: 0; z-index: 4; }
thead th .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--c); margin-right: 6px; vertical-align: baseline; }
thead th .dir { color: var(--red); }
tbody td, tbody th { padding: 6px 13px; border-bottom: 1px solid var(--hair); font-size: 13px; }
tbody th.task { position: sticky; left: 0; background: var(--paper); text-align: left; font-weight: 400; z-index: 2; max-width: 330px; }
tbody tr:hover td, tbody tr:hover th.task { background: var(--paper-2); }
.task .t { font-family: ui-monospace, Menlo, monospace; font-size: 12.5px; }
.task .m { font-size: 11px; color: var(--muted); }
td.cell { text-align: right; white-space: nowrap; min-width: 92px; }
thead th:not(.task-h) { min-width: 92px; }
sup.tr { color: #c87f0a; font-size: 0.62em; margin-left: 1px; font-weight: 600; cursor: help; }
.cell.best { color: var(--red); font-weight: 600; }
.delta { font-size: 11px; margin-left: 7px; font-weight: 400; }
.delta.up { color: var(--green); } .delta.down { color: var(--red); }
.missing { color: var(--hair); text-align: center; }

footer { margin-top: 26px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--hair); padding-top: 12px; }
footer code { font-family: ui-monospace, Menlo, monospace; font-size: 11px; }
.hidden { display: none; }
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
    <span class="lbl" style="margin-left:auto">click to toggle</span>
  </div>
  <div class="chips" id="chips"></div>
</div>

<div class="cards" id="cards"></div>
<div class="cards-note" id="cards-note"></div>

<div class="controls">
  <span class="viewtab"><button id="tab-charts">Charts</button><button id="tab-matrix" class="on">Matrix</button></span>
  <span><label>filter&nbsp;</label><input id="q" type="search" placeholder="benchmark substring…" spellcheck="false"></span>
  <span><label>harness&nbsp;</label><select id="hfilter"><option value="">all</option><option value="lmms-eval">lmms-eval</option><option value="VLMEvalKit">VLMEvalKit</option></select></span>
  <span id="basewrap" class="hidden"><label>baseline&nbsp;</label><select id="base"><option value="">none</option></select></span>
  <span class="hint" style="margin-left:auto"><span class="hbadge lmms">lmms-eval</span> <span class="hbadge vlme">VLMEvalKit</span></span>
</div>

<div class="charts hidden" id="charts"></div>
<div class="matrix-wrap" id="matrix-view"><table id="matrix"></table></div>

<footer>
  Bars share one absolute 0–100 scale · newest result per task across each model's runs ·
  canonical headline metrics via <code>metric_selection.py</code> · hover for raw value and source run ·
  generated by <code>make_dashboard.py</code>
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
  view: "matrix", q: "", sort: null, dir: -1, base: "", harness: "",
};

const fmt = v => v.toFixed(1);
const hbadge = fw => fw === "VLMEvalKit"
  ? '<span class="hbadge vlme">VLMEvalKit</span>'
  : '<span class="hbadge lmms">lmms-eval</span>';
const selModels = () => D.models.filter(m => state.sel.has(m));
const visRows = () => {
  const q = state.q.toLowerCase();
  return D.table.filter(r =>
    (!q || r.task.toLowerCase().includes(q) || r.metric.toLowerCase().includes(q)) &&
    (!state.harness || r.framework === state.harness) &&
    selModels().some(m => r.cells[m]));
};

function renderChips() {
  document.getElementById("chips").innerHTML = D.models.map(m =>
    `<span class="chip ${state.sel.has(m) ? "on" : ""}" data-m="${m}" style="--c:${color[m]}" title="${m}">` +
    `<span class="dot"></span>${D.labels[m]} <span class="cov">${coverage[m]}</span></span>`).join("");
  document.querySelectorAll(".chip").forEach(c => c.onclick = () => {
    const m = c.dataset.m;
    state.sel.has(m) ? state.sel.delete(m) : state.sel.add(m);
    renderAll();
  });
  document.getElementById("selcount").textContent = `${state.sel.size}/${D.models.length} selected`;
}

function renderCards() {
  const sel = selModels();
  const common = D.table.filter(r => sel.length && sel.every(m => r.cells[m]));
  const means = sel.map(m => {
    const vs = common.map(r => r.cells[m].v);
    return { m, mean: vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null };
  });
  const top = Math.max(...means.map(x => x.mean ?? -Infinity));
  document.getElementById("cards").innerHTML = means.map(x =>
    `<div class="card ${x.mean === top && means.length > 1 ? "best" : ""}" style="--c:${color[x.m]}" title="${x.m}">` +
    `<div class="name">${D.labels[x.m]}</div>` +
    `<div class="big mono">${x.mean == null ? "—" : fmt(x.mean)}<small> / 100</small></div>` +
    `<div class="cov">${coverage[x.m]} tasks covered</div></div>`).join("");
  document.getElementById("cards-note").textContent = sel.length
    ? `macro mean over the ${common.length} benchmarks covered by all ${sel.length} selected models`
    : "select models above";
}

function renderCharts() {
  const sel = selModels();
  document.getElementById("charts").innerHTML = visRows().map(r => {
    const present = sel.filter(m => r.cells[m]);
    const best = Math.max(...present.map(m => r.cells[m].v));
    const bars = present.map(m => {
      const c = r.cells[m];
      return `<div class="brow ${c.v === best && present.length > 1 ? "best" : ""}" ` +
        `title="${m}\\n${r.metric} = ${c.raw}\\nrun: ${c.run}">` +
        `<span class="btrack"><span class="bfill" style="--c:${color[m]};width:${Math.max(1, c.v)}%"></span></span>` +
        `<span class="bv">${fmt(c.v)}</span></div>`;
    }).join("");
    return `<div class="panel"><h3>${r.task}${hbadge(r.framework)}</h3><div class="pm">${r.metric}</div>${bars || "<div class='nodata'>no data</div>"}</div>`;
  }).join("") || "<div class='nodata' style='color:var(--muted)'>nothing matches</div>";
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
  for (const r of rows) {
    const present = sel.filter(m => r.cells[m]);
    const best = Math.max(...present.map(m => r.cells[m].v));
    h += `<tr><th class="task"><div class="t">${r.task}${hbadge(r.framework)}</div><div class="m">${r.metric}</div></th>`;
    for (const m of sel) {
      const c = r.cells[m];
      if (!c) { h += "<td class='missing'>·</td>"; continue; }
      let delta = "";
      if (state.base && state.base !== m && state.sel.has(state.base)) {
        const b = r.cells[state.base];
        if (b) {
          const d = c.v - b.v;
          delta = `<span class="delta ${d >= 0 ? "up" : "down"}">${d >= 0 ? "+" : ""}${d.toFixed(1)}</span>`;
        }
      }
      const trtip = c.t != null ? `\\ntruncated: ${c.t}% hit the 32k cap` : "";
      const tr = c.t != null ? `<sup class="tr" title="${c.t}% of outputs hit the 32k token cap (non-terminating)">⌁${Math.round(c.t)}</sup>` : "";
      h += `<td class="cell mono ${c.v === best && present.length > 1 ? "best" : ""}" ` +
           `title="${m}\\n${r.metric} = ${c.raw}\\nrun: ${c.run}${trtip}">${fmt(c.v)}${delta}${tr}</td>`;
    }
    h += "</tr>";
  }
  document.getElementById("matrix").innerHTML = h + "</tbody>";
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
  state.view === "charts" ? renderCharts() : renderMatrix();
}

document.getElementById("selall").onclick = () => { state.sel = new Set(D.models); renderAll(); };
document.getElementById("selnone").onclick = () => { state.sel.clear(); renderAll(); };
document.getElementById("q").oninput = e => { state.q = e.target.value; renderAll(); };
document.getElementById("hfilter").onchange = e => { state.harness = e.target.value; renderAll(); };
document.getElementById("base").onchange = e => { state.base = e.target.value; renderMatrix(); };
document.getElementById("tab-charts").onclick = () => setView("charts");
document.getElementById("tab-matrix").onclick = () => setView("matrix");
function setView(v) {
  state.view = v;
  document.getElementById("tab-charts").classList.toggle("on", v === "charts");
  document.getElementById("tab-matrix").classList.toggle("on", v === "matrix");
  document.getElementById("charts").classList.toggle("hidden", v !== "charts");
  document.getElementById("matrix-view").classList.toggle("hidden", v !== "matrix");
  document.getElementById("basewrap").classList.toggle("hidden", v !== "matrix");
  renderAll();
}

renderAll();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path, nargs="+",
                   help="one or more runs trees; each immediate child dir is a run identity, unioned across roots")
    p.add_argument("--models", nargs="*", help="filter model dirs by substring")
    p.add_argument("--only", nargs="*", help="curate to these exact canonical checkpoint keys (drops all others)")
    p.add_argument("--label", nargs="*", default=[], help="override column labels as 'canonical_key=Display Name'")
    p.add_argument("--vlmeval-root", type=Path, help="VLMEval_Outputs tree; ingests VLMEvalKit-owned (spatial/multi-image) benchmarks, merged by checkpoint identity")
    p.add_argument("--include-spatial", action="store_true", help="include EASI spatial benchmarks from lmms-eval data (tracked on VLMEvalKit by default)")
    p.add_argument("-o", "--output", type=Path, default=Path("dashboard.html"))
    args = p.parse_args()

    models_l: list[str] = []
    table_l: list[dict] = []
    for root in args.runs_root:
        if not root.is_dir():
            print(f"runs-root not found, skipping: {root}")
            continue
        m, t = collect(root.resolve(), args.models, args.include_spatial)
        models_l = sorted(set(models_l) | set(m))
        table_l += t
    models_v, table_v = ([], [])
    if args.vlmeval_root:
        models_v, table_v = collect_vlmeval(args.vlmeval_root.resolve(), args.models)
    models = sorted(set(models_l) | set(models_v))
    if args.only:
        present = set(models)
        models = [m for m in args.only if m in present]
    # Ownership partitions benchmarks, so no (task, metric) appears in both
    # harnesses; cells merge defensively if one ever does.
    merged: dict[tuple[str, str], dict] = {}
    for row in table_l + table_v:
        key = (row["task"], row["metric"])
        if key in merged:
            merged[key]["cells"].update(row["cells"])
        else:
            merged[key] = dict(row)
    table = [merged[key] for key in sorted(merged)]
    labels = short_labels(models)
    for pair in args.label:
        key, _, disp = pair.partition("=")
        if key in labels:
            labels[key] = disp
    keep = set(models)
    cells_rows = [
        {"task": r["task"], "metric": r["metric"], "framework": r["framework"],
         "cells": {m: v for m, v in r["cells"].items() if m in keep}}
        for r in table
    ]
    cells_rows = [r for r in cells_rows if r["cells"]]
    # Pre-select the best-covered checkpoints so the page opens with a
    # meaningful comparison instead of every sparse column at once.
    coverage = {m: sum(1 for r in cells_rows if m in r["cells"]) for m in models}
    default_selected = models if args.only else sorted(models, key=lambda m: -coverage[m])[:5]
    data = {
        "models": models,
        "labels": labels,
        "table": cells_rows,
        "defaultSelected": default_selected,
    }
    n_vk = sum(1 for r in cells_rows if r["framework"] == "VLMEvalKit")
    sources = f"lmms-eval ({len(cells_rows) - n_vk} rows)" + (f" · VLMEvalKit ({n_vk} rows)" if n_vk else "")
    meta = (
        f"<b>{len(models)}</b> checkpoints · <b>{len(cells_rows)}</b> metric rows · "
        f"{sources} · generated {datetime.datetime.now():%Y-%m-%d %H:%M}"
    )
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html_text = HTML_TEMPLATE.replace("__DATA__", payload).replace("__META__", meta)
    args.output.write_text(html_text)
    print(f"wrote {args.output.resolve()}  ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
