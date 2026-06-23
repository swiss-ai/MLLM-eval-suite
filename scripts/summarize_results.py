#!/usr/bin/env python3
"""Summarize one Apertus eval array run into Markdown, CSV, JSON, and HTML."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import quote

from metric_selection import iter_headline_metrics, normalize_score

CATEGORY_PREFIXES = [
    ("Medical", ("med", "mmlu_flan", "pubmed", "path_vqa", "vqa_rad", "slake", "pmc_vqa")),
    ("Document/OCR", ("docvqa", "infovqa", "textvqa", "chartqa", "ocrbench", "omnidocbench")),
    ("Math/STEM", ("math", "scienceqa")),
    ("Grounding", ("refcoco", "screenspot")),
    ("Spatial/Embodied", ("embspatial", "3dsrbench", "cv_bench", "mindcube", "mmsi", "viewspatial", "erqa", "vstar", "site_bench", "sparbench", "omnispatial", "blink", "osworld")),
    ("Multi-image", ("seedbench", "muirbench", "mmvp")),
    ("Knowledge/Reasoning", ("mmmu", "cmmmu", "visulogic", "visualpuzzles", "simplevqa")),
    ("Hallucination", ("hallusion", "vlmsareblind", "vlms_are_biased", "pope")),
]


def read_tasks(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_category(task: str) -> str:
    lowered = task.lower()
    for category, prefixes in CATEGORY_PREFIXES:
        if lowered.startswith(prefixes):
            return category
    return "General VQA"


def load_result_rows(run_root: Path, expected_tasks: list[str]) -> list[dict[str, Any]]:
    results_root = run_root / "results"
    rows: list[dict[str, Any]] = []

    for source_task in expected_tasks:
        task_root = results_root / source_task
        result_files = sorted(task_root.rglob("*_results.json")) if task_root.exists() else []
        for result_file in result_files:
            try:
                payload = json.loads(result_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for task, metrics in sorted((payload.get("results") or {}).items()):
                if not isinstance(metrics, dict):
                    continue
                for row_task, metric, value in iter_headline_metrics(task, metrics):
                    rows.append(
                        {
                            "task": row_task,
                            "source_task": source_task,
                            "status": "completed",
                            "category": task_category(row_task),
                            "metric": metric,
                            "value": value,
                            "result_file": str(result_file),
                        }
                    )

    return rows


def logs_for_task(log_dir: Path, model_label: str, array_job_id: str, task_index: int) -> tuple[str, str]:
    prefix = f"eval_{model_label}_{array_job_id}_{task_index}"
    out = log_dir / f"{prefix}.out"
    err = log_dir / f"{prefix}.err"
    return (str(out) if out.exists() else "", str(err) if err.exists() else "")


def build_summary(run_root: Path, task_list: Path, model_label: str, array_job_id: str) -> dict[str, Any]:
    expected_tasks = read_tasks(task_list)
    log_dir = run_root / "logs"
    result_rows = load_result_rows(run_root, expected_tasks)
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        rows_by_source.setdefault(row["source_task"], []).append(row)

    final_rows: list[dict[str, Any]] = []
    completed_sources = 0
    for index, task in enumerate(expected_tasks):
        out_log, err_log = logs_for_task(log_dir, model_label, array_job_id, index)
        task_rows = rows_by_source.get(task, [])
        if task_rows:
            completed_sources += 1
            for row in task_rows:
                row = dict(row)
                row["array_index"] = index
                row["stdout_log"] = out_log
                row["stderr_log"] = err_log
                final_rows.append(row)
        else:
            final_rows.append(
                {
                    "task": task,
                    "source_task": task,
                    "status": "missing",
                    "category": task_category(task),
                    "metric": "",
                    "value": None,
                    "array_index": index,
                    "result_file": "",
                    "stdout_log": out_log,
                    "stderr_log": err_log,
                }
            )

    category_values: dict[str, list[float]] = {}
    for row in final_rows:
        if row["status"] != "completed" or row["value"] is None:
            continue
        norm = normalize_score(row["metric"], float(row["value"]))
        if norm is None or not 0 <= norm <= 1:
            continue
        category_values.setdefault(row["category"], []).append(norm)

    category_averages = {category: round(mean(values), 6) for category, values in sorted(category_values.items()) if values}
    return {
        "model_label": model_label,
        "array_job_id": array_job_id,
        "run_root": str(run_root),
        "task_list": str(task_list),
        "total": len(expected_tasks),
        "completed": completed_sources,
        "missing": len(expected_tasks) - completed_sources,
        "category_averages": category_averages,
        "tasks": final_rows,
    }


def format_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_json(summary_dir: Path, summary: dict[str, Any]) -> None:
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")


def write_csv(summary_dir: Path, summary: dict[str, Any]) -> None:
    columns = ["task", "source_task", "status", "category", "metric", "value", "array_index", "result_file", "stdout_log", "stderr_log"]
    with (summary_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in summary["tasks"]:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_markdown(summary_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Apertus Eval Summary",
        "",
        f"- Model: `{summary['model_label']}`",
        f"- Array job: `{summary['array_job_id']}`",
        f"- Run root: `{summary['run_root']}`",
        f"- Completed: {summary['completed']}/{summary['total']}",
        f"- Missing: {summary['missing']}",
        "",
    ]
    if summary["category_averages"]:
        lines.extend(["## Category Averages", ""])
        lines.append("| Category | Score |")
        lines.append("|---|---:|")
        for category, score in summary["category_averages"].items():
            lines.append(f"| {category} | {score:.4f} |")
        lines.append("")

    lines.extend(["## Tasks", "", "| Task | Status | Metric | Value | Result | Logs |", "|---|---|---|---:|---|---|"])
    for row in summary["tasks"]:
        result = row.get("result_file") or ""
        logs = " ".join(path for path in (row.get("stdout_log"), row.get("stderr_log")) if path)
        lines.append(
            f"| {row['task']} | {row['status']} | {row.get('metric', '')} | {format_value(row.get('value'))} | {result} | {logs} |"
        )
    lines.append("")
    (summary_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def html_link(summary_dir: Path, path: str, label: str) -> str:
    if not path:
        return ""
    rel_path = os.path.relpath(path, start=summary_dir)
    href = quote(rel_path, safe="/")
    return f'<a class="artifact" href="{href}">{html.escape(label)}</a>'


def write_html(summary_dir: Path, summary: dict[str, Any]) -> None:
    total = int(summary["total"])
    completed = int(summary["completed"])
    missing = int(summary["missing"])
    completion = (completed / total) if total else 0.0
    category_scores = list(summary["category_averages"].values())
    mean_score = mean(category_scores) if category_scores else None

    stat_cards = [
        ("Completed", f"{completed}/{total}", f"{completion * 100:.1f}%"),
        ("Missing", str(missing), "needs review" if missing else "clear"),
        ("Category mean", format_value(mean_score), "normalized"),
        ("Array job", summary["array_job_id"], "afterany summary"),
    ]
    stats_html = "".join(
        "<div class=\"stat\">"
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        f"<em>{html.escape(note)}</em>"
        "</div>"
        for label, value, note in stat_cards
    )

    rows = []
    for row in summary["tasks"]:
        value = format_value(row.get("value"))
        status = str(row["status"])
        result_link = html_link(summary_dir, row.get("result_file", ""), "result")
        stdout_link = html_link(summary_dir, row.get("stdout_log", ""), "stdout")
        stderr_link = html_link(summary_dir, row.get("stderr_log", ""), "stderr")
        links = " ".join(link for link in (result_link, stdout_link, stderr_link) if link) or "<span class=\"muted\">none</span>"
        rows.append(
            f"<tr data-status=\"{html.escape(status)}\">"
            f"<td class=\"task-name\">{html.escape(str(row['task']))}</td>"
            f"<td><span class=\"badge {html.escape(status)}\">{html.escape(status)}</span></td>"
            f"<td>{html.escape(str(row.get('category', '')))}</td>"
            f"<td>{html.escape(str(row.get('metric', '')))}</td>"
            f"<td class=\"num\">{html.escape(value)}</td>"
            f"<td class=\"num\">{html.escape(str(row.get('array_index', '')))}</td>"
            f"<td class=\"links\">{links}</td>"
            "</tr>"
        )

    cat_rows = []
    for category, score in summary["category_averages"].items():
        width = max(0, min(100, int(round(score * 100))))
        cat_rows.append(
            "<div class=\"category-row\">"
            f"<div><strong>{html.escape(category)}</strong><span>{score:.4f}</span></div>"
            f"<div class=\"bar\"><span style=\"width:{width}%\"></span></div>"
            "</div>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apertus Eval Summary</title>
<style>
:root {{
  --ink: #1a1a1a;
  --paper: #faf9f7;
  --surface: #ffffff;
  --line: #e0ded9;
  --muted: #74706a;
  --blue: #2d5a7b;
  --green: #2d7b5a;
  --amber: #8a5a22;
  --red: #9b3b3b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
}}
header {{
  padding: 42px min(6vw, 72px) 30px;
  border-bottom: 1px solid var(--line);
}}
h1 {{
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(32px, 4vw, 52px);
  line-height: 1.05;
  letter-spacing: 0;
}}
.subtitle {{
  margin-top: 10px;
  color: var(--muted);
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
}}
main {{
  max-width: 1240px;
  margin: 0 auto;
  padding: 28px min(5vw, 56px) 52px;
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 28px;
}}
.stat {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px 18px;
}}
.stat span, .eyebrow {{
  display: block;
  color: var(--muted);
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .04em;
}}
.stat strong {{
  display: block;
  margin-top: 6px;
  font-size: 28px;
  line-height: 1.1;
  font-weight: 650;
  font-variant-numeric: tabular-nums;
}}
.stat em {{
  display: block;
  margin-top: 5px;
  color: var(--muted);
  font-size: 13px;
  font-style: normal;
}}
.section {{
  margin-top: 30px;
}}
h2 {{
  margin: 0 0 12px;
  font-size: 20px;
  letter-spacing: 0;
}}
.meta {{
  margin-top: 14px;
  max-width: 100%;
  color: var(--muted);
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}}
.categories {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 28px;
}}
.category-row {{
  padding: 12px 0;
  border-bottom: 1px solid var(--line);
}}
.category-row div:first-child {{
  display: flex;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 8px;
  font-variant-numeric: tabular-nums;
}}
.category-row span {{ color: var(--blue); font-weight: 650; }}
.bar {{
  height: 8px;
  background: #ece9e3;
  border-radius: 999px;
  overflow: hidden;
}}
.bar span {{
  display: block;
  height: 100%;
  background: linear-gradient(90deg, var(--blue), var(--green));
}}
.filters {{
  display: flex;
  gap: 8px;
  margin: 14px 0 12px;
  flex-wrap: wrap;
}}
.filters button {{
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--ink);
  border-radius: 8px;
  padding: 7px 11px;
  cursor: pointer;
  font: inherit;
  font-size: 13px;
}}
.filters button.active {{
  border-color: var(--ink);
  background: var(--ink);
  color: var(--surface);
}}
.table-wrap {{
  overflow-x: auto;
  border-top: 2px solid var(--ink);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}}
th {{
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--paper);
  color: var(--muted);
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .04em;
  text-align: left;
  padding: 12px 10px;
  border-bottom: 1px solid var(--line);
}}
td {{
  padding: 11px 10px;
  border-bottom: 1px solid var(--line);
  vertical-align: middle;
}}
tbody tr:hover {{ background: rgba(26, 26, 26, .035); }}
.task-name {{
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  font-weight: 600;
}}
.num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}
.badge {{
  display: inline-flex;
  min-width: 78px;
  justify-content: center;
  border-radius: 999px;
  padding: 3px 9px;
  font-size: 12px;
  font-weight: 650;
}}
.badge.completed {{ color: #174c34; background: #dff3e8; }}
.badge.missing {{ color: #6b3d05; background: #faedd3; }}
.badge.failed {{ color: #7d2424; background: #f8dddd; }}
.links {{
  white-space: nowrap;
}}
.artifact {{
  display: inline-flex;
  margin-right: 6px;
  color: var(--blue);
  text-decoration: none;
  border-bottom: 1px solid rgba(45, 90, 123, .35);
  font-size: 13px;
}}
.artifact:hover {{ border-bottom-color: var(--blue); }}
.muted {{ color: var(--muted); }}
tr.is-hidden {{ display: none; }}
@media (max-width: 820px) {{
  header {{ padding: 28px 20px 22px; }}
  main {{ padding: 22px 18px 40px; }}
  .stats, .categories {{ grid-template-columns: 1fr; }}
  .stat strong {{ font-size: 24px; }}
}}
</style>
</head>
<body>
<header>
<h1>Apertus 1.5 Evaluation</h1>
<div class="subtitle">Array summary for {html.escape(summary['model_label'])}</div>
<div class="meta">
Model: {html.escape(summary['model_label'])}<br>
Array job: {html.escape(summary['array_job_id'])}<br>
Completed: {summary['completed']}/{summary['total']} &nbsp; Missing: {summary['missing']}<br>
Run root: {html.escape(summary['run_root'])}
</div>
</header>
<main>
<section class="stats">{stats_html}</section>
<section class="section">
<span class="eyebrow">Normalized by category</span>
<h2>Category Averages</h2>
<div class="categories">{''.join(cat_rows) or '<p class="muted">No completed category scores yet.</p>'}</div>
</section>
<section class="section">
<span class="eyebrow">Task outcomes</span>
<h2>Tasks</h2>
<div class="filters">
  <button class="active" data-status="all">All</button>
  <button data-status="completed">Completed</button>
  <button data-status="missing">Missing</button>
</div>
<div class="table-wrap">
<table><thead><tr><th>Task</th><th>Status</th><th>Category</th><th>Metric</th><th class="num">Value</th><th class="num">Array</th><th>Artifacts</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
</div>
</section>
</main>
<script>
document.querySelectorAll('.filters button').forEach((button) => {{
  button.addEventListener('click', () => {{
    document.querySelectorAll('.filters button').forEach((b) => b.classList.remove('active'));
    button.classList.add('active');
    const status = button.dataset.status;
    document.querySelectorAll('tbody tr[data-status]').forEach((row) => {{
      row.classList.toggle('is-hidden', status !== 'all' && row.dataset.status !== status);
    }});
  }});
}});
</script>
</body>
</html>
"""
    (summary_dir / "dashboard.html").write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--task-list", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--array-job-id", required=True)
    args = parser.parse_args()

    summary_dir = args.run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.run_root, args.task_list, args.model_label, args.array_job_id)
    write_json(summary_dir, summary)
    write_csv(summary_dir, summary)
    write_markdown(summary_dir, summary)
    write_html(summary_dir, summary)

    print(f"Summary written to {summary_dir / 'summary.md'}")
    print(f"CSV written to {summary_dir / 'summary.csv'}")
    print(f"JSON written to {summary_dir / 'summary.json'}")
    print(f"HTML written to {summary_dir / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
