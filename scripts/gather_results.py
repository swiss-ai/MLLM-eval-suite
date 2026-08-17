#!/usr/bin/env python3
"""
Walk results/<model>/<*_results.json>, extract the headline metric per task,
and print a comparison table (rows=tasks, cols=models). Optionally write CSV.

Usage:
  python scripts/gather_results.py
  python scripts/gather_results.py --results-root results --csv summary.csv
  python scripts/gather_results.py --models constant linear DPO   # filter by substring
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from metric_selection import pick_headline_metric

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = (SCRIPT_DIR.parent / "results").resolve()


def pick_headline(task: str, metrics: dict) -> tuple[str, float] | None:
    """Return (metric_name_stripped_of_,none, value).

    Strategy: use shared task-aware policy, then skip non-numeric metrics.
    """
    metric, value = pick_headline_metric(task, metrics)
    if metric is None or value is None:
        return None
    return metric, value


def newest_per_task(model_dir: Path) -> dict[str, Path]:
    """For each task in this model dir, return path to newest *_results.json."""
    by_task: dict[str, Path] = {}
    for path in model_dir.rglob("*_results.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for task in data.get("results", {}):
            cur = by_task.get(task)
            if cur is None or path.stat().st_mtime > cur.stat().st_mtime:
                by_task[task] = path
    return by_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--csv", default=None, help="optional path to write CSV")
    p.add_argument("--models", nargs="*", help="filter model dirs by substring")
    p.add_argument("--markdown", action="store_true", help="emit markdown table")
    args = p.parse_args()

    root = Path(args.results_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"results root not found: {root}")

    model_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.models:
        model_dirs = [d for d in model_dirs if any(s in d.name for s in args.models)]
    if not model_dirs:
        raise SystemExit(f"no model dirs in {root}")

    # table[task][model] = (metric_name, value)
    table: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
    metric_name_per_task: dict[str, str] = {}

    for mdir in model_dirs:
        for task, path in newest_per_task(mdir).items():
            data = json.loads(path.read_text())
            metrics = data["results"].get(task, {})
            headline = pick_headline(task, metrics)
            if headline is None:
                continue
            mname, val = headline
            table[task][mdir.name] = (mname, val)
            metric_name_per_task.setdefault(task, mname)

    if not table:
        raise SystemExit("no results found")

    tasks = sorted(table)
    model_names = [d.name for d in model_dirs]

    # Width for pretty printing
    task_w = max(len(t) for t in tasks)
    task_w = max(task_w, len("task"))
    metric_w = max(len(m) for m in metric_name_per_task.values()) if metric_name_per_task else 6
    metric_w = max(metric_w, len("metric"))
    model_w = max((len(m) for m in model_names), default=8)
    model_w = max(model_w, 8)

    if args.markdown:
        cols = ["task", "metric"] + model_names
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join("---" for _ in cols) + "|")
        for t in tasks:
            row = [t, metric_name_per_task.get(t, "")]
            for m in model_names:
                cell = table[t].get(m)
                row.append(f"{cell[1]:.4f}" if cell else "")
            print("| " + " | ".join(row) + " |")
    else:
        header = f"{'task':<{task_w}}  {'metric':<{metric_w}}  " + "  ".join(
            f"{m:>{model_w}}" for m in model_names
        )
        print(header)
        print("-" * len(header))
        for t in tasks:
            mname = metric_name_per_task.get(t, "")
            row = f"{t:<{task_w}}  {mname:<{metric_w}}  "
            for m in model_names:
                cell = table[t].get(m)
                row += f"{cell[1]:>{model_w}.4f}" if cell else f"{'':>{model_w}}"
                row += "  "
            print(row.rstrip())

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["task", "metric"] + model_names)
            for t in tasks:
                row = [t, metric_name_per_task.get(t, "")]
                for m in model_names:
                    cell = table[t].get(m)
                    row.append(f"{cell[1]:.6f}" if cell else "")
                w.writerow(row)
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
