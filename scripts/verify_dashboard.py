#!/usr/bin/env python3
"""Audit canonical-key collisions in the dashboard pipeline.

A collision is when more than one run dir maps to the same canonical checkpoint
key AND those runs disagree on a shared benchmark. The dashboard merge keeps the
last writer, so a collision silently shows one run's number under another run's
identity (e.g. a -thinking-32k run overwriting its direct sibling). This catches
that class of bug before it reaches the page.

Usage:
  python3 verify_dashboard.py --runs-root /path [--vlmeval-root /path]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gather_results import newest_per_task
from make_dashboard import REGISTRY, canonical_model_key
from metric_selection import iter_headline_metrics


def task_values(mdir: Path) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for task, (path, data) in newest_per_task(mdir).items():
        metrics = data.get("results", {}).get(task, {})
        for row_task, metric, value in iter_headline_metrics(task, metrics, *REGISTRY.headline_for("lmms-eval", task)):
            out[(row_task, metric)] = value
    return out


def audit(roots: list[Path]) -> int:
    by_key: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        for d in sorted(root.iterdir()):
            if d.is_dir():
                by_key[canonical_model_key(d.name)].append(d)

    bad = 0
    for key, dirs in sorted(by_key.items()):
        if len(dirs) < 2:
            continue
        vals = {d.name: task_values(d) for d in dirs}
        conflicts = []
        for t in sorted(set().union(*(v.keys() for v in vals.values()))):
            present = {n: v[t] for n, v in vals.items() if t in v}
            if len({round(x, 3) for x in present.values()}) > 1:
                conflicts.append((t, present))
        if conflicts:
            bad += 1
            print(f"COLLISION  key '{key}'  <-  {len(dirs)} dirs:")
            for d in dirs:
                print(f"             {d.name}")
            print(f"           {len(conflicts)} conflicting benchmark(s), e.g.:")
            for (task, metric), present in conflicts[:3]:
                shown = ", ".join(f"{n[-28:]}={x:.1f}" for n, x in present.items())
                print(f"             {task}: {shown}")
    print(f"\n{'PASS — no contaminating collisions' if not bad else f'FAIL — {bad} colliding key(s)'}")
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--vlmeval-root", type=Path)
    args = p.parse_args()
    roots = [args.runs_root.resolve()] + ([args.vlmeval_root.resolve()] if args.vlmeval_root else [])
    sys.exit(1 if audit(roots) else 0)


if __name__ == "__main__":
    main()
