"""Run states from manifests, optionally annotated with Slurm accounting."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from suite import REPO_ROOT

TERMINAL_OK = {"COMPLETED"}


def scan(results_root: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(results_root).rglob("run_meta.json")):
        try:
            m = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({"run_id": m.get("run_id"), "framework": m.get("framework"), "task": m.get("task"),
                     "model": (m.get("model") or {}).get("name"), "status": m.get("status"), "error": m.get("error"),
                     "started_at": m.get("started_at"), "finished_at": m.get("finished_at"),
                     "job_id": (m.get("slurm") or {}).get("job_id"), "path": str(path)})
    rows.sort(key=lambda r: (r["run_id"] or "", r["framework"] or "", r["task"] or ""))
    return rows


def annotate_slurm(rows: list[dict], sacct=None) -> None:
    ids = [r["job_id"] for r in rows if r["status"] == "running" and r["job_id"]]
    if not ids:
        return
    if sacct is None:
        out = subprocess.run(["sacct", "-j", ",".join(ids), "-X", "-n", "-o", "JobID,State"],
                             capture_output=True, text=True).stdout
        sacct = {l.split()[0]: l.split()[1] for l in out.splitlines() if l.strip()}
    for r in rows:
        if r["status"] == "running" and r["job_id"] in sacct:
            state = sacct[r["job_id"]]
            r["status"] = f"running/{state}" if state == "RUNNING" or state == "PENDING" else f"aborted/{state}"


def format_table(rows: list[dict]) -> str:
    lines = [f"{'run_id':28} {'fw':10} {'task':24} {'model':30} {'status':18} error"]
    for r in rows:
        lines.append(f"{(r['run_id'] or '')[:28]:28} {(r['framework'] or '')[:10]:10} {(r['task'] or '')[:24]:24} "
                     f"{(r['model'] or '')[:30]:30} {(r['status'] or '')[:18]:18} {(r['error'] or '')[:80]}")
    return "\n".join(lines)


def summary(rows: list[dict]) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        key = (r["status"] or "?").split("/")[0]
        counts[key] = counts.get(key, 0) + 1
    return "status: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluation run status from manifests")
    p.add_argument("--root", default=str(REPO_ROOT / "results"))
    p.add_argument("--run-id")
    p.add_argument("--slurm", action="store_true", help="annotate running manifests with sacct state")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    rows = [r for r in scan(Path(a.root)) if not a.run_id or r["run_id"] == a.run_id]
    if a.slurm:
        annotate_slurm(rows)
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        print(format_table(rows))
        print(summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
