#!/usr/bin/env python3
"""Re-grade cached HealthBench completions with a different grader model.

Replays the exact rubric-grading protocol by calling the helpers in
third_party/lmms-eval/lmms_eval/tasks/healthbench/utils.py (_grade_rubric,
_convo_str, calculate_score, agg_clipped_mean) against an arbitrary
OpenAI-compatible grader — e.g. gpt-4.1, the official HealthBench grader —
without re-running inference. Grades append to a jsonl keyed by
prompt_id:rubric_idx as each call completes, so interrupted runs resume for
free.

Usage:
  OPENAI_API_KEY=... python3 scripts/regrade_healthbench.py \
    --samples <samples_healthbench.jsonl> --grader gpt-4.1 \
    --out <grades.jsonl> [--limit N] [--concurrency 32]
"""
import argparse
import importlib.util
import json
import os
import sys
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CANONICAL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"


def load_healthbench_utils(grader: str):
    os.environ["HEALTHBENCH_GRADER_MODEL"] = grader
    if "loguru" not in sys.modules:
        stub = types.ModuleType("loguru")
        stub.logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None)
        sys.modules["loguru"] = stub
    path = Path(__file__).resolve().parents[1] / "third_party/lmms-eval/lmms_eval/tasks/healthbench/utils.py"
    spec = importlib.util.spec_from_file_location("healthbench_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_docs(cache: Path) -> dict:
    if not cache.exists():
        urllib.request.urlretrieve(CANONICAL, cache)
    docs = {}
    for ln in cache.read_text().splitlines():
        d = json.loads(ln)
        docs[d["prompt_id"]] = d
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--grader", default="gpt-4.1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    hb = load_healthbench_utils(args.grader)
    docs = load_docs(Path(args.out).parent / "healthbench_oss_eval.jsonl")
    samples = [json.loads(ln) for ln in open(args.samples)]
    if args.limit:
        samples = samples[: args.limit]

    grades = {}
    out = Path(args.out)
    if out.exists():
        for ln in out.read_text().splitlines():
            g = json.loads(ln)
            grades[(g["prompt_id"], g["rubric_idx"])] = g

    jobs = []
    for s in samples:
        pid = s["target"]
        doc = docs[pid]
        fr = s["filtered_resps"]
        completion = fr if isinstance(fr, str) else fr[0]
        convo_str = hb._convo_str(list(doc["prompt"]) + [{"role": "assistant", "content": completion}])
        for idx, rubric in enumerate(doc["rubrics"]):
            if (pid, idx) not in grades:
                jobs.append((pid, idx, convo_str, rubric))

    print(f"{len(jobs)} gradings to run ({len(grades)} already done)", flush=True)
    with open(out, "a") as sink, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(hb._grade_rubric, convo, rubric): (pid, idx)
                   for pid, idx, convo, rubric in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            pid, idx = futures[fut]
            g = fut.result()
            rec = {"prompt_id": pid, "rubric_idx": idx,
                   "criteria_met": bool(g.get("criteria_met", False)),
                   "failed": bool(g.get("healthbench_grader_failed", False))}
            grades[(pid, idx)] = rec
            sink.write(json.dumps(rec) + "\n")
            if n % 200 == 0:
                sink.flush()
                print(f"{n}/{len(jobs)}", flush=True)

    scores, failures, total = [], 0, 0
    for s in samples:
        pid = s["target"]
        rubrics = docs[pid]["rubrics"]
        glist = [grades.get((pid, i), {"criteria_met": False, "failed": True}) for i in range(len(rubrics))]
        failures += sum(g.get("failed", False) for g in glist)
        total += len(rubrics)
        scores.append(hb.calculate_score(rubrics, glist))
    print(f"healthbench_score ({args.grader}): {hb.agg_clipped_mean(scores) * 100:.1f}")
    print(f"grader_failure_rate: {failures / total * 100:.2f}%  ({failures}/{total})")


if __name__ == "__main__":
    main()
