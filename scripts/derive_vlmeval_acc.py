#!/usr/bin/env python3
"""Derive *_acc.csv for VLMEvalKit judge benchmarks from the job logs.

This VLMEvalKit fork computes the judge score but does not persist a result
file to the work-dir — only a run log. The eval job's summary table in the
.out log carries the headline value, so we parse it and emit the minimal
acc.csv that make_dashboard.parse_vk_acc reads. Runs whose judge_fail_rate is
high (a broken judge, e.g. MMVet's wrapper pre-flight) are skipped — those have
no real score, so we never fabricate one.
"""
import glob
import re
from pathlib import Path

SUITE = Path("/iopsstor/scratch/cscs/xyixuan/apertus/MLLM-eval-suite")
LOGS = SUITE / "logs/VLMEvalKit"
RESULTS = SUITE / "results/VLMEvalKit"
# Single source of truth: the VLMEvalKit judge suite. Adding a judge benchmark
# to llm_judge.txt is enough; the deriver picks it up here.
JUDGE_BENCH = {
    ln.strip()
    for ln in (SUITE / "task_suites/VLMEvalKit/llm_judge.txt").read_text().splitlines()
    if ln.strip() and not ln.lstrip().startswith("#")
}
MAX_JUDGE_FAIL = 20.0  # percent; above this the judge was broken -> no real score

# benchmark  infer% (n/N)  judge% (n/N)  metric  value  ...
ROW = re.compile(r"^(\S+)\s+([\d.]+)%\s+\([\d/]+\)\s+([\d.]+)%\s+\([\d/]+\)\s+.+?\s+([\d.]+)(?=\s|$)")


def main() -> None:
    written = skipped = 0
    for log in glob.glob(str(LOGS / "*/vlmeval-*.out")):
        run_id = Path(log).parent.name
        try:
            text = Path(log).read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            m = ROW.match(line.strip())
            if not m:
                continue
            bench, infer_fail, judge_fail, value = m.groups()
            if bench not in JUDGE_BENCH:
                continue
            if float(judge_fail) > MAX_JUDGE_FAIL or float(infer_fail) > 5.0 or float(value) <= 0:
                skipped += 1
                continue
            for wd in glob.glob(str(RESULTS / run_id / "*" / bench)):
                if glob.glob(f"{wd}/**/*acc*.csv", recursive=True):
                    continue
                (Path(wd) / "derived_acc.csv").write_text(f"metric,value\noverall,{value}\n")
                written += 1
    print(f"derived {written} acc.csv, skipped {skipped} (broken judge)")


if __name__ == "__main__":
    main()
