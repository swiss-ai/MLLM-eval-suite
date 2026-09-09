import json

from suite.status import annotate_slurm, format_table, scan, summary


def _write(path, **fields):
    path.mkdir(parents=True)
    base = {"run_id": "r1", "framework": "lmms-eval", "task": "gqa", "model": {"name": "m"}, "status": "ok",
            "error": None, "started_at": 1, "finished_at": 2, "slurm": {"job_id": "9"}}
    base.update(fields)
    (path / "run_meta.json").write_text(json.dumps(base))


def test_scan_format_and_summary(tmp_path):
    _write(tmp_path / "lmms-eval" / "m" / "r1" / "gqa")
    _write(tmp_path / "VLMEvalKit" / "r1" / "m" / "BLINK", framework="VLMEvalKit", task="BLINK", status="failed",
           error="OOM", slurm={"job_id": "10"})
    _write(tmp_path / "lmms-eval" / "m" / "r1" / "pope", task="pope", status="running", finished_at=None, slurm={"job_id": "11"})
    rows = scan(tmp_path)
    assert [(r["task"], r["status"]) for r in rows] == [("BLINK", "failed"), ("gqa", "ok"), ("pope", "running")]
    text = format_table(rows)
    assert "OOM" in text and "gqa" in text
    assert summary(rows) == "status: failed=1, ok=1, running=1"


def test_annotate_slurm_marks_aborted_runs(tmp_path):
    _write(tmp_path / "lmms-eval" / "m" / "r1" / "pope", task="pope", status="running", slurm={"job_id": "11"})
    _write(tmp_path / "lmms-eval" / "m" / "r1" / "gqa", task="gqa", status="running", slurm={"job_id": "12"})
    rows = scan(tmp_path)
    annotate_slurm(rows, {"11": "RUNNING", "12": "CANCELLED+"})
    assert {r["task"]: r["status"] for r in rows} == {"pope": "running/RUNNING", "gqa": "aborted/CANCELLED+"}
