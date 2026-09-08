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
HARNESS_DIRS = {"lmms-eval": "lmms-eval", "VLMEvalKit": "VLMEvalKit", "lm-eval": "lm-eval-harness"}
ERROR_PATTERNS = (
    r"Error during evaluation: (.{0,300})",
    r"(torch\.OutOfMemoryError: .{0,200})",
    r"(EngineDeadError.{0,200})",
    r"(RuntimeError: cancelled)",
    r"(thinking canary failed.{0,200})",
    r"(Engine core initialization failed.{0,100})",
    r"(Worker proc \S+ died unexpectedly.{0,80})",
)
FLAG_RE = re.compile(r"apertus_1p5(?:_vllm)?: enable_thinking=(\S+)")
CANARY_RE = re.compile(r"thinking canary (passed|failed)")


def _git(path: Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain", "--untracked-files=no")) if commit else None
    return {"commit": commit or None, "dirty": dirty, "checkout": str(path)}


def _model_identity(model_path: Path) -> dict:
    model_path = Path(model_path)
    ident = {"path": str(model_path), "realpath": str(model_path.resolve()), "config_sha256": None,
             "weights": {"index_sha256": None, "shards": []}}
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
    harness_dir = REPO_ROOT / "third_party" / HARNESS_DIRS.get(framework, framework)
    manifest = {
        "schema": SCHEMA, "run_id": run_id, "framework": framework, "task": task,
        "status": "running", "error": None, "started_at": int(time.time()), "finished_at": None,
        "model": {"name": Path(model_path).name, **_model_identity(Path(model_path))},
        "tokenizer": _tokenizer_identity(tokenizer_path, chat_template),
        "thinking": {"requested": bool(thinking), "effective": None, "canary": "not-run"},
        "generation": {"model_args": model_args, "gen_kwargs": gen_kwargs},
        "harness": {"name": framework, **_git(harness_dir)},
        "suite": _git(REPO_ROOT),
        "container": {"image": os.environ.get("SUITE_CONTAINER_IMAGE") or None, "size": None},
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME")},
        "results": None,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("APERTUS_", "VLLM_APERTUS_", "IMAGE_TOKEN_CACHE", "PYTORCH_CUDA_ALLOC_CONF"))},
    }
    image = manifest["container"]["image"]
    if image and Path(image).exists():
        manifest["container"].update(file_stat(Path(image)))
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


def finalize(manifest_path: Path, log_paths, results_dir: Path, harness_rc: int,
             thinking_min_tokens: int = THINKING_MIN_MEAN_TOKENS) -> tuple[str, dict]:
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
    s.add_argument("--out", required=True)
    s.add_argument("--framework", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--run-id", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--tokenizer")
    s.add_argument("--chat-template")
    s.add_argument("--model-args", default="")
    s.add_argument("--gen-kwargs", default="")
    s.add_argument("--thinking", action="store_true")
    s.add_argument("--extra-json", default="{}")
    f = sub.add_parser("finalize")
    f.add_argument("--manifest", required=True)
    f.add_argument("--log", action="append", default=[])
    f.add_argument("--results-dir", required=True)
    f.add_argument("--harness-rc", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "start":
        start(Path(a.out), framework=a.framework, task=a.task, run_id=a.run_id, model_path=Path(a.model),
              tokenizer_path=Path(a.tokenizer) if a.tokenizer else None,
              chat_template=Path(a.chat_template) if a.chat_template else None,
              model_args=a.model_args, gen_kwargs=a.gen_kwargs, thinking=a.thinking, extra=json.loads(a.extra_json))
        return 0
    status, manifest = finalize(Path(a.manifest), a.log, Path(a.results_dir), a.harness_rc)
    print(f"run_meta: status={status} error={manifest.get('error')}")
    return {"ok": 0, "failed": EXIT_FAILED, "invalid": EXIT_INVALID}[status]


if __name__ == "__main__":
    sys.exit(main())
