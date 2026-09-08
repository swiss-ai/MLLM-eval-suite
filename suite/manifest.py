"""Run manifests: what a job actually ran with, and how it ended."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from suite import REPO_ROOT
from suite.fsutil import file_stat, sha256_file
from suite.result_selection import limited
from suite.result_selection import declared_chat_template
from suite.tasks import load_registry

SCHEMA = 1
EXIT_FAILED = 3
EXIT_INVALID = 4
EXIT_BY_STATUS = {"ok": 0, "failed": EXIT_FAILED, "invalid": EXIT_INVALID}
THINKING_MIN_MEAN_TOKENS = 8
GENERATION_FIELDS = ("tp", "dp", "batch_size", "gpu_memory_utilization", "max_model_len", "limit", "apply_chat_template")
# A fatal pattern means generation was lost or invalid part way through, so a
# results file written afterwards can be partial; the run is failed whatever the
# exit code says. Advisory patterns are recorded as warnings on an ok run.
FATAL_PATTERNS = (
    r"(torch\.OutOfMemoryError: .{0,200})",
    r"(EngineDeadError.{0,200})",
    r"(RuntimeError: cancelled)",
    r"(thinking canary failed.{0,200})",
    r"(Engine core initialization failed.{0,100})",
    r"(Worker proc \S+ died unexpectedly.{0,80})",
)
ADVISORY_PATTERNS = (
    r"Error during evaluation: (.{0,300})",
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


def _container_identity() -> dict:
    image = os.environ.get("SUITE_CONTAINER_IMAGE") or None
    out = {"image": image, "size": None}
    if image and Path(image).exists():
        out.update(file_stat(Path(image)))
    return out


def _artifact_signature(path: Path) -> dict | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def _result_paths(results_dir: Path, framework: str) -> tuple[list[Path], list[Path]]:
    """All framework result and sample paths, sorted for deterministic snapshots."""
    results_dir = Path(results_dir)
    if framework == "VLMEvalKit":
        return sorted([*results_dir.rglob("*_acc.csv"), *results_dir.rglob("*_score.csv")]), []
    if framework == "lm-eval":
        return sorted(results_dir.rglob("results_*.json")), sorted(results_dir.rglob("samples_*.jsonl"))
    return sorted(results_dir.rglob("*_results.json")), sorted(results_dir.rglob("*samples_*.jsonl"))


def _artifact_snapshot(results_dir: Path, framework: str) -> dict[str, dict]:
    result_paths, sample_paths = _result_paths(results_dir, framework)
    paths = [*result_paths, *sample_paths]
    return {str(path.resolve()): sig for path in paths if (sig := _artifact_signature(path)) is not None}


def start(out_dir: Path, *, framework: str, task: str, run_id: str, model_path: Path, harness_dir: Path,
          tokenizer_path: Path | None, chat_template: Path | None, model_args: str, gen_kwargs: str, thinking: bool,
          generation: dict | None = None) -> dict:
    """Write run_meta.json before inference. harness_dir is the checkout the job imports, worktree or pinned."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_at_start = _artifact_snapshot(out_dir, framework)
    manifest = {
        "schema": SCHEMA, "run_id": run_id, "framework": framework, "task": task,
        "status": "running", "error": None, "started_at": int(time.time()), "finished_at": None,
        "model": {"name": Path(model_path).name, **_model_identity(Path(model_path))},
        "tokenizer": _tokenizer_identity(tokenizer_path, chat_template),
        "thinking": {"requested": bool(thinking), "effective": None, "canary": "not-run"},
        "generation": {"model_args": model_args, "gen_kwargs": gen_kwargs, **(generation or {})},
        "harness": {"name": framework, **_git(Path(harness_dir))},
        "suite": _git(REPO_ROOT),
        "container": _container_identity(),
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME")},
        "results": None,
        # A resumed run may reuse an output directory. Finalization accepts a
        # pre-existing artifact only when the harness changed it after this point.
        "artifacts_at_start": artifacts_at_start,
        "warnings": [],
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("APERTUS_", "VLLM_APERTUS_", "IMAGE_TOKEN_CACHE", "PYTORCH_CUDA_ALLOC_CONF"))},
    }
    (out_dir / "run_meta.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _sample_task(path: Path) -> str:
    stem = path.stem
    # lmms-eval: <timestamp>_samples_<task>; lm-eval: samples_<task>_<timestamp>.
    # Split on the format marker, never on a task's final underscore.
    if not stem.startswith("samples_") and "_samples_" in stem:
        return stem.split("_samples_", 1)[1]
    stem = stem.removeprefix("samples_")
    match = re.match(r"(.+)_\d{4}-\d{2}-\d{2}T", stem)
    return match.group(1) if match else re.sub(r"_\d+$", "", stem)


def output_token_stats(sample_files) -> dict | None:
    records: dict[tuple[str, str], int | None] = {}
    ordered = sorted(sample_files, key=lambda path: (((sig := _artifact_signature(Path(path))) or {}).get("mtime_ns", 0), str(path)))
    for path in ordered:
        with open(path) as fh:
            for line_no, line in enumerate(fh):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                task = str(rec.get("task") or _sample_task(Path(path)))
                doc_id = rec.get("doc_id")
                try:
                    doc_key = json.dumps(doc_id, sort_keys=True) if doc_id is not None else f"{Path(path).resolve()}:{line_no}"
                except (TypeError, ValueError):
                    doc_key = f"{Path(path).resolve()}:{line_no}"
                tc = rec.get("token_counts")
                while isinstance(tc, list) and tc:
                    tc = tc[0]
                tokens = tc.get("output_tokens") if isinstance(tc, dict) else None
                if (isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0
                        or isinstance(tokens, float) and math.isfinite(tokens) and tokens >= 0 and tokens.is_integer()):
                    records[(task, doc_key)] = int(tokens)
                else:
                    records[(task, doc_key)] = None
    counts = [count for count in records.values() if count is not None]
    if not counts:
        return None
    return {"n": len(counts), "records": len(records), "mean": statistics.fmean(counts),
            "median": statistics.median(counts), "max": max(counts)}


def sample_record_count(sample_files) -> int | None:
    """Count deduplicated sample records, even if no output-token data was logged."""
    # Reuse the same replacement semantics as token statistics without treating
    # an absent token count as a zero-token generation.
    records: dict[tuple[str, str], None] = {}
    for path in sample_files:
        fallback_task = _sample_task(Path(path))
        try:
            # JSON strings may carry raw U+2028 or U+0085, which splitlines() treats as line ends.
            lines = Path(path).read_text().split("\n")
        except OSError:
            continue
        for line_no, line in enumerate(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            try:
                doc_key = json.dumps(rec.get("doc_id"), sort_keys=True) if rec.get("doc_id") is not None else f"{Path(path).resolve()}:{line_no}"
            except (TypeError, ValueError):
                doc_key = f"{Path(path).resolve()}:{line_no}"
            records[(str(rec.get("task") or fallback_task), doc_key)] = None
    return len(records) if records else None


def _scan_logs(log_paths) -> tuple[str | None, list[str], bool | None, str]:
    """First fatal match, every advisory match, the observed thinking flag, and the canary verdict."""
    fatal, warnings, effective, canary = None, [], None, "not-run"
    for path in log_paths:
        try:
            text = Path(path).read_text(errors="ignore")
        except OSError:
            continue
        for pat in FATAL_PATTERNS:
            m = re.search(pat, text)
            if m and fatal is None:
                fatal = m.group(1).strip()
        for pat in ADVISORY_PATTERNS:
            warnings += [m.strip() for m in re.findall(pat, text)]
        m = FLAG_RE.search(text)
        if m:
            effective = {"True": True, "False": False}.get(m.group(1))
        m = CANARY_RE.search(text)
        if m:
            canary = m.group(1)
    return fatal, warnings, effective, canary


def _newest(paths) -> Path | None:
    stamped = []
    for p in paths:
        try:
            stamped.append((p.stat().st_mtime, p))
        except OSError:
            continue
    return max(stamped)[1] if stamped else None


def _find_results(results_dir: Path, framework: str) -> tuple[Path | None, list[Path]]:
    """The newest headline result file for the framework's layout, plus lmms-eval sample logs."""
    result_paths, sample_paths = _result_paths(results_dir, framework)
    return _newest(result_paths), sample_paths


def _is_fresh(path: Path, manifest: dict) -> bool:
    signature = _artifact_signature(path)
    if signature is None:
        return False
    before = (manifest.get("artifacts_at_start") or {}).get(str(path.resolve()))
    if before is not None:
        return signature != before
    # Pre-hardening manifests have no snapshot. The timestamp makes their old
    # result files fail closed without rejecting a result written during this run.
    started_at = manifest.get("started_at")
    return not isinstance(started_at, int) or signature["mtime_ns"] > started_at * 1_000_000_000


def _task_ids(manifest: dict) -> set[str]:
    task = str(manifest["task"])
    ids = {task}
    try:
        registered = load_registry().lookup(manifest["framework"], task)
    except (OSError, ValueError):
        registered = None
    if registered:
        ids.add(registered.name)
        if harness_id := registered.harness_id_for(manifest["framework"]):
            ids.add(harness_id)
    return ids


def _sample_counts(value) -> tuple[int | None, int | None]:
    def count(number):
        if number is None:
            return None
        if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
            return number
        if isinstance(number, float) and math.isfinite(number) and number >= 0 and number.is_integer():
            return int(number)
        raise ValueError(f"invalid sample count {number!r}")

    if not isinstance(value, dict):
        return count(value), None
    counts = {key: count(value.get(key)) for key in ("effective", "sample_len", "original")}
    effective = next((counts[key] for key in ("effective", "sample_len", "original") if counts[key] is not None), None)
    original = counts["original"]
    if effective is not None and original is not None and effective > original:
        raise ValueError(f"effective sample count {effective} exceeds original {original}")
    return effective, original


def _sample_evidence(sample_map: dict, names: list[str]) -> dict:
    counts = [_sample_counts(sample_map.get(name)) for name in names]
    effective = sum(count[0] for count in counts) if counts and all(count[0] is not None for count in counts) else None
    original = sum(count[1] for count in counts) if counts and all(count[1] is not None for count in counts) else None
    return {"effective": effective, "original": original,
            "partial": any(e is not None and o is not None and e < o for e, o in counts)}


def _has_numeric_metrics(record: dict) -> bool:
    metadata = {"name", "alias", "sample_len", "sample_count", "samples", "n_samples", "count", "n"}
    return any(key.split(",", 1)[0] not in metadata and not any(part in key.lower() for part in ("stderr", "_clt", "_clustered"))
               and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
               for key, value in record.items())


def _validate_json_result(path: Path, manifest: dict) -> tuple[str | None, dict | None]:
    """Validate lm-eval/lmms-eval structure and return a known effective count."""
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return f"result file {path} is not valid JSON", None
    if not isinstance(data, dict) or not isinstance(data.get("results"), dict) or not data["results"]:
        return f"result file {path} has no non-empty results mapping", None
    results = data["results"]
    expected = _task_ids(manifest)
    if manifest.get("framework") == "lm-eval":
        for task_id in sorted(expected):
            declared = declared_chat_template(task_id)
            if declared is not None and bool(data.get("chat_template")) != declared:
                return (f"prompting protocol mismatch: the registry declares chat_template={declared} for {task_id}, "
                        f"the run used {bool(data.get('chat_template'))}"), None
    group_subtasks = data.get("group_subtasks") if isinstance(data.get("group_subtasks"), dict) else {}
    sample_map = data.get("n-samples") if isinstance(data.get("n-samples"), dict) else {}
    registered = load_registry().lookup(manifest["framework"], manifest["task"])
    # Harness tags expand into independent tasks and have no group metadata.
    # Their registry contract lists every required result, so one successful
    # component cannot validate an incomplete tag run.
    if registered and registered.result_tasks:
        for name in registered.result_tasks:
            if not isinstance(results.get(name), dict) or not _has_numeric_metrics(results[name]):
                return f"result tag {manifest['task']} is missing numeric metrics for task {name}", None
        try:
            return None, _sample_evidence(sample_map, list(registered.result_tasks))
        except ValueError as exc:
            return str(exc), None
    # A group result often also has an aggregate entry in results. When the
    # harness declares group members, validate those members rather than taking
    # the aggregate as proof that every requested task ran.
    for group in sorted(expected):
        subtasks = group_subtasks.get(group)
        if not isinstance(subtasks, list) or not subtasks:
            continue
        leaves = set()

        def visit(name, ancestors):
            if not isinstance(name, str):
                raise ValueError(f"result group {group} has an invalid subtask name")
            if name in ancestors:
                raise ValueError(f"result group {group} has a subtask cycle at {name}")
            children = group_subtasks.get(name, [])
            if not isinstance(children, list):
                raise ValueError(f"result group {name} has invalid subtask metadata")
            if children:
                for child in children:
                    visit(child, ancestors | {name})
            elif not isinstance(results.get(name), dict):
                raise ValueError(f"result group {group} is missing one or more declared subtasks ({name})")
            elif not _has_numeric_metrics(results[name]):
                raise ValueError(f"result group {group} has no numeric metrics for task {name}")
            else:
                leaves.add(name)

        try:
            # Groups may be organizational only, with no aggregate score. Their
            # declared leaves are the evidence, including nested subgroups.
            visit(group, set())
            return None, _sample_evidence(sample_map, sorted(leaves))
        except ValueError as exc:
            return str(exc), None
    direct = next((name for name in sorted(expected) if name in results and isinstance(results[name], dict)), None)
    if direct:
        if not _has_numeric_metrics(results[direct]):
            return f"result task {direct} has no numeric metrics", None
        try:
            return None, _sample_evidence(sample_map, [direct])
        except ValueError as exc:
            return str(exc), None
    return f"result file {path} does not contain requested task {manifest['task']}", None


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _csv_metric_name(value: object) -> bool:
    if not isinstance(value, str):
        return False
    name = _normalized(value)
    metadata = {"", "category", "dataset", "task", "benchmark", "suite", "metric", "split", "subtask", "group", "type",
                "samples", "sample", "count", "n", "num", "samplecount", "nsamples", "samplelen", "original", "effective",
                "index", "id", "row", "rowid"}
    return name not in metadata and not name.startswith("unnamed") and "stderr" not in name


def _csv_filename_matches(path: Path, task_ids: set[str]) -> bool:
    def key(value):
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    # Match the dataset within the basename, allowing both model prefixes and
    # arbitrary judge suffixes. A task-like model prefix must not override the
    # actual dataset later in the filename; overlapping IDs prefer the longest.
    known_ids = set(task_ids)
    try:
        for task in load_registry().tasks.values():
            known_ids.update(name for name in (task.name, task.harness_task, task.lmms_task) if name)
    except (OSError, ValueError):
        pass
    stem = key(re.sub(r"_(?:acc|score)$", "", path.stem, flags=re.I))
    matches = [(match.end(), len(name), name)
               for name in {key(task) for task in known_ids}
               for match in re.finditer(r"(?:^|_)" + re.escape(name) + r"(?=_|$)", stem)]
    return bool(matches) and max(matches)[2] in {key(task) for task in task_ids}


def _validate_vlmeval_csv(path: Path, manifest: dict) -> str | None:
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        return f"result file {path} is not parseable CSV"
    if not reader.fieldnames or not rows:
        return f"result file {path} has no CSV rows"
    metric_columns = [field for field in reader.fieldnames if _csv_metric_name(field)]
    metric_labels = [field for field in reader.fieldnames if _normalized(field) == "metric"]
    numeric = any(
        isinstance(row.get(field), str) and row[field].strip() and _is_finite_number(row[field])
        and (not metric_labels or any(_csv_metric_name(row.get(label, "")) for label in metric_labels))
        for row in rows for field in metric_columns
    )
    if not numeric:
        return f"result file {path} has no numeric CSV metrics"
    task_ids = _task_ids(manifest)
    expected = {_normalized(name) for name in task_ids}
    metadata_values = [_normalized(value) for row in rows for field, value in row.items()
                       if value and _normalized(field) in {"dataset", "task", "benchmark", "suite"}]
    if any(value not in expected for value in metadata_values):
        return f"result file {path} has metadata contradicting requested task {manifest['task']}"
    filename_matches = _csv_filename_matches(path, task_ids)
    generic_filename = path.stem.lower() in {"derived_acc", "derived_score", "results_acc", "results_score"}
    if not filename_matches and not (generic_filename and metadata_values):
        return f"result file {path} does not identify requested task {manifest['task']}"
    return None


def _is_finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def finalize(manifest_path: Path, log_paths, results_dir: Path, harness_rc: int,
             thinking_min_tokens: int = THINKING_MIN_MEAN_TOKENS) -> tuple[str, dict]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    error, warnings, effective, canary = _scan_logs(log_paths)
    manifest["thinking"]["effective"] = effective
    manifest["thinking"]["canary"] = canary
    manifest["warnings"] = warnings
    result_file, samples = _find_results(results_dir, manifest["framework"])
    status = "ok"
    if error or harness_rc != 0 or result_file is None:
        status = "failed"
        error = error or (f"harness exit code {harness_rc}" if harness_rc else "no results file produced")
    else:
        all_results, all_samples = _result_paths(results_dir, manifest["framework"])
        fresh_results = [path for path in all_results if _is_fresh(path, manifest)]
        fresh_samples = [path for path in all_samples if _is_fresh(path, manifest)]
        if not fresh_results:
            status, error = "invalid", "all result files predate this run"
        else:
            result_file = _newest(fresh_results)
            result_error, evidence = (None, {"effective": None, "original": None, "partial": False})
            if manifest["framework"] == "VLMEvalKit":
                result_error = _validate_vlmeval_csv(result_file, manifest)
            else:
                result_error, evidence = _validate_json_result(result_file, manifest)
            stats = output_token_stats(fresh_samples) if fresh_samples else None
            sample_records = sample_record_count(fresh_samples) if fresh_samples else None
            result_samples = evidence["effective"] if evidence else None
            manifest["results"] = {"file": str(result_file), "n_samples": sample_records,
                                   "result_samples": result_samples,
                                   "result_original_samples": evidence["original"] if evidence else None,
                                   "partial": bool(evidence and evidence["partial"]), "output_tokens": stats}
            if result_error:
                status, error = "invalid", result_error
            elif result_samples == 0:
                status, error = "invalid", "result reports zero effective samples"
            elif result_samples is not None and sample_records is not None and result_samples != sample_records:
                status, error = "invalid", f"result sample count {result_samples} disagrees with {sample_records} sample records"
            elif evidence and evidence["partial"] and not limited(manifest.get("generation", {}).get("limit")):
                status, error = "invalid", "result metadata is partial without an explicit limit"
            elif manifest["thinking"]["requested"]:
                if effective is not True or canary != "passed":
                    status, error = "invalid", "thinking requested but effective evidence or canary pass is missing"
                elif stats is not None and stats["mean"] < thinking_min_tokens:
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
    s.add_argument("--harness-dir", required=True, help="the harness checkout this job imports")
    s.add_argument("--tokenizer")
    s.add_argument("--chat-template")
    s.add_argument("--model-args", default="")
    s.add_argument("--gen-kwargs", default="")
    s.add_argument("--thinking", action="store_true")
    for name in GENERATION_FIELDS:
        s.add_argument(f"--{name.replace('_', '-')}")
    f = sub.add_parser("finalize")
    f.add_argument("--manifest", required=True)
    f.add_argument("--log", action="append", default=[])
    f.add_argument("--results-dir", required=True)
    f.add_argument("--harness-rc", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "start":
        generation = {name: _number(getattr(a, name)) for name in GENERATION_FIELDS if getattr(a, name) not in (None, "")}
        start(Path(a.out), framework=a.framework, task=a.task, run_id=a.run_id, model_path=Path(a.model),
              harness_dir=Path(a.harness_dir), tokenizer_path=Path(a.tokenizer) if a.tokenizer else None,
              chat_template=Path(a.chat_template) if a.chat_template else None,
              model_args=a.model_args, gen_kwargs=a.gen_kwargs, thinking=a.thinking, generation=generation)
        return 0
    status, manifest = finalize(Path(a.manifest), a.log, Path(a.results_dir), a.harness_rc)
    print(f"run_meta: status={status} error={manifest.get('error')}")
    return EXIT_BY_STATUS[status]


def _number(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


if __name__ == "__main__":
    sys.exit(main())
