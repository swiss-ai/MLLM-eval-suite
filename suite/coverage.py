"""Which dashboard cells are missing, and why."""
from __future__ import annotations

import json
from pathlib import Path

from suite.tasks import Registry
from suite.fsutil import sha256_file
from suite.result_selection import ineligible_reason


def _artifact_signature(path: Path) -> tuple[int, ...]:
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns


class Manifests:
    """Every run_meta.json under the results roots, read once.

    lmms-eval and lm-eval manifests sit at <root>/<model>/<run>/<task>/run_meta.json
    and VLMEvalKit ones at <root>/<run>/<model>/<dataset>/run_meta.json; the
    framework field tells the layouts apart.
    """

    def __init__(self, roots):
        self.entries: list[tuple[Path, Path, dict]] = []
        self.by_dir: dict[Path, dict] = {}
        self.rejected_results: set[tuple[str, str, str, str]] = set()
        for root in sorted({Path(root).resolve() for root in roots}):
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("run_meta.json")):
                try:
                    man = json.loads(path.read_text())
                    if not isinstance(man, dict):
                        raise ValueError("manifest must be an object")
                except (OSError, ValueError):
                    man = {"status": "invalid", "error": "unreadable or malformed manifest"}
                self.entries.append((root, path, man))
                self.by_dir[path.parent.resolve()] = man

    def _valid_artifact(self, path: Path, expected) -> bool:
        """Require the attested content and a stable file across the hash read."""
        try:
            before = _artifact_signature(path)
            return sha256_file(path) == expected and _artifact_signature(path) == before
        except OSError:
            return False

    def for_result(self, path: Path) -> dict | None:
        """The manifest of the run that produced a results file, or None."""
        path = Path(path).resolve()
        for parent in path.parents:
            if parent in self.by_dir:
                manifest = self.by_dir[parent]
                results = manifest.get("results")
                if manifest.get("status") == "ok" and isinstance(results, dict):
                    if "artifacts" in results:
                        artifacts = results["artifacts"]
                        valid = (isinstance(artifacts, dict) and str(path) in artifacts
                                 and self._valid_artifact(path, artifacts[str(path)]))
                    elif results.get("file"):
                        # Earlier manifests attest only their recorded headline file.
                        valid = isinstance(results["file"], str) and path == Path(results["file"]).resolve()
                    else:
                        valid = True
                    if not valid:
                        return {**manifest, "status": "invalid", "error": "artifact was not validated by this run or has changed"}
                return manifest
        return None

    def by_cell(self, registry: Registry, canonical_key) -> dict[tuple[str, str], dict]:
        """Newest manifest per (dashboard task, model key)."""
        out: dict[tuple[str, str], dict] = {}
        for root, path, man in self.entries:
            rel = path.relative_to(root).parts
            if len(rel) < 3:
                continue
            if man.get("framework") == "VLMEvalKit":
                # Shared output roots omit the outer suite run directory.
                model_dir = rel[0] if len(rel) == 3 else rel[1]
                harness_id = man.get("task") or rel[-2]
                task = registry.resolve("VLMEvalKit", harness_id)
                task_name = task.name if task else harness_id.lower()
            else:
                model_dir, task_name = rel[0], man.get("task") or rel[2]
                task = registry.resolve(man.get("framework", "lmms-eval"), task_name)
                task_name = task.name if task else task_name
            key = (task_name, canonical_key(model_dir))
            prev = out.get(key)
            if prev is None or (man.get("started_at") or 0) >= (prev.get("started_at") or 0):
                out[key] = man
        return out

    def rejected_runs(self, registry: Registry, canonical_key, aliases: dict) -> set[tuple[str, str, str]]:
        """Known rejected run identities must not reappear through a legacy snapshot."""
        rejected = {(task, aliases.get(model, model), run)
                    for task, model, run, _source in self.rejected_results}
        for root, path, man in self.entries:
            if ineligible_reason(man) is None:
                continue
            rel = path.relative_to(root).parts
            if len(rel) < 4:
                continue
            vk = man.get("framework") == "VLMEvalKit"
            model, run = (rel[1], rel[0]) if vk else (rel[0], rel[1])
            task_id = man.get("task") or rel[2]
            task = registry.resolve(man.get("framework", "lmms-eval"), task_id)
            key = canonical_key(model)
            rejected.add((task.name if task else task_id, aliases.get(key, key), man.get("run_id") or run))
            if vk:
                # Earlier dashboard builds stored the bridge directory name.
                rejected.add((task.name if task else task_id, aliases.get(key, key), f"{rel[2]}__{run}"))
        return rejected

    def reject_result(self, path: Path, task: str, model: str, run: str) -> None:
        """Keep result-level rejection evidence for the later legacy import."""
        self.rejected_results.add((task, model, run, str(path.resolve())))


def cell_provenance(manifest: dict | None) -> dict | None:
    if not manifest:
        return None
    return {"run_id": manifest.get("run_id"), "status": manifest.get("status"),
            "thinking": (manifest.get("thinking") or {}).get("effective"),
            "model_sha": (manifest.get("model") or {}).get("config_sha256"),
            "harness": (manifest.get("harness") or {}).get("commit"),
            "image": (manifest.get("container") or {}).get("image")}


def coverage(table: list[dict], models: list[str], registry: Registry, manifests: dict[tuple[str, str], dict],
             tasks: list[str] | None = None) -> dict:
    present = {(row["task"], m) for row in table for m in row["cells"]}
    task_names = tasks if tasks is not None else list(registry.tasks)
    missing, cells = [], 0
    for task in task_names:
        for model in models:
            cells += 1
            if (task, model) in present:
                continue
            man = manifests.get((task, model))
            if man is None:
                reason = "no-run"
            elif man.get("status") == "failed":
                reason = f"run-failed:{man.get('error')}"
            elif man.get("status") == "invalid":
                reason = f"run-invalid:{man.get('error')}"
            elif man.get("status") == "running":
                reason = "running"
            else:
                reason = "no-parsable-result"
            missing.append({"task": task, "model": model, "reason": reason})
    by_reason: dict[str, int] = {}
    for m in missing:
        head = m["reason"].split(":", 1)[0]
        by_reason[head] = by_reason.get(head, 0) + 1
    return {"missing": missing, "counts": {"cells": cells, "present": cells - len(missing), "missing": len(missing)},
            "by_reason": by_reason}
