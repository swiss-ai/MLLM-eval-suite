"""Which dashboard cells are missing, and why."""
from __future__ import annotations

import json
from pathlib import Path

from suite.tasks import Registry


class Manifests:
    """Every run_meta.json under the results roots, read once.

    lmms-eval and lm-eval manifests sit at <root>/<model>/<run>/<task>/run_meta.json
    and VLMEvalKit ones at <root>/<run>/<model>/<dataset>/run_meta.json; the
    framework field tells the layouts apart.
    """

    def __init__(self, roots):
        self.entries: list[tuple[Path, Path, dict]] = []
        self.by_dir: dict[Path, dict] = {}
        for root in map(Path, roots):
            if not root.is_dir():
                continue
            for path in root.rglob("run_meta.json"):
                try:
                    man = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                self.entries.append((root, path, man))
                self.by_dir[path.parent] = man

    def for_result(self, path: Path) -> dict | None:
        """The manifest of the run that produced a results file, or None."""
        for parent in Path(path).parents:
            if parent in self.by_dir:
                return self.by_dir[parent]
        return None

    def by_cell(self, registry: Registry, canonical_key) -> dict[tuple[str, str], dict]:
        """Newest manifest per (dashboard task, model key)."""
        out: dict[tuple[str, str], dict] = {}
        for root, path, man in self.entries:
            rel = path.relative_to(root).parts
            if len(rel) < 3:
                continue
            if man.get("framework") == "VLMEvalKit":
                model_dir, harness_id = rel[1], man.get("task") or rel[2]
                task = registry.resolve("VLMEvalKit", harness_id)
                task_name = task.name if task else harness_id.lower()
            else:
                model_dir, task_name = rel[0], man.get("task") or rel[2]
            key = (task_name, canonical_key(model_dir))
            prev = out.get(key)
            if prev is None or (man.get("started_at") or 0) >= (prev.get("started_at") or 0):
                out[key] = man
        return out


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
