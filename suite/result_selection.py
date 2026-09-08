"""Shared publication eligibility and deterministic artifact selection."""
from __future__ import annotations

import math
from pathlib import Path


def limited(value) -> bool:
    if value in (None, "", False):
        return False
    try:
        return float(value) != 0
    except (TypeError, ValueError):
        return True


def ineligible_reason(manifest: dict | None, data: dict | None = None, task: str | None = None) -> str | None:
    if manifest is not None:
        if manifest.get("status") != "ok":
            return f"run-{manifest.get('status', 'invalid')}"
        for field in ("generation", "thinking", "model", "harness", "container", "results"):
            if manifest.get(field) is not None and not isinstance(manifest[field], dict):
                return "invalid-manifest"
        if limited((manifest.get("generation") or {}).get("limit")):
            return "limited-run"
        if (manifest.get("results") or {}).get("partial"):
            return "incomplete-results"
    if data is not None:
        if not isinstance(data, dict):
            return "invalid-results"
        for field in ("config", "n-samples"):
            if data.get(field) is not None and not isinstance(data[field], dict):
                return "invalid-results"
        if limited((data.get("config") or {}).get("limit")):
            return "limited-run"
        counts = (data.get("n-samples") or {}).get(task)
        if counts is None:
            return None
        if isinstance(counts, dict):
            original = counts.get("original")
            effective = counts.get("effective", counts.get("sample_len", original))
        else:
            original, effective = None, counts
        for count in (original, effective):
            if count is not None and (isinstance(count, bool) or not isinstance(count, (int, float))
                                      or not math.isfinite(count) or count <= 0 or count % 1 != 0):
                return "incomplete-results"
        if original is not None and effective is not None and effective != original:
            return "incomplete-results"
    return None


def artifact_metadata(path: Path, manifest: dict | None) -> dict:
    path = path.resolve()
    metadata = {"source": {"path": str(path), "mtime_ns": path.stat().st_mtime_ns}}
    if manifest is None:
        metadata["legacy"] = True
    return metadata


def cell_order(cell: dict) -> tuple:
    source = cell.get("source") or {}
    return source.get("mtime_ns", -1), source.get("path", "")


def merge_cells(target: dict, incoming: dict) -> None:
    """Newest artifact wins independent of traversal, root, or alias order."""
    for model, cell in incoming.items():
        if model not in target or cell_order(cell) > cell_order(target[model]):
            target[model] = cell
