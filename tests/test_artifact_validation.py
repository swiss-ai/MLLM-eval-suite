"""Artifact validation must retain eligibility and detect changing content."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_dashboard as dashboard
import suite.coverage as coverage_module
from suite.coverage import Manifests
from suite.result_selection import ineligible_reason


def _result(tmp_path, tasks=1):
    directory = tmp_path / "model" / "run" / "task"
    directory.mkdir(parents=True)
    path = directory / "scores_results.json"
    data = {"results": {f"cache_task_{i:02}": {"exact_match,none": i / 100}
                        for i in range(tasks)}, "config": {"limit": None}}
    path.write_text(json.dumps(data))
    manifest = {"status": "ok", "framework": "lmms-eval", "run_id": "run",
                "thinking": {"effective": True}, "model": {"config_sha256": "model-sha"},
                "harness": {"commit": "harness-sha"}, "container": {"image": "image"},
                "results": {"artifacts": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}}}
    (directory / "run_meta.json").write_text(json.dumps(manifest))
    return path, manifest


@pytest.fixture
def hash_reads(monkeypatch):
    """Count full file reads while keeping real SHA256 computation."""
    reads = []
    real_hash = coverage_module.sha256_file

    def counted(path):
        reads.append(path)
        return real_hash(path)

    monkeypatch.setattr(coverage_module, "sha256_file", counted)
    return reads


def test_twenty_task_collection_preserves_cells_and_provenance(tmp_path):
    path, manifest = _result(tmp_path, tasks=20)
    manifests = Manifests([tmp_path])
    expected = [{"task": f"cache_task_{i:02}", "metric": "exact_match", "framework": "lmms-eval",
                 "cells": {"model": {"v": float(i), "raw": i / 100, "run": "run",
                    "source": {"path": str(path), "mtime_ns": path.stat().st_mtime_ns},
                    "prov": {"run_id": "run", "status": "ok", "thinking": True,
                             "model_sha": "model-sha", "harness": "harness-sha", "image": "image"}}}}
                for i in range(20)]

    assert dashboard.collect(tmp_path, None, manifests) == (["model"], expected)
    assert dashboard.collect(tmp_path, None, manifests) == (["model"], expected)
    assert manifests.for_result(path) == manifest
    assert manifests.by_dir[path.parent] == manifest


def _mutate(path, mutation):
    before = path.stat()
    contents = path.read_bytes()
    changed = contents.replace(b"0.0", b"0.9")
    assert changed != contents and len(changed) == len(contents)
    if mutation == "rewrite":
        path.write_bytes(changed + b" ")
    elif mutation == "same-size-restored-mtime":
        # Linux filesystem timestamps may share a clock tick across quick writes.
        time.sleep(0.01)
        path.write_bytes(changed)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
        assert after.st_ctime_ns != before.st_ctime_ns
    elif mutation == "replace":
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(changed)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
        assert path.stat().st_ino != before.st_ino
    elif mutation == "remove":
        path.unlink()
    elif mutation == "directory":
        path.unlink()
        path.mkdir()
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize("mutation", ["rewrite", "same-size-restored-mtime", "replace", "remove", "directory"])
def test_mutation_invalidates_existing_inventory_and_publication(tmp_path, mutation):
    path, original = _result(tmp_path)
    manifests = Manifests([tmp_path])
    assert dashboard.collect(tmp_path, None, manifests)[1]
    _mutate(path, mutation)

    rejected = manifests.for_result(path)
    assert rejected["status"] == "invalid"
    assert rejected["results"] == original["results"]
    assert manifests.by_dir[path.parent] == original
    assert dashboard.collect(tmp_path, None, manifests)[1] == []


def test_symlink_retarget_rechecks_resolved_artifact(tmp_path, hash_reads):
    path, _ = _result(tmp_path)
    other = path.with_name("unattested.json")
    other.write_bytes(path.read_bytes())
    link = tmp_path / "selected.json"
    link.symlink_to(path)
    manifests = Manifests([tmp_path])
    assert manifests.for_result(link)["status"] == "ok"
    assert hash_reads == [path]
    link.unlink()
    link.symlink_to(other)
    assert manifests.for_result(link)["status"] == "invalid"
    link.unlink()
    link.symlink_to(path)
    assert manifests.for_result(link)["status"] == "ok"
    assert hash_reads == [path, path]


def test_changed_expected_digest_does_not_reuse_validity(tmp_path, hash_reads):
    path, _ = _result(tmp_path)
    manifests = Manifests([tmp_path])
    original = manifests.for_result(path)
    artifacts = original["results"]["artifacts"]
    digest = artifacts[str(path)]
    # Returned manifests/results remain mutable dictionaries, including malformed values.
    for expected in ["wrong-digest", [], None]:
        artifacts[str(path)] = expected
        rejected = manifests.for_result(path)
        assert rejected["status"] == "invalid"
        assert rejected["results"] is original["results"]
        assert original["status"] == "ok"
    artifacts[str(path)] = digest
    assert manifests.for_result(path) is original
    assert hash_reads == [path] * 5


@pytest.mark.parametrize("mutation", ["same-size-restored-mtime", "replace", "remove"])
def test_changes_during_hash_are_rejected_and_can_recover(tmp_path, monkeypatch, mutation):
    path, original = _result(tmp_path)
    contents = path.read_bytes()
    manifests = Manifests([tmp_path])
    real_hash = coverage_module.sha256_file
    reads = []

    def changing_hash(artifact):
        reads.append(artifact)
        digest = real_hash(artifact)
        if len(reads) == 1:
            _mutate(artifact, mutation)
        return digest

    monkeypatch.setattr(coverage_module, "sha256_file", changing_hash)
    assert manifests.for_result(path)["status"] == "invalid"
    if mutation != "remove":
        assert manifests.for_result(path)["status"] == "invalid"
        assert reads == [path, path]
    path.write_bytes(contents)
    assert manifests.for_result(path) == original
    assert manifests.for_result(path) == original


def test_rapid_same_size_rewrites_are_rehashed_with_mtime_restored(tmp_path, hash_reads):
    path, original = _result(tmp_path)
    contents = path.read_bytes()
    before = path.stat()
    manifests = Manifests([tmp_path])
    assert manifests.for_result(path) == original
    path.write_bytes(contents.replace(b"0.0", b"0.9"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    # ctime can also remain identical within one filesystem clock tick.
    assert manifests.for_result(path)["status"] == "invalid"
    path.write_bytes(contents)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert manifests.for_result(path) == original
    assert hash_reads == [path] * 3


@pytest.mark.parametrize("changes,status,reason", [
    ({"status": "failed"}, "failed", "run-failed"),
    ({"status": "running"}, "running", "run-running"),
    ({"status": "invalid"}, "invalid", "run-invalid"),
    ({"results": []}, "ok", "invalid-manifest"),
    ({"results": {"artifacts": []}}, "invalid", "run-invalid"),
    ({"results": {"artifacts": {}}}, "invalid", "run-invalid"),
    ({"results": {"file": "other.json"}}, "invalid", "run-invalid"),
    ({"results": {}}, "ok", None),
    ({"results": None}, "ok", None),
])
def test_unattested_manifest_semantics_do_not_require_hashes(tmp_path, hash_reads, changes, status, reason):
    path, manifest = _result(tmp_path)
    manifest.update(changes)
    (path.parent / "run_meta.json").write_text(json.dumps(manifest))
    manifests = Manifests([tmp_path])
    for _ in range(2):
        found = manifests.for_result(path)
        assert found["status"] == status
        assert ineligible_reason(found) == reason
    assert hash_reads == []


@pytest.mark.parametrize("contents", ["not JSON", "[]", "null"])
def test_malformed_manifests_remain_invalid(tmp_path, hash_reads, contents):
    path, _ = _result(tmp_path)
    (path.parent / "run_meta.json").write_text(contents)
    manifests = Manifests([tmp_path])
    assert manifests.for_result(path)["status"] == "invalid"
    assert dashboard.collect(tmp_path, None, manifests)[1] == []
    assert hash_reads == []


def test_legacy_headline_and_manifestless_results_keep_eligibility(tmp_path, hash_reads):
    path, manifest = _result(tmp_path)
    manifest["results"] = {"file": str(path)}
    meta = path.parent / "run_meta.json"
    meta.write_text(json.dumps(manifest))
    manifests = Manifests([tmp_path])
    assert manifests.for_result(path) == manifest
    _mutate(path, "same-size-restored-mtime")
    assert manifests.for_result(path) == manifest
    assert dashboard.collect(tmp_path, None, manifests)[1][0]["cells"]["model"]["v"] == 90
    meta.unlink()
    manifests = Manifests([tmp_path])
    assert manifests.for_result(path) is None
    assert dashboard.collect(tmp_path, None, manifests)[1][0]["cells"]["model"]["legacy"] is True
    assert hash_reads == []
