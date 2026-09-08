import json

from suite.coverage import cell_provenance, collect_manifests, coverage, find_manifest
from suite.tasks import Registry, Task


def _reg():
    return Registry(tasks={"gqa": Task("gqa", "lmms-eval", "gqa", card=True),
                           "pope": Task("pope", "lmms-eval", "pope", card=True),
                           "blink": Task("blink", "VLMEvalKit", "BLINK", lmms_task="blink")},
                    dashboard={})


def test_coverage_reasons():
    table = [{"task": "gqa", "metric": "exact_match", "framework": "lmms-eval", "cat": "x", "cells": {"m1": {"v": 59.2, "run": "r"}}}]
    manifests = {("pope", "m1"): {"status": "failed", "error": "OOM"}, ("gqa", "m2"): {"status": "invalid", "error": "no thinking"},
                 ("blink", "m1"): {"status": "running"}}
    cov = coverage(table, ["m1", "m2"], _reg(), manifests)
    reasons = {(m["task"], m["model"]): m["reason"] for m in cov["missing"]}
    assert reasons[("pope", "m1")] == "run-failed:OOM"
    assert reasons[("gqa", "m2")] == "run-invalid:no thinking"
    assert reasons[("pope", "m2")] == "no-run" and reasons[("blink", "m1")] == "running"
    assert cov["counts"] == {"cells": 6, "present": 1, "missing": 5}
    assert cov["by_reason"] == {"no-run": 2, "run-failed": 1, "run-invalid": 1, "running": 1}


def test_find_manifest_walks_up_and_provenance(tmp_path):
    run = tmp_path / "results" / "lmms-eval" / "m" / "r1" / "gqa"
    leaf = run / "textview__m"
    leaf.mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"status": "ok", "run_id": "r1", "thinking": {"effective": True},
                                                   "model": {"config_sha256": "abc"}, "harness": {"commit": "h"}, "container": {"image": "i"}}))
    f = leaf / "x_results.json"
    f.write_text("{}")
    man = find_manifest(f)
    assert man["run_id"] == "r1" and find_manifest(tmp_path / "nowhere.json") is None
    assert cell_provenance(man) == {"run_id": "r1", "status": "ok", "thinking": True, "model_sha": "abc", "harness": "h", "image": "i"}
    assert cell_provenance(None) is None


def test_collect_manifests_both_layouts(tmp_path):
    lm = tmp_path / "lmms-eval" / "8B-Final" / "r1" / "gqa"
    lm.mkdir(parents=True)
    (lm / "run_meta.json").write_text(json.dumps({"framework": "lmms-eval", "task": "gqa", "status": "ok", "started_at": 5}))
    old = tmp_path / "lmms-eval" / "8B-Final" / "r0" / "gqa"
    old.mkdir(parents=True)
    (old / "run_meta.json").write_text(json.dumps({"framework": "lmms-eval", "task": "gqa", "status": "failed", "started_at": 1}))
    vk = tmp_path / "VLMEvalKit" / "r1" / "8B-Final" / "BLINK"
    vk.mkdir(parents=True)
    (vk / "run_meta.json").write_text(json.dumps({"framework": "VLMEvalKit", "task": "BLINK", "status": "failed", "error": "ctx"}))
    got = collect_manifests([tmp_path / "lmms-eval", tmp_path / "VLMEvalKit"], _reg(), canonical_key=str.lower)
    assert got[("gqa", "8b-final")]["status"] == "ok"
    assert got[("blink", "8b-final")]["error"] == "ctx"
