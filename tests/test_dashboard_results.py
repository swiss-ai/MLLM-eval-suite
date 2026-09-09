"""Regressions for incorrect published scores, using real collector inputs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_dashboard as dashboard
from suite.coverage import Manifests
from verify_dashboard import audit


@pytest.mark.parametrize("collision", [False, True])
def test_refresh_preserves_dashboard_on_collision(tmp_path, collision):
    runner = tmp_path / "python-probe"
    # Run the real dashboard entrypoints; isolate unrelated log derivation.
    runner.write_text(f"#!{sys.executable}\n" + '''import os, sys
if any(arg.endswith('derive_vlmeval_acc.py') for arg in sys.argv):
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
    runner.chmod(0o755)
    out = tmp_path / "index.html"
    outputs = [out, tmp_path / "dashboard.json", tmp_path / "coverage.json"]
    for path in outputs:
        path.write_text("previous dashboard")
    result(tmp_path / "runs", "one", .6, 100)
    result(tmp_path / "lmms", "two", .9 if collision else .6, 200)
    for root, score, stamp in [(tmp_path / "vk-shared", .2, 100), (tmp_path / "vk" / "run", .7, 200)]:
        benchmark = root / "model" / "BLINK"
        benchmark.mkdir(parents=True)
        vk_path = benchmark / "model_BLINK_acc.csv"
        vk_path.write_text(f"accuracy\n{score}\n")
        os.utime(vk_path, (stamp, stamp))
    models = tmp_path / "models.txt"
    models.write_text("model=Model\n")
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    preserved = bridge / "existing-cache"
    preserved.write_text("untouched")
    env = dict(os.environ, PY=str(runner), OUT=str(out), MODELS_FILE=str(models),
               RUNS_ROOT=str(tmp_path / "runs"), SUITE_LMMS=str(tmp_path / "lmms"),
               VLMEVAL_OUTPUTS=str(tmp_path / "vk-shared"), SUITE_VLMEVAL=str(tmp_path / "vk"),
               BRIDGE=str(tmp_path / "bridge"))
    script = Path(__file__).resolve().parents[1] / "scripts/refresh_dashboard.sh"
    process = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert preserved.read_text() == "untouched"
    if collision:
        assert process.returncode == 1
        assert "COLLISION" in process.stdout
        assert all(path.read_text() == "previous dashboard" for path in outputs)
    else:
        assert process.returncode == 0, process.stderr
        assert "PASS — no contaminating collisions" in process.stdout
        assert 'name="robots"' in out.read_text()
        rows = json.loads(outputs[1].read_text())["table"]
        assert next(row for row in rows if row["task"] == "gqa")["cells"]["model"]["v"] == 60
        blink = next(row for row in rows if row["task"] == "blink")["cells"]["model"]
        assert blink["v"] == 70
        assert blink["source"]["path"] == str(vk_path)


def result(root, run, score, stamp, *, task="gqa", framework="lmms-eval",
           status="ok", limit=None, model="model", counts=None):
    directory = root / model / run / task
    directory.mkdir(parents=True, exist_ok=True)
    text = framework == "lm-eval"
    path = directory / ("results_x.json" if text else "x_results.json")
    metric = "exact_match,flexible-extract" if text else "exact_match,none"
    data = {"results": {task: {metric: score}}, "config": {"limit": limit}}
    if text:
        data["chat_template"] = "{{ messages }}"
    if counts:
        data["n-samples"] = {task: counts}
    path.write_text(json.dumps(data))
    os.utime(path, (stamp, stamp))
    manifest = {"framework": framework, "task": task, "status": status,
                "run_id": run, "generation": {"limit": limit}, "thinking": {},
                "harness": {"commit": "test-commit"}, "model": {}, "container": {}}
    (directory / "run_meta.json").write_text(json.dumps(manifest))
    return path


def build(tmp_path, monkeypatch, roots, *extra):
    out = tmp_path / "build"
    out.mkdir(exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["make_dashboard", "--runs-root", *map(str, roots),
                                    *map(str, extra), "-o", str(out / "index.html")])
    dashboard.main()
    return json.loads((out / "dashboard.json").read_text())


def test_newer_limited_run_keeps_older_full_score(tmp_path):
    result(tmp_path, "full", .6, 100)
    result(tmp_path, "smoke", 1, 200, limit=10)
    _, rows = dashboard.collect(tmp_path, None, Manifests([tmp_path]))
    assert rows[0]["cells"]["model"]["v"] == 60
    assert rows[0]["cells"]["model"]["prov"]["run_id"] == "full"


@pytest.mark.parametrize("status", ["invalid", "failed", "running"])
def test_ineligible_status_cannot_publish(tmp_path, status):
    result(tmp_path, "bad", .99, 100, status=status)
    _, rows = dashboard.collect(tmp_path, None, Manifests([tmp_path]))
    assert rows == []


def test_malformed_manifest_is_not_legacy(tmp_path):
    path = result(tmp_path, "bad", .99, 100)
    (path.parent / "run_meta.json").write_text("not JSON")
    _, rows = dashboard.collect(tmp_path, None, Manifests([tmp_path]))
    assert rows == []


def test_known_partial_sample_counts_cannot_publish(tmp_path):
    result(tmp_path, "partial", .99, 100, counts={"original": 100, "effective": 10})
    _, rows = dashboard.collect(tmp_path, None, Manifests([tmp_path]))
    assert rows == []


def test_manifestless_limited_run_cannot_publish(tmp_path):
    path = result(tmp_path, "smoke", .99, 100, limit=10)
    (path.parent / "run_meta.json").unlink()
    _, rows = dashboard.collect(tmp_path, None, Manifests([tmp_path]))
    assert rows == []


def test_manifestless_full_result_is_explicitly_legacy(tmp_path):
    path = result(tmp_path, "old", .5, 100)
    (path.parent / "run_meta.json").unlink()
    _, rows = dashboard.collect(tmp_path, None, Manifests([tmp_path]))
    assert rows[0]["cells"]["model"]["legacy"] is True


def test_cli_indexes_text_manifests_and_keeps_full_provenance(tmp_path, monkeypatch):
    root = tmp_path / "text"
    empty = tmp_path / "empty"
    empty.mkdir()
    result(root, "full", .6, 100, task="gsm8k", framework="lm-eval")
    result(root, "smoke", 1, 200, task="gsm8k", framework="lm-eval", limit=10)
    data = build(tmp_path, monkeypatch, [empty], "--lm-eval-root", root)
    cell = data["table"][0]["cells"]["model"]
    assert cell["v"] == 60
    assert cell["prov"]["run_id"] == "full"


def test_root_order_cannot_change_score(tmp_path, monkeypatch):
    newer, older = tmp_path / "newer", tmp_path / "older"
    result(newer, "new", .9, 200)
    result(older, "old", .2, 100)
    for roots in ([newer, older], [older, newer]):
        data = build(tmp_path, monkeypatch, roots)
        assert data["table"][0]["cells"]["model"]["v"] == 90


def test_alias_order_cannot_change_score(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    result(root, "new", .9, 200, model="z_alias")
    result(root, "old", .2, 100, model="a_main")
    models = tmp_path / "models.txt"
    models.write_text("a_main|z_alias=Model\n")
    data = build(tmp_path, monkeypatch, [root], "--models-file", models)
    assert data["table"][0]["cells"]["a_main"]["v"] == 90


def test_verifier_detects_same_basename_across_roots(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    result(a, "a", .9, 200)
    result(b, "b", .2, 100)
    assert audit([a, b]) == 1


def test_verifier_ignores_ineligible_conflicting_run(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    result(a, "a", .9, 200)
    result(b, "b", .2, 100, status="invalid")
    assert audit([a, b]) == 0


def test_provenance_coverage_distinguishes_legacy(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    result(root, "new", .9, 200)
    old = result(root, "old", .2, 100, model="legacy-model")
    (old.parent / "run_meta.json").unlink()
    data = build(tmp_path, monkeypatch, [root])
    assert data["evidence"] == {"manifest": 1, "legacy": 1}


@pytest.mark.parametrize("legacy_run, retained", [("bad", False), ("old-full", True)])
def test_legacy_import_cannot_resurrect_known_invalid_run(tmp_path, monkeypatch, legacy_run, retained):
    root = tmp_path / "runs"
    result(root, "bad", .9, 200, status="invalid")
    old = tmp_path / "legacy.json"
    old.write_text(json.dumps({"table": [{"task": "gqa", "metric": "exact_match", "framework": "lmms-eval",
        "cells": {"model": {"v": 90, "raw": .9, "run": legacy_run}}}]}))
    data = build(tmp_path, monkeypatch, [root], "--legacy-json", old)
    assert bool(data["table"]) is retained


def test_symlinked_vlmeval_results_keep_manifest_eligibility(tmp_path):
    raw, bridge = tmp_path / "raw", tmp_path / "bridge"
    for run, value, limit in [("old", .6, None), ("new", .9, 10)]:
        directory = raw / run / "model" / "BLINK"
        directory.mkdir(parents=True)
        path = directory / "model_BLINK_acc.csv"
        path.write_text(f"accuracy\n{value}\n")
        os.utime(path, (100 if run == "old" else 200,) * 2)
        (directory / "run_meta.json").write_text(json.dumps({"status": "ok", "run_id": run,
            "framework": "VLMEvalKit", "task": "BLINK", "generation": {"limit": limit}}))
        link = bridge / "model" / f"BLINK__{run}"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(directory, target_is_directory=True)
    _, rows = dashboard.collect_vlmeval(bridge, None, Manifests([raw]))
    cell = rows[0]["cells"]["model"]
    assert cell["v"] == 60
    assert cell["prov"]["run_id"] == "old"


@pytest.mark.parametrize("content, expected", [
    ("accuracy (%)\n0.5\n", .5),
    ("accuracy\n0.5%\n", .5),
    ("metric,value\noverall (%),0.5\n", .5),
    ("accuracy\n0.5\n", 50),
    ("accuracy\nnan\n", None),
    ("accuracy\ninf\n", None),
])
def test_csv_score_units_and_nonfinite_values(tmp_path, content, expected):
    path = tmp_path / "score.csv"
    path.write_text(content)
    assert dashboard.parse_vk_acc(path) == expected


def test_valid_zero_refspatial_result_is_visible(tmp_path):
    directory = tmp_path / "model" / "RefSpatial_wo_unseen"
    directory.mkdir(parents=True)
    (directory / "model_RefSpatial_wo_unseen_acc.csv").write_text("accuracy\n0\n")
    _, rows = dashboard.collect_vlmeval(tmp_path, None, Manifests([]))
    assert rows[0]["task"] == "refspatial"
    assert rows[0]["cells"]["model"]["v"] == 0


def test_judge_derivation_preserves_zero_and_rejects_ten_percent_failures(tmp_path, monkeypatch):
    import derive_vlmeval_acc as derive
    logs, results = tmp_path / "logs", tmp_path / "results"
    for run, failures, value in [("zero", 0, 0), ("broken", 10, 75)]:
        (logs / run).mkdir(parents=True)
        (logs / run / "vlmeval-test.out").write_text(
            f"MMVet 0.0% (0/100) {failures}.0% ({failures}/100) overall {value}\n")
        (results / run / "model" / "MMVet").mkdir(parents=True)
    monkeypatch.setattr(derive, "LOGS", logs)
    monkeypatch.setattr(derive, "RESULTS", results)
    derive.main()
    assert (results / "zero/model/MMVet/derived_acc.csv").exists()
    assert not (results / "broken/model/MMVet/derived_acc.csv").exists()


@pytest.mark.parametrize("bad_metrics", [
    {"exact_match,none": float("nan")},
    {"exact_match,none": .9, "grader_failure_rate,none": .1},
    {"alias": "GQA"},
])
def test_unusable_newest_score_keeps_older_eligible_result(tmp_path, bad_metrics):
    result(tmp_path, "full", .6, 100)
    path = result(tmp_path, "unusable", .9, 200)
    data = json.loads(path.read_text())
    data["results"]["gqa"] = bad_metrics
    path.write_text(json.dumps(data))
    _, rows = dashboard.collect(tmp_path, None, Manifests([tmp_path]))
    assert rows[0]["cells"]["model"]["v"] == 60


@pytest.mark.parametrize("counts, published", [
    (100, True), (0, False),
    ({"original": 100, "sample_len": 10}, False),
    ({"original": 100, "sample_len": 100}, True),
    ({"original": 100, "effective": 101}, False),
    ({"original": 100, "effective": 99.5}, False),
    ({"original": 100.5, "effective": 100.5}, False),
    (10.5, False),
])
def test_supported_sample_count_shapes(tmp_path, counts, published):
    path = result(tmp_path, "run", .6, 100)
    data = json.loads(path.read_text())
    data["n-samples"] = {"gqa": counts}
    path.write_text(json.dumps(data))
    _, rows = dashboard.collect(tmp_path, None, Manifests([tmp_path]))
    assert bool(rows) is published


def test_malformed_manifest_fields_do_not_crash_collection(tmp_path):
    path = result(tmp_path, "run", .6, 100)
    meta = json.loads((path.parent / "run_meta.json").read_text())
    meta["generation"] = "broken"
    (path.parent / "run_meta.json").write_text(json.dumps(meta))
    _, rows = dashboard.collect(tmp_path, None, Manifests([tmp_path]))
    assert rows == []


@pytest.mark.parametrize("reason", ["partial", "limited", "unusable", "malformed"])
def test_result_rejection_blocks_same_run_legacy_import(tmp_path, monkeypatch, reason):
    root = tmp_path / "runs"
    path = result(root, "same-run", .9, 100)
    data = build(tmp_path, monkeypatch, [root])
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(data))
    contents = json.loads(path.read_text())
    if reason == "partial":
        contents["n-samples"] = {"gqa": {"original": 100, "effective": 10}}
    elif reason == "limited":
        contents["config"]["limit"] = 10
    else:
        contents["results"]["gqa"]["exact_match,none"] = float("nan")
    path.write_text("{" if reason == "malformed" else json.dumps(contents))
    data = build(tmp_path, monkeypatch, [root], "--legacy-json", legacy)
    assert data["table"] == []


@pytest.mark.parametrize("provenance", ["none", "full", "without-source"])
@pytest.mark.parametrize("native", [False, True])
def test_invalid_vlmeval_bridge_snapshot_cannot_be_resurrected(tmp_path, monkeypatch, provenance, native):
    raw, bridge, empty = tmp_path / "raw", tmp_path / "bridge", tmp_path / "empty"
    empty.mkdir()
    directory = raw / "bad-run" / "model" / "BLINK"
    directory.mkdir(parents=True)
    (directory / "model_BLINK_acc.csv").write_text("accuracy\n.9\n")
    meta = {"status": "invalid", "framework": "VLMEvalKit", "task": "BLINK", "run_id": "bad-run"}
    (directory / "run_meta.json").write_text("{" if provenance == "without-source" else json.dumps(meta))
    link = bridge / "model" / "BLINK__bad-run"
    link.parent.mkdir(parents=True)
    link.symlink_to(directory, target_is_directory=True)
    cell = {"v": 90, "raw": 90, "run": "BLINK__bad-run"}
    if provenance != "none":
        cell["prov"] = {"run_id": "bad-run", "status": "ok"}
    if provenance == "full":
        cell["source"] = {"path": str(directory / "model_BLINK_acc.csv")}
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"table": [{"task": "blink", "metric": "acc",
        "framework": "VLMEvalKit", "cells": {"model": cell}}]}))
    args = [] if native else ["--vlmeval-root", bridge]
    data = build(tmp_path, monkeypatch, [empty], *args, "--vlmeval-results-root", raw, "--legacy-json", legacy)
    assert data["table"] == []


def test_verifier_uses_older_eligible_score(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    result(a, "old", .6, 100)
    result(a, "new", float("nan"), 200)
    result(b, "other", .6, 100)
    assert audit([a, b]) == 0


def test_verifier_detects_vlmeval_csv_collision(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b"]
    for root, score in zip(roots, [.9, .2]):
        directory = root / "model" / "BLINK"
        directory.mkdir(parents=True)
        (directory / "model_BLINK_acc.csv").write_text(f"accuracy\n{score}\n")
    assert audit([], vlmeval_roots=roots) == 1


def test_verifier_checks_model_aliases_in_same_root(tmp_path):
    result(tmp_path, "a", .9, 100, model="main")
    result(tmp_path, "b", .2, 100, model="alias")
    assert audit([tmp_path], aliases={"alias": "main"}) == 1


@pytest.mark.parametrize("selected_conflict", [False, True])
def test_verifier_cli_limits_collisions_to_curated_models(tmp_path, selected_conflict):
    result(tmp_path, "selected", .6, 100, model="selected")
    result(tmp_path, "a", .9, 100, model="EXCLUDED")
    result(tmp_path, "b", .2, 100, model="excluded")
    if selected_conflict:
        result(tmp_path, "alias", .9, 100, model="selected_alias")
    models = tmp_path / "models.txt"
    models.write_text("selected|selected_alias=Selected\n")
    verifier = Path(__file__).resolve().parents[1] / "scripts/verify_dashboard.py"
    process = subprocess.run([sys.executable, str(verifier), "--runs-root", str(tmp_path),
                              "--models-file", str(models)], capture_output=True, text=True)
    assert process.returncode == (1 if selected_conflict else 0)
    assert "key 'excluded'" not in process.stdout
    assert ("key 'selected'" in process.stdout) is selected_conflict


def test_verifier_rejects_invalid_vlmeval_bridge_run(tmp_path):
    raw, bridge = tmp_path / "raw", tmp_path / "bridge"
    for model, status, score in [("MODEL", "ok", .6), ("model", "invalid", .9)]:
        directory = raw / "run" / model / "BLINK"
        directory.mkdir(parents=True)
        (directory / "model_BLINK_acc.csv").write_text(f"accuracy\n{score}\n")
        (directory / "run_meta.json").write_text(json.dumps({"status": status, "framework": "VLMEvalKit",
                                                            "task": "BLINK", "run_id": "run"}))
        link = bridge / model / "BLINK__run"
        link.parent.mkdir(parents=True)
        link.symlink_to(directory, target_is_directory=True)
    assert audit([], vlmeval_roots=[bridge], manifest_roots=[raw]) == 0


def test_old_derived_score_is_invalidated_after_judge_failure_rejection(tmp_path, monkeypatch):
    import derive_vlmeval_acc as derive
    logs, results = tmp_path / "logs", tmp_path / "results"
    (logs / "run").mkdir(parents=True)
    (logs / "run" / "vlmeval-test.out").write_text(
        "MMVet 0.0% (0/100) 10.0% (10/100) overall 75\n")
    directory = results / "run/model/MMVet"
    directory.mkdir(parents=True)
    derived = directory / "derived_acc.csv"
    derived.write_text("metric,value\noverall,75\n")
    monkeypatch.setattr(derive, "LOGS", logs)
    monkeypatch.setattr(derive, "RESULTS", results)
    derive.main()
    assert dashboard.parse_vk_acc(derived) is None
    manifests = Manifests([results])
    assert dashboard.collect_vlmeval(results / "run", None, manifests)[1] == []
    assert ("mmvet", "model", "MMVet", str(derived)) in manifests.rejected_results
    # A successful re-judge can replace this rejected generated artifact.
    (logs / "run" / "vlmeval-test.out").write_text(
        "MMVet 0.0% (0/100) 0.0% (0/100) overall 75\n")
    derive.main()
    assert dashboard.parse_vk_acc(derived) == 75


def test_cli_indexes_manifests_in_direct_vlmeval_root(tmp_path, monkeypatch):
    vk, empty = tmp_path / "vk", tmp_path / "empty"
    empty.mkdir()
    directory = vk / "model" / "BLINK"
    directory.mkdir(parents=True)
    (directory / "model_BLINK_acc.csv").write_text("accuracy\n.9\n")
    (directory / "run_meta.json").write_text(json.dumps({"status": "invalid", "framework": "VLMEvalKit",
                                                        "task": "BLINK", "run_id": "bad-run"}))
    data = build(tmp_path, monkeypatch, [empty], "--vlmeval-root", vk)
    assert data["table"] == []


def test_gather_cli_keeps_older_eligible_result(tmp_path, monkeypatch, capsys):
    import gather_results
    result(tmp_path, "full", .6, 100)
    result(tmp_path, "bad", .99, 200, status="invalid")
    monkeypatch.setattr(sys, "argv", ["gather_results", "--results-root", str(tmp_path), "--markdown"])
    gather_results.main()
    output = capsys.readouterr().out
    assert "0.6000" in output and "0.9900" not in output
