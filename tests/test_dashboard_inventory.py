"""One collection drives collision verification and dashboard publication."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_dashboard as dashboard
from test_dashboard_results import result


def vk_result(root, model, run, value, stamp):
    directory = root / model / "BLINK"
    directory.mkdir(parents=True)
    path = directory / f"{model}_BLINK_acc.csv"
    path.write_text(f"accuracy\n{value}\n")
    os.utime(path, (stamp, stamp))
    (directory / "run_meta.json").write_text(json.dumps({
        "framework": "VLMEvalKit", "task": "BLINK", "status": "ok", "run_id": run,
        "generation": {"limit": None}, "harness": {"commit": "test-commit"},
    }))
    return path


def inventory_fixture(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    paths = [result(roots[0], "first", .6, 100, model="Apertus-1p5-8B-main"),
             result(roots[1], "second", .6, 200, model="alias"),
             result(roots[1], "limited", .9, 300, model="alias", limit=10)]
    # Rejection from result contents, with an otherwise eligible manifest, must
    # survive raw-directory collection and prevent source-less legacy revival.
    bad = result(roots[1], "bad", .99, 300, model="Apertus-1p5-8B-rejected")
    content = json.loads(bad.read_text())
    content["config"]["limit"] = 10
    bad.write_text(json.dumps(content))
    paths.append(bad)
    legacy_raw = result(roots[0], "legacy-raw", .3, 100, model="legacy")
    (legacy_raw.parent / "run_meta.json").unlink()
    paths.append(legacy_raw)
    text_root, vk_root = tmp_path / "text", tmp_path / "vk"
    paths.append(result(text_root, "text", .4, 100, model="alias", task="gsm8k", framework="lm-eval"))
    paths.append(vk_result(vk_root, "alias", "vision", .5, 100))
    models = tmp_path / "models.txt"
    models.write_text("# group: Main\nmain|alias=Main\nlegacy=Legacy\nrejected=Rejected\n")
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"table": [
        {"task": "gqa", "metric": "exact_match", "framework": "lmms-eval", "cells": {
            "rejected": {"v": 99, "raw": .99, "run": "bad"}}},
        {"task": "mmvp", "metric": "mmvp_accuracy", "framework": "lmms-eval", "cells": {
            "legacy": {"v": 25, "raw": .25, "run": "purged"}}},
    ]}))
    args = ["--runs-root", *map(str, roots), "--vlmeval-root", str(vk_root),
            "--lm-eval-root", str(text_root), "--models-file", str(models),
            "--legacy-json", str(legacy)]
    return args, paths


def test_verified_build_collects_each_lane_once_with_shared_manifests(tmp_path, monkeypatch, capsys):
    args, paths = inventory_fixture(tmp_path)
    reads = dict.fromkeys(paths, 0)
    original_read = Path.read_text
    def read_text(path, *args, **kwargs):
        if path in reads:
            reads[path] += 1
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)
    instances = []
    original_manifests = dashboard.Manifests
    class CountingManifests(original_manifests):
        def __init__(self, roots):
            super().__init__(roots)
            instances.append(self)
    monkeypatch.setattr(dashboard, "Manifests", CountingManifests)
    calls = []
    for name in ("collect", "collect_lm_eval", "collect_vlmeval"):
        original = getattr(dashboard, name)
        def tracked(*args, _original=original, _name=name, **kwargs):
            calls.append((_name, args[0], args[2]))
            return _original(*args, **kwargs)
        monkeypatch.setattr(dashboard, name, tracked)
    monkeypatch.setattr(sys, "argv", ["make_dashboard", *args, "--verify", "-o", str(tmp_path / "index.html")])
    dashboard.main()
    assert "PASS — no contaminating collisions" in capsys.readouterr().out
    assert len(instances) == 1
    assert len(calls) == 4
    assert all(manifests is instances[0] for _, _, manifests in calls)
    assert reads == dict.fromkeys(paths, 1)
    data = json.loads((tmp_path / "dashboard.json").read_text())
    cells = {(row["task"], model): cell for row in data["table"] for model, cell in row["cells"].items()}
    assert cells["gqa", "main"]["v"] == 60
    assert cells["gqa", "main"]["prov"]["run_id"] == "second"
    assert cells["gqa", "main"]["source"]["path"] == str(paths[1])
    assert cells["gsm8k", "main"]["v"] == 40
    assert cells["blink", "main"]["v"] == 50
    assert cells["gqa", "legacy"]["legacy"] is True
    assert cells["mmvp", "legacy"]["legacy"] is True
    assert ("gqa", "rejected") not in cells
    assert data["evidence"] == {"manifest": 3, "legacy": 2}
    coverage = json.loads((tmp_path / "coverage.json").read_text())
    assert data["coverage"] == coverage["counts"]


@pytest.mark.parametrize("kind", ["cross-root", "canonical", "alias", "text", "vlmeval"])
def test_verified_cli_refuses_collisions_before_replacing_any_output(tmp_path, kind):
    runs = tmp_path / "runs"
    runs.mkdir()
    args = ["--runs-root", str(runs)]
    if kind == "cross-root":
        other = tmp_path / "other"
        result(runs, "one", .6, 100)
        result(other, "two", .9, 200)
        args.append(str(other))
    else:
        models = ("model", "Apertus-1p5-8B-model" if kind == "canonical" else "alias")
        if kind == "vlmeval":
            root = tmp_path / "vk"
            for model, value in zip(models, (.6, .9)):
                vk_result(root, model, "run", value, 100)
            args.extend(["--vlmeval-root", str(root)])
        else:
            root = tmp_path / "text" if kind == "text" else runs
            for model, value in zip(models, (.6, .9)):
                result(root, "run", value, 100, model=model,
                       **({"task": "gsm8k", "framework": "lm-eval"} if kind == "text" else {}))
            if kind == "text":
                args.extend(["--lm-eval-root", str(root)])
        if kind != "canonical":
            models_file = tmp_path / "models.txt"
            models_file.write_text("model|alias=Model\n")
            args.extend(["--models-file", str(models_file)])
    outputs = [tmp_path / name for name in ("index.html", "dashboard.json", "coverage.json")]
    for path in outputs:
        path.write_text(f"existing {path.name}")
    process = subprocess.run([sys.executable, str(Path(dashboard.__file__)), *args,
                              "--verify", "-o", str(outputs[0])], capture_output=True, text=True)
    assert process.returncode == 1
    assert "COLLISION" in process.stdout
    assert "FAIL — 1 colliding key(s)" in process.stdout
    assert [path.read_text() for path in outputs] == [f"existing {path.name}" for path in outputs]


@pytest.mark.parametrize("legacy", [False, True])
def test_source_mutation_during_render_refuses_publication(tmp_path, monkeypatch, legacy):
    root = tmp_path / "runs"
    source = result(root, "run", .6, 100)
    if legacy:
        (source.parent / "run_meta.json").unlink()
    else:
        path = source.parent / "run_meta.json"
        manifest = json.loads(path.read_text())
        manifest["results"] = {"artifacts": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
        path.write_text(json.dumps(manifest))
        # A filesystem may report identical metadata for a rapid rewrite. The
        # final attested-content check must still catch different source bytes.
        signature = dashboard._source_signature(source)
        monkeypatch.setattr(dashboard, "_source_signature", lambda _path: signature)
    outputs = [tmp_path / name for name in ("index.html", "dashboard.json", "coverage.json")]
    for path in outputs:
        path.write_text("existing")
    original_coverage = dashboard.coverage_report
    def mutate(*args, **kwargs):
        data = source.read_text()
        source.write_text(data.replace("0.6", "0.9") + ("\n" if legacy else ""))
        os.utime(source, (100, 100))
        return original_coverage(*args, **kwargs)
    monkeypatch.setattr(dashboard, "coverage_report", mutate)
    monkeypatch.setattr(sys, "argv", ["make_dashboard", "--runs-root", str(root),
                                     "--verify", "-o", str(outputs[0])])
    with pytest.raises(SystemExit) as raised:
        dashboard.main()
    assert raised.value.code == 1
    assert all(path.read_text() == "existing" for path in outputs)


@pytest.mark.parametrize("framework", ["lmms-eval", "VLMEvalKit", "lm-eval"])
def test_verified_build_hashes_artifact_once_on_collection_and_once_before_write(tmp_path, monkeypatch, framework):
    import suite.coverage as coverage
    root = tmp_path / "runs"
    root.mkdir()
    args = ["--runs-root", str(root)]
    if framework == "VLMEvalKit":
        source = vk_result(tmp_path / "vk", "model", "run", .6, 100)
        args.extend(["--vlmeval-root", str(tmp_path / "vk")])
    elif framework == "lm-eval":
        source = result(tmp_path / "text", "run", .6, 100, framework=framework, task="gsm8k")
        args.extend(["--lm-eval-root", str(tmp_path / "text")])
    else:
        source = result(root, "run", .6, 100)
        data = json.loads(source.read_text())
        data["results"]["vqav2"] = {"exact_match,none": .7}
        source.write_text(json.dumps(data))
    path = source.parent / "run_meta.json"
    manifest = json.loads(path.read_text())
    manifest["results"] = {"artifacts": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
    path.write_text(json.dumps(manifest))
    hashes = []
    original_hash = coverage.sha256_file
    def tracked(path, *args, **kwargs):
        hashes.append(path)
        return original_hash(path, *args, **kwargs)
    monkeypatch.setattr(coverage, "sha256_file", tracked)
    monkeypatch.setattr(sys, "argv", ["make_dashboard", *args, "--verify", "-o", str(tmp_path / "index.html")])
    dashboard.main()
    assert hashes == [source, source]
