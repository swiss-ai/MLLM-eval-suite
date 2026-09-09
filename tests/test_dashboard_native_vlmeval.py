"""Native VLMEvalKit discovery must select the same evidence as the old bridge."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_dashboard as dashboard
from suite.coverage import Manifests
from test_dashboard_results import build
from verify_dashboard import audit


def artifact(root, model, benchmark, kind, value, stamp, *, run, status="ok"):
    directory = root / model / benchmark
    directory.mkdir(parents=True, exist_ok=True)
    name = "derived_acc.csv" if kind == "derived" else f"{model}_{benchmark}_{kind}.csv"
    path = directory / name
    path.write_text(f"accuracy\n{value}\n")
    os.utime(path, (stamp, stamp))
    (directory / "run_meta.json").write_text(json.dumps({
        "framework": "VLMEvalKit", "task": benchmark, "status": status,
        "run_id": run, "generation": {"limit": None},
        "results": {"artifacts": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}},
    }))
    return path


def old_bridge(bridge, shared, suite):
    # The removed shell bridge linked every benchmark of every run, rather
    # than selecting the newest run directory. This is a test-only oracle.
    sources = [(shared, "outputs"), *((run, run.name) for run in sorted(suite.iterdir()))]
    for root, tag in sources:
        for model in sorted(root.iterdir()):
            if not model.is_dir():
                continue
            (bridge / model.name).mkdir(parents=True, exist_ok=True)
            for benchmark in sorted(model.iterdir()):
                if benchmark.is_dir():
                    (bridge / model.name / f"{benchmark.name}__{tag}").symlink_to(benchmark)


@pytest.fixture
def native_results(tmp_path):
    shared, suite = tmp_path / "shared", tmp_path / "suite"
    paths = {}
    for root, benchmark, kind, value, stamp, run, status in [
        (shared, "BLINK", "acc", .2, 100, "shared", "ok"),
        (suite / "new-run", "BLINK", "acc", .4, 200, "new-run", "ok"),
        (suite / "old-run", "BLINK", "acc", .7, 300, "old-run", "ok"),
        (suite / "failed", "BLINK", "acc", .9, 500, "failed", "failed"),
        (shared, "MMVet", "acc", .6, 100, "shared", "ok"),
        (suite / "new-run", "MMVet", "derived", .9, 400, "new-run", "ok"),
        (suite / "old-run", "MUIRBench", "score", .5, 100, "old-run", "ok"),
        (shared, "MUIRBench", "derived", .8, 200, "shared", "ok"),
    ]:
        paths[benchmark, run] = artifact(root, "model", benchmark, kind, value, stamp,
                                       run=run, status=status)
    # Different model spellings retain their own newest candidates for the
    # canonical/curated-alias collision audit.
    paths["alias"] = artifact(shared, "alias", "BLINK", "acc", .7, 200, run="alias")
    bridge = tmp_path / "bridge"
    old_bridge(bridge, shared, suite)
    return shared, suite, bridge, paths


def test_native_and_bridge_select_identical_full_cells(native_results):
    shared, suite, bridge, paths = native_results
    expected = dashboard.collect_vlmeval(bridge, None, Manifests([shared, suite]))
    actual = dashboard.collect_vlmeval(shared, None, Manifests([shared, suite]), results_roots=[suite])
    assert actual == expected
    cells = {row["task"]: row["cells"]["model"] for row in actual[1]}
    assert {task: cell["v"] for task, cell in cells.items()} == {"blink": 70, "mmvet": 60, "muirbench": 50}
    assert cells["blink"]["source"]["path"] == str(paths["BLINK", "old-run"])
    assert cells["mmvet"]["source"]["path"] == str(paths["MMVet", "shared"])
    assert cells["muirbench"]["source"]["path"] == str(paths["MUIRBench", "old-run"])


def test_native_union_reads_shared_manifest_eligibility(tmp_path):
    shared, suite = tmp_path / "shared", tmp_path / "suite"
    artifact(shared, "model", "BLINK", "acc", .9, 300, run="shared", status="invalid")
    valid = artifact(suite / "old-run", "model", "BLINK", "acc", .5, 100, run="old-run")
    _manifests, collections = dashboard.collect_inventory([], vlmeval_roots=[shared], vlmeval_results_roots=[suite])
    cell = collections[0][2][0]["cells"]["model"]
    assert cell["v"] == 50
    assert cell["source"]["path"] == str(valid)
    assert cell["prov"]["run_id"] == "old-run"


def test_native_cli_matches_bridge_with_aliases(tmp_path, monkeypatch, native_results, capsys):
    shared, suite, bridge, paths = native_results
    empty = tmp_path / "empty"
    empty.mkdir()
    models = tmp_path / "models.txt"
    models.write_text("model|alias=Model\n")
    args = ["--models-file", models, "--verify"]
    # Index shared manifests for the bridge baseline too (rglob does not follow
    # its directory symlinks). There are no lmms artifacts in this fixture.
    previous = build(tmp_path, monkeypatch, [empty, shared], "--vlmeval-root", bridge,
                     "--vlmeval-results-root", suite, *args)
    current = build(tmp_path, monkeypatch, [empty, shared], "--vlmeval-root", shared,
                    "--vlmeval-results-root", suite, *args)
    assert current == previous
    assert current["models"] == ["model"]
    assert audit([], vlmeval_roots=[shared], vlmeval_results_roots=[suite], aliases={"alias": "model"}) == 0
    # A newly conflicting alias must still stop verified publication.
    paths["alias"].write_text("accuracy\n.9\n")
    (paths["alias"].parent / "run_meta.json").unlink()
    assert audit([], vlmeval_roots=[shared], vlmeval_results_roots=[suite], aliases={"alias": "model"}) == 1
    diagnostic = capsys.readouterr().out
    assert str(paths["alias"]) in diagnostic
    assert str(paths["BLINK", "old-run"]) in diagnostic


def test_native_run_root_order_and_model_filter(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    artifact(a / "z-run", "Apertus-1p5-8B-main", "BLINK", "acc", .2, 100, run="z-run")
    newer = artifact(b / "a-run", "Apertus-1p5-8B-main", "BLINK", "acc", .7, 200, run="a-run")
    artifact(a / "z-run", "excluded", "BLINK", "acc", .8, 300, run="z-run")
    outputs = [dashboard.collect_vlmeval(None, ["main"], Manifests([a, b]), results_roots=roots)
               for roots in ([a, b], [b, a])]
    assert outputs[0] == outputs[1]
    assert outputs[0][0] == ["main"]
    assert outputs[0][1][0]["cells"]["main"]["source"]["path"] == str(newer)


@pytest.mark.parametrize("manifest", [False, True])
@pytest.mark.parametrize("offset_ns", [0, 1])
def test_native_preserves_bridge_ties_and_manifestless_run_labels(tmp_path, manifest, offset_ns):
    shared, suite, bridge = tmp_path / "shared", tmp_path / "suite", tmp_path / "bridge"
    paths = [artifact(shared, "model", "BLINK", "acc", .2, 100, run="shared"),
             artifact(suite / "old-run", "model", "BLINK", "acc", .7, 100, run="old-run")]
    for index, path in enumerate(paths):
        stamp = 1_700_000_000_000_000_000 + index * offset_ns
        os.utime(path, ns=(stamp, stamp))
    if not manifest:
        for path in paths:
            (path.parent / "run_meta.json").unlink()
    old_bridge(bridge, shared, suite)
    native = dashboard.collect_vlmeval(shared, None, Manifests([shared, suite]), results_roots=[suite])
    previous = dashboard.collect_vlmeval(bridge, None, Manifests([shared, suite]))
    # Hand-checked on b796c3b: BLINK__outputs sorts after BLINK__old-run,
    # even though the suite's resolved filesystem path sorts after shared.
    cell = native[1][0]["cells"]["model"]
    assert cell["v"] == 20
    assert cell["source"]["path"] == str(paths[0])
    assert cell["run"] == ("shared" if manifest else "BLINK__outputs")
    assert native == previous


def test_multiple_shared_roots_with_suite_sources_is_explicitly_ambiguous(tmp_path):
    shared = [tmp_path / "a", tmp_path / "b"]
    for root in shared:
        root.mkdir()
    suite = tmp_path / "suite"
    suite.mkdir()
    with pytest.raises(ValueError, match="one shared VLMEvalKit root"):
        dashboard.collect_inventory([], vlmeval_roots=shared, vlmeval_results_roots=[suite])


@pytest.mark.parametrize("change", ["content", "manifest", "new-root-manifest"])
def test_native_source_changes_after_audit_refuse_publication(tmp_path, monkeypatch, change):
    empty, suite = tmp_path / "empty", tmp_path / "suite"
    empty.mkdir()
    source = artifact(suite / "run", "model", "BLINK", "acc", .6, 100, run="run")
    if change == "new-root-manifest":
        (source.parent / "run_meta.json").unlink()
    outputs = [tmp_path / name for name in ("index.html", "dashboard.json", "coverage.json")]
    for path in outputs:
        path.write_text("existing")
    # The fresh content check must work even if a filesystem reports the same
    # signature for a rapid rewrite of an already attested source.
    signature = dashboard._source_signature(source)
    monkeypatch.setattr(dashboard, "_source_signature", lambda _path: signature)
    original_audit = dashboard.audit_inventory
    def mutate(*args, **kwargs):
        result = original_audit(*args, **kwargs)
        if change == "content":
            source.write_text("accuracy\n0.9\n")
            os.utime(source, (100, 100))
        elif change == "manifest":
            (source.parent / "run_meta.json").write_text(json.dumps({"status": "failed"}))
        else:
            (suite / "run_meta.json").write_text(json.dumps({"status": "failed"}))
        return result
    monkeypatch.setattr(dashboard, "audit_inventory", mutate)
    monkeypatch.setattr(sys, "argv", ["make_dashboard", "--runs-root", str(empty),
                                     "--vlmeval-results-root", str(suite), "--verify", "-o", str(outputs[0])])
    with pytest.raises(SystemExit) as raised:
        dashboard.main()
    assert raised.value.code == 1
    assert all(path.read_text() == "existing" for path in outputs)
