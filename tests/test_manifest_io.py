"""I/O budgets for finalization must preserve the existing result evidence."""
import builtins
from collections import Counter
import io
import json
import os
from pathlib import Path

import suite.manifest as manifest


def begin(root, task, framework="lmms-eval"):
    manifest.start(root, framework=framework, task=task, run_id="run", model_path=root / "checkpoint",
                   harness_dir=root, tokenizer_path=None, chat_template=None,
                   model_args="", gen_kwargs="", thinking=False)


def test_finalize_streams_each_sample_file_once(tmp_path, monkeypatch):
    begin(tmp_path, "gqa,chartqa")
    result = tmp_path / "fixture_results.json"
    result.write_text(json.dumps({"results": {"gqa": {"acc": .5}, "chartqa": {"acc": .6}},
                                 "n-samples": {"gqa": 3, "chartqa": 1}}))
    paths = [tmp_path / name for name in ("old_samples_gqa.jsonl", "new_samples_gqa.jsonl", "old_samples_chartqa.jsonl")]
    records = [
        [{"doc_id": 1, "token_counts": [{"output_tokens": 4}]},
         {"doc_id": 2, "token_counts": {"output_tokens": 6}},
         {"resps": ["a\u2028b\u0085c"], "token_counts": {"output_tokens": 3}}],
        [{"doc_id": 1, "token_counts": {"output_tokens": 10}}, {"doc_id": 2}],
        [{"doc_id": 1, "token_counts": {"output_tokens": 2}}],
    ]
    for i, (path, docs) in enumerate(zip(paths, records)):
        path.write_text("\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs) + "\nnull\n{broken\n")
        # Choose replacement order explicitly, independent of filename order.
        stamp = result.stat().st_mtime_ns + (i + 1) * 1_000_000
        os.utime(path, ns=(stamp, stamp))
    opens = Counter()
    real_open, real_io_open = builtins.open, io.open

    class StreamingFile:
        def __init__(self, fh):
            self.fh = fh

        def __enter__(self):
            self.fh.__enter__()
            return self

        def __exit__(self, *args):
            return self.fh.__exit__(*args)

        def __iter__(self):
            return iter(self.fh)

        def read(self, *args):
            raise AssertionError("sample logs must be streamed, not materialized")

        def __getattr__(self, name):
            return getattr(self.fh, name)

    def counted_open(opener, path, *args, **kwargs):
        fh = opener(path, *args, **kwargs)
        if not isinstance(path, int) and Path(path) in paths:
            opens[Path(path)] += 1
            return StreamingFile(fh)
        return fh

    monkeypatch.setattr(builtins, "open", lambda *a, **kw: counted_open(real_open, *a, **kw))
    monkeypatch.setattr(io, "open", lambda *a, **kw: counted_open(real_io_open, *a, **kw))
    status, data = manifest.finalize(tmp_path / "run_meta.json", [], tmp_path, 0)
    assert status == "ok", data["error"]
    assert data["results"]["n_samples"] == 4
    assert data["results"]["output_tokens"] == {"n": 3, "records": 4, "mean": 5, "median": 3, "max": 10}
    assert opens == {path: 1 for path in paths}


def test_finalize_parses_multitask_result_and_registry_once(tmp_path, monkeypatch):
    tasks = [f"fixture_task_{i}" for i in range(20)]
    begin(tmp_path, ",".join(tasks))
    result = tmp_path / "fixture_results.json"
    result.write_text(json.dumps({"results": {task: {"acc": .5} for task in tasks},
                                 "n-samples": {task: 1 for task in tasks}}))
    counts = Counter()
    read, load, discover = Path.read_text, manifest.load_registry, manifest._result_paths

    def counted_read(path, *a, **kw):
        if path == result:
            counts["result_reads"] += 1
        return read(path, *a, **kw)

    def counted_load():
        counts["registry_loads"] += 1
        return load()

    def counted_discover(*a):
        counts["discovery_passes"] += 1
        return discover(*a)

    monkeypatch.setattr(Path, "read_text", counted_read)
    monkeypatch.setattr(manifest, "load_registry", counted_load)
    monkeypatch.setattr(manifest, "_result_paths", counted_discover)
    status, data = manifest.finalize(tmp_path / "run_meta.json", [], tmp_path, 0)
    assert status == "ok", data["error"]
    assert data["results"]["result_samples"] == 20
    assert counts == {"result_reads": 1, "registry_loads": 1, "discovery_passes": 1}


def test_finalize_reads_each_multidataset_csv_once(tmp_path, monkeypatch):
    begin(tmp_path, "MMVP,POPE", "VLMEvalKit")
    files = [tmp_path / f"model_{task}_acc.csv" for task in ("MMVP", "POPE")]
    for path in files:
        path.write_text("Category,Accuracy\nOverall,75\n")
    counts = Counter()
    real_open = builtins.open

    def counted_open(path, *a, **kw):
        mode = a[0] if a else kw.get("mode", "r")
        if not isinstance(path, int) and Path(path) in files and "b" not in mode:
            counts[Path(path)] += 1
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", counted_open)
    status, data = manifest.finalize(tmp_path / "run_meta.json", [], tmp_path, 0)
    assert status == "ok", data["error"]
    assert set(data["results"]["artifacts"]) == {str(path.resolve()) for path in files}
    assert counts == {path: 1 for path in files}
