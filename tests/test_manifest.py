import json

import pytest

from suite.manifest import EXIT_FAILED, EXIT_INVALID, finalize, output_token_stats, sample_record_count, start


def _model_dir(tmp_path):
    m = tmp_path / "model"
    m.mkdir()
    (m / "config.json").write_text('{"architectures": ["ApertusForCausalLM"]}')
    (m / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (m / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 16)
    t = tmp_path / "tok"
    t.mkdir()
    (t / "tokenizer.json").write_text("{}")
    (t / "chat_template.jinja").write_text("{{ bos_token }}")
    return m, t


def test_start_records_identity(tmp_path):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    man = start(out, framework="lmms-eval", task="gqa", run_id="r1", model_path=m, harness_dir=tmp_path, tokenizer_path=t,
                chat_template=t / "chat_template.jinja", model_args="model=x,enable_thinking=True",
                gen_kwargs="max_new_tokens=32768", thinking=True, generation={"tp": 4, "max_model_len": 131072})
    on_disk = json.loads((out / "run_meta.json").read_text())
    assert on_disk == man and man["schema"] == 1 and man["status"] == "running"
    assert man["model"]["config_sha256"] and man["model"]["weights"]["shards"][0]["size"] == 16
    assert man["tokenizer"]["chat_template_sha256"] and man["thinking"]["requested"] is True
    assert man["generation"]["tp"] == 4 and man["generation"]["model_args"] == "model=x,enable_thinking=True"
    assert man["harness"]["checkout"] == str(tmp_path) and man["harness"]["commit"] is None


def test_output_token_stats(tmp_path):
    s = tmp_path / "samples_x.jsonl"
    s.write_text("\n".join(json.dumps({"token_counts": [{"output_tokens": n}]}) for n in (2, 4, 600)) + "\n")
    st = output_token_stats([s])
    assert st["n"] == 3 and st["mean"] == pytest.approx(202) and st["max"] == 600


def test_output_token_stats_deduplicates_resumed_doc_ids_per_task(tmp_path):
    alpha = tmp_path / "samples_alpha_1.jsonl"
    beta = tmp_path / "samples_beta_1.jsonl"
    alpha.write_text("\n".join(json.dumps({"doc_id": 7, "token_counts": [{"output_tokens": n}]}) for n in (2, 10)) + "\n")
    beta.write_text(json.dumps({"doc_id": 7, "token_counts": [{"output_tokens": 4}]}) + "\n")
    st = output_token_stats([alpha, beta])
    assert st["n"] == 2 and st["records"] == 2 and st["mean"] == pytest.approx(7) and st["max"] == 10


@pytest.mark.parametrize("names", [
    ("2026-09-08T12-00-00_samples_mmlu_a", "2026-09-08T12-00-00_samples_mmlu_b", "2026-09-08T13-00-00_samples_mmlu_a"),
    ("samples_mmlu_a_2026-09-08T12-00-00", "samples_mmlu_b_2026-09-08T12-00-00", "samples_mmlu_a_2026-09-08T13-00-00"),
])
def test_real_harness_sample_names_keep_tasks_separate_across_resumes(tmp_path, names):
    import os
    files = [tmp_path / f"{name}.jsonl" for name in names]
    for i, (path, tokens) in enumerate(zip(files, (2, 4, 10))):
        path.write_text(json.dumps({"doc_id": 7, "token_counts": [{"output_tokens": tokens}]}) + "\n")
        os.utime(path, (100 + i, 100 + i))
    assert sample_record_count(files[:2]) == 2
    assert sample_record_count(files) == 2
    stats = output_token_stats(files)
    assert stats["n"] == 2 and stats["mean"] == 7 and stats["max"] == 10


def test_sample_logs_skip_valid_json_that_is_not_a_record(tmp_path):
    samples = tmp_path / "2026-09-08T12-00-00_samples_mmvp.jsonl"
    samples.write_text("\n".join(json.dumps(value) for value in (
        None, [], 7, "text", {"doc_id": 0, "token_counts": [{"output_tokens": 9}]},
    )) + "\n")
    assert sample_record_count([samples]) == 1
    assert output_token_stats([samples])["mean"] == 9


@pytest.mark.parametrize("tokens", [-1, 1.5, float("nan"), float("inf"), True])
def test_invalid_output_token_counts_remain_unknown(tmp_path, tokens):
    samples = tmp_path / "samples_mmvp_2026-09-08T12-00-00.jsonl"
    samples.write_text(json.dumps({"doc_id": 0, "token_counts": [{"output_tokens": tokens}]}) + "\n")
    assert sample_record_count([samples]) == 1
    assert output_token_stats([samples]) is None


def _run(tmp_path, thinking, tokens, log_text="", results=True, rc=0, result_data=None, task="mmvp", generation=None):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="lmms-eval", task=task, run_id="r1", model_path=m, harness_dir=tmp_path, tokenizer_path=t,
          chat_template=None, model_args="", gen_kwargs="", thinking=thinking, generation=generation)
    sub = out / "textview__m"
    sub.mkdir()
    if results:
        (sub / "20260908_x_results.json").write_text(json.dumps(
            result_data if result_data is not None else {"results": {"mmvp": {"mmvp_accuracy,none": 0.7}}}
        ))
        (sub / "20260908_x_samples_mmvp.jsonl").write_text(
            "\n".join(json.dumps({"token_counts": [{"output_tokens": n}]}) for n in tokens) + "\n")
    log = tmp_path / "job.out"
    log.write_text(log_text)
    return finalize(out / "run_meta.json", [log], out, harness_rc=rc)


def test_finalize_ok(tmp_path):
    status, man = _run(tmp_path, thinking=True, tokens=[300, 500],
                       log_text="apertus_1p5_vllm: enable_thinking=True\napertus_1p5_vllm: thinking canary passed (120 tokens)\n")
    assert status == "ok" and man["thinking"]["effective"] is True and man["thinking"]["canary"] == "passed"
    assert man["results"]["n_samples"] == 2 and man["results"]["file"].endswith("_results.json")


def test_finalize_invalid_when_thinking_did_not_engage(tmp_path):
    status, man = _run(tmp_path, thinking=True, tokens=[2, 3, 4], log_text="apertus_1p5_vllm: enable_thinking=None\n")
    assert status == "invalid" and man["thinking"]["effective"] is None and "effective evidence" in man["error"]


def test_finalize_rejects_unobserved_thinking_even_with_long_outputs(tmp_path):
    status, man = _run(tmp_path, thinking=True, tokens=[300], log_text="")
    assert status == "invalid" and "effective evidence" in man["error"]


def test_finalize_rejects_malformed_result_json(tmp_path):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="lmms-eval", task="mmvp", run_id="r1", model_path=m, harness_dir=tmp_path,
          tokenizer_path=t, chat_template=None, model_args="", gen_kwargs="", thinking=False)
    (out / "x_results.json").write_text("{")
    log = tmp_path / "job.out"
    log.write_text("")
    status, man = finalize(out / "run_meta.json", [log], out, harness_rc=0)
    assert status == "invalid" and "not valid JSON" in man["error"]


def test_finalize_rejects_result_for_another_task(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[5], result_data={"results": {"pope": {"acc": 0.7}}})
    assert status == "invalid" and "does not contain requested task" in man["error"]


def test_finalize_rejects_known_sample_count_mismatch(tmp_path):
    status, man = _run(
        tmp_path, thinking=False, tokens=[5, 6],
        result_data={"results": {"mmvp": {"acc": 0.7}}, "n-samples": {"mmvp": {"effective": 3}}},
    )
    assert status == "invalid" and "sample count 3 disagrees with 2 sample records" in man["error"]


def test_finalize_rejects_unlimited_partial_result_metadata(tmp_path):
    status, man = _run(
        tmp_path, thinking=False, tokens=[],
        result_data={"results": {"mmvp": {"acc": 0.7}}, "n-samples": {"mmvp": {"original": 10, "effective": 5}}},
    )
    assert status == "invalid" and "partial" in man["error"]


def test_finalize_keeps_explicitly_limited_partial_run_ok(tmp_path):
    status, man = _run(
        tmp_path, thinking=False, tokens=[], generation={"limit": 5},
        result_data={"results": {"mmvp": {"acc": 0.7}}, "n-samples": {"mmvp": {"original": 10, "effective": 5}}},
    )
    assert status == "ok" and man["results"]["partial"] is True


@pytest.mark.parametrize("limit", [0, "0", False])
def test_finalize_zero_limit_does_not_authorize_partial_results(tmp_path, limit):
    status, man = _run(tmp_path, thinking=False, tokens=[], generation={"limit": limit}, result_data={
        "results": {"mmvp": {"acc": 0.7}}, "n-samples": {"mmvp": {"original": 10, "effective": 5}},
    })
    assert status == "invalid" and "partial" in man["error"]


def test_finalize_rejects_task_with_no_numeric_metrics(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[], result_data={"results": {"mmvp": {}}})
    assert status == "invalid" and "numeric metrics" in man["error"]


def test_finalize_rejects_json_with_only_error_or_count_metrics(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[], result_data={
        "results": {"mmvp": {"acc,none": "N/A", "acc_stderr,none": 0.0, "samples": 10}},
    })
    assert status == "invalid" and "numeric metrics" in man["error"]


@pytest.mark.parametrize("counts", [
    {"effective": -1, "original": 10},
    {"effective": 11, "original": 10},
    {"effective": float("nan"), "original": 10},
    {"effective": float("inf"), "original": 10},
    {"effective": True, "original": 10},
    {"effective": "5", "original": 10},
    {"effective": 1.5, "original": 10},
    {"effective": 5, "original": -1},
])
def test_finalize_rejects_invalid_known_sample_counts(tmp_path, counts):
    status, man = _run(tmp_path, thinking=False, tokens=[], result_data={
        "results": {"mmvp": {"acc": 0.7}}, "n-samples": {"mmvp": counts},
    })
    assert status == "invalid" and "sample count" in man["error"]


def test_finalize_rejects_known_partial_member_when_group_total_is_unknown(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[], task="mmlu", result_data={
        "results": {"mmlu_a": {"acc": 0.7}, "mmlu_b": {"acc": 0.8}},
        "groups": {"mmlu": {"acc": 0.75}},
        "group_subtasks": {"mmlu": ["mmlu_a", "mmlu_b"]},
        "n-samples": {"mmlu_a": {"original": 10, "effective": 5}, "mmlu_b": {"effective": 3}},
    })
    assert status == "invalid" and man["results"]["partial"] is True


def test_finalize_accepts_a_complete_group_result(tmp_path):
    status, man = _run(
        tmp_path, thinking=False, tokens=[],
        result_data={
            "results": {"mmlu_a": {"acc": 0.7}, "mmlu_b": {"acc": 0.8}},
            "groups": {"mmlu": {"acc": 0.75}},
            "group_subtasks": {"mmlu": ["mmlu_a", "mmlu_b"]},
            "n-samples": {"mmlu_a": {"effective": 2}, "mmlu_b": {"effective": 3}},
        }, task="mmlu",
    )
    # The group is accepted from its declared members; no sample logs means the count is explicitly unknown.
    assert status == "ok" and man["results"]["n_samples"] is None and man["results"]["result_samples"] == 5


def test_finalize_accepts_nested_nonaggregating_group_and_counts_each_leaf_once(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[], task="mmlu", result_data={
        "results": {"mmlu": {" ": " ", "alias": "mmlu"}, "stem": {" ": " ", "alias": "stem"},
                    "a": {"acc,none": 0.7}, "b": {"acc,none": 0.8}},
        "group_subtasks": {"mmlu": ["stem", "b"], "stem": ["a", "b"], "a": [], "b": []},
        "n-samples": {"a": {"original": 2, "effective": 2}, "b": {"original": 3, "effective": 3}},
    })
    assert status == "ok" and man["results"]["result_samples"] == 5


def test_finalize_rejects_nested_group_missing_a_leaf(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[], task="mmlu", result_data={
        "results": {"mmlu": {"acc": 0.7}, "stem": {"acc": 0.7}, "a": {"acc": 0.7}},
        "groups": {"mmlu": {"acc": 0.7}, "stem": {"acc": 0.7}},
        "group_subtasks": {"mmlu": ["stem"], "stem": ["a", "b"]},
    })
    assert status == "invalid" and "missing" in man["error"]


def test_finalize_rejects_cyclic_group_declarations(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[], task="mmlu", result_data={
        "results": {"mmlu": {"acc": 0.7}, "stem": {"acc": 0.7}},
        "groups": {"mmlu": {"acc": 0.7}, "stem": {"acc": 0.7}},
        "group_subtasks": {"mmlu": ["stem"], "stem": ["mmlu"]},
    })
    assert status == "invalid" and "cycle" in man["error"]


def test_finalize_rejects_a_group_with_a_missing_declared_subtask(tmp_path):
    status, man = _run(
        tmp_path, thinking=False, tokens=[], task="mmlu",
        result_data={
            "results": {"mmlu": {"acc": 0.7}, "mmlu_a": {"acc": 0.7}},
            "groups": {"mmlu": {"acc": 0.7}},
            "group_subtasks": {"mmlu": ["mmlu_a", "mmlu_b"]},
        },
    )
    assert status == "invalid" and "missing one or more declared subtasks" in man["error"]


def test_finalize_rejects_result_that_existed_when_the_run_started(tmp_path):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    old = out / "old_results.json"
    old.write_text(json.dumps({"results": {"mmvp": {"acc": 0.7}}}))
    start(out, framework="lmms-eval", task="mmvp", run_id="r1", model_path=m, harness_dir=tmp_path,
          tokenizer_path=t, chat_template=None, model_args="", gen_kwargs="", thinking=False)
    log = tmp_path / "job.out"
    log.write_text("")
    status, man = finalize(out / "run_meta.json", [log], out, harness_rc=0)
    assert status == "invalid" and "predate this run" in man["error"]


def test_finalize_accepts_a_preexisting_result_rewritten_by_a_resumed_run(tmp_path):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = out / "resumed_results.json"
    result.write_text(json.dumps({"results": {"mmvp": {"acc": 0.1}}}))
    start(out, framework="lmms-eval", task="mmvp", run_id="r1", model_path=m, harness_dir=tmp_path,
          tokenizer_path=t, chat_template=None, model_args="", gen_kwargs="", thinking=False)
    result.write_text(json.dumps({"results": {"mmvp": {"acc": 0.71}}}))
    log = tmp_path / "job.out"
    log.write_text("")
    status, man = finalize(out / "run_meta.json", [log], out, harness_rc=0)
    assert status == "ok" and man["results"]["file"] == str(result)


def test_finalize_failed_when_a_worker_died_despite_results(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[5], log_text="Worker proc VllmWorker-2 died unexpectedly at rank 2\n")
    assert status == "failed" and "died unexpectedly" in man["error"]


def test_finalize_ok_with_warning_on_recovered_error(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[5], log_text="Error during evaluation: judge retry 1 of 3\n")
    assert status == "ok" and man["warnings"] == ["judge retry 1 of 3"] and man["error"] is None


def test_finalize_failed_on_error_without_results(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[], log_text="Error during evaluation: boom\n", results=False)
    assert status == "failed" and man["error"] == "no results file produced" and man["warnings"] == ["boom"]


def test_finalize_failed_when_no_results(tmp_path):
    status, _ = _run(tmp_path, thinking=False, tokens=[], results=False)
    assert status == "failed"


def test_cli_exit_codes(tmp_path):
    from suite.manifest import main
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    assert main(["start", "--out", str(out), "--framework", "lmms-eval", "--task", "gqa", "--run-id", "r",
                 "--model", str(m), "--tokenizer", str(t), "--harness-dir", str(tmp_path), "--tp", "2", "--limit", "8"]) == 0
    assert json.loads((out / "run_meta.json").read_text())["generation"]["tp"] == 2
    log = tmp_path / "l.out"
    log.write_text("torch.OutOfMemoryError: CUDA out of memory\n")
    assert main(["finalize", "--manifest", str(out / "run_meta.json"), "--log", str(log),
                 "--results-dir", str(out), "--harness-rc", "0"]) == EXIT_FAILED


def test_cli_exit_code_for_invalid_thinking(tmp_path):
    from suite.manifest import main
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    assert main(["start", "--out", str(out), "--framework", "lmms-eval", "--task", "mmvp", "--run-id", "r",
                 "--model", str(m), "--tokenizer", str(t), "--harness-dir", str(tmp_path), "--thinking"]) == 0
    (out / "x_results.json").write_text(json.dumps({"results": {"mmvp": {"mmvp_accuracy,none": 0.5}}}))
    (out / "x_samples_mmvp.jsonl").write_text(json.dumps({"token_counts": [{"output_tokens": 3}]}) + "\n")
    log = tmp_path / "l.out"
    log.write_text("apertus_1p5_vllm: enable_thinking=None\n")
    assert main(["finalize", "--manifest", str(out / "run_meta.json"), "--log", str(log),
                 "--results-dir", str(out), "--harness-rc", "0"]) == EXIT_INVALID


def test_vlmevalkit_results_pick_newest_file(tmp_path):
    import os
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="VLMEvalKit", task="MMVP", run_id="r1", model_path=m, harness_dir=tmp_path, tokenizer_path=t,
          chat_template=None, model_args="", gen_kwargs="", thinking=False)
    old = out / "MMVP_score.csv"
    old.write_text('"Category","Accuracy (%)"\n"Overall",12.5\n')
    new = out / "MMVP_acc.csv"
    new.write_text('"Category","Accuracy (%)"\n"Overall",50.0\n')
    os.utime(old, (1, 1))
    log = tmp_path / "l.out"
    log.write_text("")
    status, man = finalize(out / "run_meta.json", [log], out, harness_rc=0)
    assert status == "ok" and man["results"]["file"].endswith("MMVP_acc.csv")


def test_finalize_accepts_thinking_vlmevalkit_wide_csv_without_token_logs(tmp_path):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="VLMEvalKit", task="MMVP", run_id="r1", model_path=m, harness_dir=tmp_path, tokenizer_path=t,
          chat_template=None, model_args="", gen_kwargs="", thinking=True)
    (out / "MMVP_acc.csv").write_text('"Category","Accuracy (%)","Samples"\n"Overall",68.33,300\n')
    log = tmp_path / "job.out"
    log.write_text("apertus_1p5_vllm: enable_thinking=True\nthinking canary passed\n")
    status, man = finalize(out / "run_meta.json", [log], out, harness_rc=0)
    assert status == "ok" and man["results"]["output_tokens"] is None


def test_finalize_rejects_garbage_vlmevalkit_csv(tmp_path):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="VLMEvalKit", task="MMVP", run_id="r1", model_path=m, harness_dir=tmp_path, tokenizer_path=t,
          chat_template=None, model_args="", gen_kwargs="", thinking=False)
    (out / "MMVP_acc.csv").write_text("not a score\n")
    log = tmp_path / "job.out"
    log.write_text("")
    status, man = finalize(out / "run_meta.json", [log], out, harness_rc=0)
    assert status == "invalid" and "CSV" in man["error"]


def test_finalize_rejects_vlmevalkit_csv_with_only_sample_counts(tmp_path):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="VLMEvalKit", task="MMVP", run_id="r1", model_path=m, harness_dir=tmp_path, tokenizer_path=t,
          chat_template=None, model_args="", gen_kwargs="", thinking=False)
    (out / "MMVP_acc.csv").write_text("Category,Samples\nOverall,300\n")
    log = tmp_path / "job.out"
    log.write_text("")
    status, man = finalize(out / "run_meta.json", [log], out, harness_rc=0)
    assert status == "invalid" and "numeric CSV metrics" in man["error"]


def test_finalize_rejects_vlmevalkit_csv_for_another_dataset(tmp_path):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="VLMEvalKit", task="MMVP", run_id="r1", model_path=m, harness_dir=tmp_path, tokenizer_path=t,
          chat_template=None, model_args="", gen_kwargs="", thinking=False)
    (out / "POPE_acc.csv").write_text('"Category","Accuracy (%)"\n"Overall",50.0\n')
    log = tmp_path / "job.out"
    log.write_text("")
    status, man = finalize(out / "run_meta.json", [log], out, harness_rc=0)
    assert status == "invalid" and "does not identify requested task" in man["error"]


def test_finalize_accepts_vlmevalkit_long_csv_with_dataset_metadata(tmp_path):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="VLMEvalKit", task="MMVP", run_id="r1", model_path=m, harness_dir=tmp_path, tokenizer_path=t,
          chat_template=None, model_args="", gen_kwargs="", thinking=False)
    (out / "derived_acc.csv").write_text("dataset,metric,value\nMMVP,overall,0.6833\n")
    log = tmp_path / "job.out"
    log.write_text("")
    status, _ = finalize(out / "run_meta.json", [log], out, harness_rc=0)
    assert status == "ok"


@pytest.mark.parametrize("filename,contents", [
    ("POPE_acc.csv", "Category,Accuracy (%)\nOverall,50.0\n"),
    ("model_MMVP_acc.csv", "dataset,Accuracy (%)\nPOPE,50.0\n"),
    ("model_POPE_acc.csv", "dataset,Accuracy (%)\nMMVP,50.0\n"),
    ("MMVP_model_POPE_acc.csv", "Category,Accuracy (%)\nOverall,50.0\n"),
])
def test_finalize_rejects_contradictory_vk_identity_in_task_output_dir(tmp_path, filename, contents):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "MMVP"
    start(out, framework="VLMEvalKit", task="MMVP", run_id="r1", model_path=m, harness_dir=tmp_path,
          tokenizer_path=t, chat_template=None, model_args="", gen_kwargs="", thinking=False)
    (out / filename).write_text(contents)
    status, _ = finalize(out / "run_meta.json", [], out, harness_rc=0)
    assert status == "invalid"


@pytest.mark.parametrize("column", ["", "Unnamed: 0", "index", "sample_count", "n_samples"])
def test_finalize_rejects_vk_index_or_counts_without_scores(tmp_path, column):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="VLMEvalKit", task="MMVP", run_id="r1", model_path=m, harness_dir=tmp_path,
          tokenizer_path=t, chat_template=None, model_args="", gen_kwargs="", thinking=False)
    (out / "model_MMVP_acc.csv").write_text(f"{column},Category,Accuracy (%)\n0,Overall,\n")
    status, man = finalize(out / "run_meta.json", [], out, harness_rc=0)
    assert status == "invalid" and "numeric CSV metrics" in man["error"]


@pytest.mark.parametrize("metric", ["samples", "sample_count", "index", "", None])
def test_finalize_rejects_vk_long_csv_without_a_score_metric(tmp_path, metric):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="VLMEvalKit", task="MMVP", run_id="r1", model_path=m, harness_dir=tmp_path,
          tokenizer_path=t, chat_template=None, model_args="", gen_kwargs="", thinking=False)
    contents = f"metric,value\n{metric},300\n" if metric is not None else "value,metric\n300\n"
    (out / "MMVP_acc.csv").write_text(contents)
    status, man = finalize(out / "run_meta.json", [], out, harness_rc=0)
    assert status == "invalid" and "numeric CSV metrics" in man["error"]


@pytest.mark.parametrize("filename", [
    "my_model_MathVista_MINI_acc.csv", "my_model_MathVista_MINI_gpt-4o_score.csv",
    "my_model_MathVista_MINI_acme_customjudge_v9_score.csv",
])
def test_finalize_accepts_vk_model_prefixed_dataset_filename(tmp_path, filename):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="VLMEvalKit", task="MathVista_MINI", run_id="r1", model_path=m, harness_dir=tmp_path,
          tokenizer_path=t, chat_template=None, model_args="", gen_kwargs="", thinking=False)
    (out / filename).write_text("Category,Accuracy (%)\nOverall,50.0\n")
    status, _ = finalize(out / "run_meta.json", [], out, harness_rc=0)
    assert status == "ok"


def test_finalize_finds_lm_eval_results(tmp_path):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="lm-eval", task="gsm8k", run_id="r1", model_path=m, harness_dir=tmp_path, tokenizer_path=t,
          chat_template=None, model_args="", gen_kwargs="", thinking=False)
    sub = out / "model"
    sub.mkdir()
    (sub / "results_2026-09-08T00-00-00.json").write_text(json.dumps({"results": {"gsm8k": {"exact_match,flexible-extract": 0.81}}}))
    log = tmp_path / "job.out"
    log.write_text("")
    status, man = finalize(out / "run_meta.json", [log], out, harness_rc=0)
    assert status == "ok" and man["results"]["file"].endswith("results_2026-09-08T00-00-00.json")
