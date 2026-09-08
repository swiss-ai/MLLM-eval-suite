import json

import pytest

from suite.manifest import EXIT_FAILED, finalize, output_token_stats, start


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
    man = start(out, framework="lmms-eval", task="gqa", run_id="r1", model_path=m, tokenizer_path=t,
                chat_template=t / "chat_template.jinja", model_args="model=x,enable_thinking=True",
                gen_kwargs="max_new_tokens=32768", thinking=True, extra={"slurm": {"job_id": "1"}})
    on_disk = json.loads((out / "run_meta.json").read_text())
    assert on_disk == man and man["schema"] == 1 and man["status"] == "running"
    assert man["model"]["config_sha256"] and man["model"]["weights"]["shards"][0]["size"] == 16
    assert man["tokenizer"]["chat_template_sha256"] and man["thinking"]["requested"] is True
    assert man["slurm"]["job_id"] == "1"


def test_output_token_stats(tmp_path):
    s = tmp_path / "samples_x.jsonl"
    s.write_text("\n".join(json.dumps({"token_counts": [{"output_tokens": n}]}) for n in (2, 4, 600)) + "\n")
    st = output_token_stats([s])
    assert st["n"] == 3 and st["mean"] == pytest.approx(202) and st["max"] == 600


def _run(tmp_path, thinking, tokens, log_text="", results=True, rc=0):
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    start(out, framework="lmms-eval", task="mmvp", run_id="r1", model_path=m, tokenizer_path=t,
          chat_template=None, model_args="", gen_kwargs="", thinking=thinking, extra={})
    sub = out / "textview__m"
    sub.mkdir()
    if results:
        (sub / "20260908_x_results.json").write_text(json.dumps({"results": {"mmvp": {"mmvp_accuracy,none": 0.7}}}))
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
    assert status == "invalid" and man["thinking"]["effective"] is None and "output tokens" in man["error"]


def test_finalize_failed_on_harness_error(tmp_path):
    status, man = _run(tmp_path, thinking=False, tokens=[], log_text="Error during evaluation: boom\n", results=False)
    assert status == "failed" and "boom" in man["error"]


def test_finalize_failed_when_no_results(tmp_path):
    status, _ = _run(tmp_path, thinking=False, tokens=[], results=False)
    assert status == "failed"


def test_cli_exit_codes(tmp_path):
    from suite.manifest import main
    m, t = _model_dir(tmp_path)
    out = tmp_path / "out"
    assert main(["start", "--out", str(out), "--framework", "lmms-eval", "--task", "gqa", "--run-id", "r",
                 "--model", str(m), "--tokenizer", str(t)]) == 0
    log = tmp_path / "l.out"
    log.write_text("Error during evaluation: x\n")
    assert main(["finalize", "--manifest", str(out / "run_meta.json"), "--log", str(log),
                 "--results-dir", str(out), "--harness-rc", "0"]) == EXIT_FAILED
