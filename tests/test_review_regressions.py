"""Exercise launch and publication boundaries that previously accepted bad evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from suite.coverage import Manifests
from suite.manifest import finalize, start

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import make_dashboard as dashboard


def launch_env(tmp_path):
    harness = tmp_path / "harness/lmms_eval/tasks"
    harness.mkdir(parents=True)
    for task in ("gqa", "frieda"):
        (harness / f"{task}.yaml").write_text(f"task: {task}\n")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "chat_template.jinja").write_text("{{ messages }}")
    vq = tmp_path / "models/BAAI/Emu3.5-VisionTokenizer"
    vq.mkdir(parents=True)
    for name in ("config.yaml", "model.ckpt"):
        (vq / name).write_text("fixture")
    image = tmp_path / "image.sqsh"
    image.write_text("fixture")
    edf = tmp_path / "image.toml"
    edf.write_text(f'image = "{image}"\n')
    # Only remote authentication is replaced; the real launcher/preflight run.
    (tmp_path / "huggingface_hub.py").write_text("def auth_check(*args, **kwargs): pass\n")
    env = dict(os.environ, ORCH_REPO_ROOT=str(REPO), PYTHONPATH=str(tmp_path),
               TOKENIZER_PATH=str(tokenizer), LMMS_EVAL_DEV_PATH=str(harness.parents[1]),
               EVAL_ENVIRONMENT=str(edf), LMMS_EVAL_MODELS_CACHE=str(tmp_path / "models"),
               RS_DATASETS_ROOT=str(tmp_path / "datasets"), HF_HOME=str(tmp_path / "hf"),
               LOG_DIR=str(tmp_path / "logs"), OUTPUT_PATH=str(tmp_path / "outputs"),
               CACHE_BASE=str(tmp_path / "cache"), SKIP_PREFLIGHT="0")
    for key in ("FRIEDA_DIR", "VRSBENCH_DIR", "GEOBENCH_DIR", "BIGEARTH_S2_DIR"):
        env.pop(key, None)
    return env


def launch(model, task, env):
    return subprocess.run(["bash", str(REPO / "launchers/lmms-eval/eval.sh"), str(model),
                           "--tasks", task, "--dry-run"], env=env, capture_output=True, text=True)


def test_foreign_hf_model_passes_launcher_preflight(tmp_path):
    env = launch_env(tmp_path)
    env["MODEL_BACKEND"] = "qwen3_vl"
    p = launch("Qwen/Qwen3-VL-8B-Instruct", "gqa", env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "dry-run:" in p.stdout


def test_staged_default_asset_path_passes_launcher_preflight(tmp_path):
    env = launch_env(tmp_path)
    env["MODEL_BACKEND"] = "apertus_1p5_vllm"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    assets = tmp_path / "datasets/frieda/images"
    assets.mkdir(parents=True)
    for i in range(1000):
        (assets / str(i)).touch()
    p = launch(model, "frieda", env)
    assert p.returncode == 0, p.stdout + p.stderr
    env["FRIEDA_DIR"] = str(tmp_path / "explicit-missing")
    p = launch(model, "frieda", env)
    assert p.returncode == 2 and "explicit-missing" in p.stdout


def begin(out, framework, task):
    return start(out, framework=framework, task=task, run_id="current", model_path=out.parent,
                 harness_dir=out.parent, tokenizer_path=None, chat_template=None,
                 model_args="", gen_kwargs="", thinking=False)


@pytest.mark.parametrize("framework,task,name,old_name,contents,old_contents", [
    ("lmms-eval", "gqa", "new_results.json", "old_results.json",
     '{"results":{"gqa":{"exact_match,none":0.5}}}',
     '{"results":{"chartqa":{"relaxed_overall,none":0.99}}}'),
    ("lm-eval", "gsm8k", "results_new.json", "results_old.json",
     '{"results":{"gsm8k":{"exact_match,flexible-extract":0.5}},"chat_template":"x"}',
     '{"results":{"gsm8k":{"exact_match,flexible-extract":0.99}},"chat_template":"x"}'),
    ("VLMEvalKit", "MMVP", "model_MMVP_acc.csv", "old_MMVP_acc.csv",
     'Category,Accuracy\nOverall,50\n', 'Category,Accuracy\nOverall,99\n'),
])
def test_only_finalized_unchanged_artifacts_receive_success_provenance(
        tmp_path, framework, task, name, old_name, contents, old_contents):
    old = tmp_path / old_name
    old.write_text(old_contents)
    begin(tmp_path, framework, task)
    fresh = tmp_path / name
    fresh.write_text(contents)
    status, _ = finalize(tmp_path / "run_meta.json", [], tmp_path, 0)
    assert status == "ok"
    manifests = Manifests([tmp_path])
    assert manifests.for_result(fresh)["status"] == "ok"
    assert manifests.for_result(old)["status"] != "ok"
    fresh.write_text(old_contents)
    assert manifests.for_result(fresh)["status"] != "ok"


def test_stale_other_task_is_excluded_from_dashboard(tmp_path):
    out = tmp_path / "model/run/gqa"
    out.mkdir(parents=True)
    (out / "old_results.json").write_text(json.dumps({"results": {"chartqa": {"relaxed_overall,none": .99}}}))
    begin(out, "lmms-eval", "gqa")
    (out / "new_results.json").write_text(json.dumps({"results": {"gqa": {"exact_match,none": .5}}}))
    assert finalize(out / "run_meta.json", [], out, 0)[0] == "ok"
    _, rows = dashboard.collect(tmp_path, None, Manifests([tmp_path]))
    assert [(r["task"], r["cells"]["model"]["v"]) for r in rows] == [("gqa", 50)]


@pytest.mark.parametrize("requested", ["gqa, chartqa", "gqa,chartqa,gqa"])
def test_csv_tasks_validate_every_task_and_count_each_sample_once(tmp_path, requested):
    begin(tmp_path, "lmms-eval", requested)
    (tmp_path / "new_results.json").write_text(json.dumps({
        "results": {"gqa": {"exact_match": .5}, "chartqa": {"relaxed_overall": .6}},
        "n-samples": {"gqa": {"original": 2, "effective": 2}, "chartqa": {"original": 3, "effective": 3}},
    }))
    status, man = finalize(tmp_path / "run_meta.json", [], tmp_path, 0)
    assert status == "ok", man["error"]
    assert man["results"]["result_samples"] == 5


def test_csv_request_requires_every_task(tmp_path):
    begin(tmp_path, "lmms-eval", "gqa,chartqa")
    (tmp_path / "new_results.json").write_text('{"results":{"gqa":{"exact_match":0.5}}}')
    status, man = finalize(tmp_path / "run_meta.json", [], tmp_path, 0)
    assert status == "invalid" and "chartqa" in man["error"]


def test_csv_request_rejects_a_task_with_zero_samples(tmp_path):
    begin(tmp_path, "lmms-eval", "gqa,chartqa")
    (tmp_path / "new_results.json").write_text(json.dumps({
        "results": {"gqa": {"acc": .5}, "chartqa": {"acc": .5}},
        "n-samples": {"gqa": {"effective": 2}, "chartqa": {"effective": 0}},
    }))
    status, man = finalize(tmp_path / "run_meta.json", [], tmp_path, 0)
    assert status == "invalid" and "chartqa" in man["error"]


@pytest.mark.parametrize("task", ["gqa,gqa", " gqa, "])
def test_single_task_csv_is_normalized(tmp_path, task):
    begin(tmp_path, "lmms-eval", task)
    (tmp_path / "new_results.json").write_text('{"results":{"gqa":{"exact_match":0.5}}}')
    status, man = finalize(tmp_path / "run_meta.json", [], tmp_path, 0)
    assert status == "ok", man["error"]


def test_csv_request_includes_group_leaves_without_double_counting(tmp_path):
    begin(tmp_path, "lmms-eval", "mmlu,mmlu_a,gqa")
    (tmp_path / "new_results.json").write_text(json.dumps({
        "results": {"mmlu_a": {"acc": .5}, "mmlu_b": {"acc": .6}, "gqa": {"acc": .7}},
        "group_subtasks": {"mmlu": ["mmlu_a", "mmlu_b"]},
        "n-samples": {"mmlu_a": 2, "mmlu_b": 3, "gqa": 4},
    }))
    status, man = finalize(tmp_path / "run_meta.json", [], tmp_path, 0)
    assert status == "ok", man["error"]
    assert man["results"]["result_samples"] == 9


def test_vlmeval_csv_request_requires_fresh_result_for_each_dataset(tmp_path):
    (tmp_path / "model_POPE_acc.csv").write_text('Category,Accuracy\nOverall,99\n')
    begin(tmp_path, "VLMEvalKit", "MMVP,POPE")
    (tmp_path / "model_MMVP_acc.csv").write_text('Category,Accuracy\nOverall,50\n')
    assert finalize(tmp_path / "run_meta.json", [], tmp_path, 0)[0] == "invalid"
    (tmp_path / "model_POPE_acc.csv").write_text('Category,Accuracy\nOverall,60\n')
    assert finalize(tmp_path / "run_meta.json", [], tmp_path, 0)[0] == "ok"
    manifests = Manifests([tmp_path])
    assert manifests.for_result(tmp_path / "model_POPE_acc.csv")["status"] == "ok"
    assert manifests.for_result(tmp_path / "model_MMVP_acc.csv")["status"] == "ok"
