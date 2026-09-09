"""Only selected Apertus runs stage the vision tokenizer before submission."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from test_review_regressions import REPO, launch_env


def staging_env(tmp_path):
    env = launch_env(tmp_path)
    for path in (tmp_path / "models/BAAI/Emu3.5-VisionTokenizer").iterdir():
        path.unlink()
    (tmp_path / "huggingface_hub.py").write_text('''import json, os
from pathlib import Path
def auth_check(*args, **kwargs): pass
def snapshot_download(repo_id, local_dir, allow_patterns):
    with open(os.environ['STAGING_LOG'], 'a') as log:
        log.write(json.dumps([repo_id, str(local_dir), allow_patterns]) + '\\n')
    destination = Path(local_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in allow_patterns:
        (destination / name).write_text('fixture')
''')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scheduler = bin_dir / "sbatch"
    scheduler.write_text('#!/bin/sh\nprintf "submitted\\n" >> "$SUBMIT_LOG"\necho "Submitted batch job 123"\n')
    scheduler.chmod(0o755)
    env.update(PATH=f"{bin_dir}:{env['PATH']}", STAGING_LOG=str(tmp_path / "staging.jsonl"),
               SUBMIT_LOG=str(tmp_path / "submitted"), PREFETCH_EMU35_VISION_TOKENIZER="true",
               RUNTIME_CACHE=str(tmp_path), WORK_BASE=str(tmp_path / "vk-results"),
               VLMEVAL_RESPONSE_CACHE=str(tmp_path / "vk-cache"), LMUData=str(tmp_path / "datasets"))
    return env


def run_launcher(args, env, framework=None):
    path = REPO / "launchers" / framework / "eval.sh" if framework else REPO / "launchers/eval.sh"
    return subprocess.run(["bash", str(path), *args], env=env, text=True, capture_output=True)


def test_combined_dry_run_never_downloads(tmp_path):
    env = staging_env(tmp_path)
    run = run_launcher(["--model", "/model/to/inspect", "--suite", "smoke", "--dry-run"], env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[all] -> lmms-eval" in run.stdout and "[all] -> VLMEvalKit" in run.stdout
    assert not Path(env["STAGING_LOG"]).exists()
    assert not Path(env["SUBMIT_LOG"]).exists()


@pytest.mark.parametrize("framework", ["lmms-eval", "VLMEvalKit"])
def test_foreign_dispatch_never_stages_apertus_assets(tmp_path, framework):
    env = staging_env(tmp_path)
    env.update(MODEL_BACKEND="qwen3_vl", SKIP_PREFLIGHT="1")
    run = run_launcher(["--eval-framework", framework, "--model", "Qwen/Qwen3-VL-8B-Instruct",
                        "--tasks", "gqa" if framework == "lmms-eval" else "MMVP"], env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert not Path(env["STAGING_LOG"]).exists()
    assert Path(env["SUBMIT_LOG"]).read_text() == "submitted\n"


@pytest.mark.parametrize("framework", ["lmms-eval", "VLMEvalKit"])
@pytest.mark.parametrize("runtime_cache_override", [False, True])
def test_native_launcher_stages_once_before_submitting(tmp_path, framework, runtime_cache_override):
    env = staging_env(tmp_path)
    models_cache = tmp_path / "runtime-models" if runtime_cache_override else tmp_path / "models"
    if runtime_cache_override:
        env["VLLM_APERTUS_MODELS_CACHE"] = str(models_cache)
    model = tmp_path / "Apertus-model"
    model.mkdir()
    (model / "config.json").write_text('{"architectures":["ApertusForCausalLM"]}')
    (model / "model.safetensors").write_text("weights")
    if framework == "lmms-eval":
        args = [str(model), "--tasks", "gqa"]
        env["MODEL_BACKEND"] = "apertus_1p5_vllm"
    else:
        args = ["--model", "Apertus-model", "--tasks", "MMVP"]
        env.update(APERTUS_MODEL_PATH=str(model), APERTUS_TOKENIZER_PATH=env["TOKENIZER_PATH"])
    for _ in range(2):
        run = run_launcher(args, env, framework)
        assert run.returncode == 0, run.stdout + run.stderr
    records = [json.loads(line) for line in Path(env["STAGING_LOG"]).read_text().splitlines()]
    assert records == [["BAAI/Emu3.5-VisionTokenizer", str(models_cache / "BAAI/Emu3.5-VisionTokenizer"),
                        ["config.yaml", "model.ckpt"]]]
    assert Path(env["SUBMIT_LOG"]).read_text() == "submitted\nsubmitted\n"


def test_native_dry_run_reports_missing_assets_without_download(tmp_path):
    env = staging_env(tmp_path)
    model = tmp_path / "Apertus-model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_text("weights")
    env["MODEL_BACKEND"] = "apertus_1p5_vllm"
    run = run_launcher([str(model), "--tasks", "gqa", "--dry-run"], env, "lmms-eval")
    assert run.returncode == 2
    assert "vision_tokenizer" in run.stdout
    assert not Path(env["STAGING_LOG"]).exists()
    assert not Path(env["SUBMIT_LOG"]).exists()
