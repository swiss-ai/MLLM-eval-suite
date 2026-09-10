import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("framework", ["lm-eval", "lmms-eval", "VLMEvalKit"])
def test_submission_drops_parent_spank_options_and_preserves_eval_settings(tmp_path, framework):
    repo = tmp_path / "repo"
    repo.mkdir()
    for directory in ["launchers", "slurm", "suite", "task_suites"]:
        (repo / directory).symlink_to(REPO / directory, target_is_directory=True)
    model = tmp_path / "model"
    model.mkdir()
    edf = tmp_path / "eval environment.toml"
    edf.write_text('image = "/images/eval.sqsh"\n')
    capture = tmp_path / "submission.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "keys = ['_SLURM_SPANK_OPTION_pyxis_environment', 'SLURM_SPANK_TEST', "
        "'SLURM_JOB_ID', 'SLURM_CONF', 'HF_HOME', 'GEN_KWARGS']\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({"
        "'env': {k: os.environ.get(k) for k in keys}, 'argv': sys.argv[1:]}))\n"
        "print('Submitted batch job 123')\n"
    )
    sbatch.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{os.defpath}",
        "ORCH_REPO_ROOT": str(repo),
        "EVAL_ENVIRONMENT": str(edf),
        "CAPTURE": str(capture),
        "SKIP_PREFLIGHT": "1",
        "PREFETCH_EMU35_VISION_TOKENIZER": "false",
        "WANDB_API_KEY": "test-key",
        "HF_TOKEN": "test-token",
        "RUN_ID": "submission-test",
        "_SLURM_SPANK_OPTION_pyxis_environment": "/parent/container.toml",
        "SLURM_SPANK_TEST": "parent-option",
        "SLURM_JOB_ID": "42",
        "SLURM_CONF": "/cluster/slurm.conf",
        "HF_HOME": str(tmp_path / "hf"),
        "GEN_KWARGS": "max_new_tokens=16384,temperature=0",
    }
    args = {
        "lm-eval": [str(model), "--tasks", "gsm8k"],
        "lmms-eval": [str(model), "--tasks", "pope"],
        "VLMEvalKit": ["--model", "test-model", "--tasks", "MMVP"],
    }[framework]
    result = subprocess.run(
        ["bash", str(repo / "launchers" / framework / "eval.sh"), *args],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    submitted = json.loads(capture.read_text())
    assert submitted["env"]["_SLURM_SPANK_OPTION_pyxis_environment"] is None
    assert submitted["env"]["SLURM_SPANK_TEST"] is None
    for key in ["SLURM_JOB_ID", "SLURM_CONF", "HF_HOME", "GEN_KWARGS"]:
        assert submitted["env"][key] == env[key]
    assert f"--environment={edf}" in submitted["argv"]
