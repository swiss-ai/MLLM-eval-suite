"""Exercise the text launcher's scheduler and harness boundaries without inference."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch_env(tmp_path):
    root = tmp_path / "suite"
    for directory in ("launchers/lm-eval", "slurm/shared", "slurm/lm-eval", "suite", "task_suites/lm-eval"):
        shutil.copytree(REPO / directory, root / directory)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors"):
        (model / name).write_text("{}")
    image = tmp_path / "test.sqsh"
    image.write_bytes(b"fixture")
    edf = tmp_path / "test.toml"
    edf.write_text(f'image = "{image}"\n')
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "sbatch").write_text(f"#!{sys.executable}\n" + '''import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['LAUNCH_CAPTURE'], 'a') as fh:
    fh.write(json.dumps({'args': args, 'annotators': os.environ.get('ALPACA_EVAL_ANNOTATORS_CONFIG')}) + '\\n')
if os.environ.get('EXECUTE_JOB') == '1':
    job = next(i for i, arg in enumerate(args) if arg.endswith('.slurm'))
    sys.exit(subprocess.run(['bash', *args[job:]]).returncode)
''')
    (bindir / "sbatch").chmod(0o755)
    # Replace inference only; the real launcher, job script, preflight and
    # manifest creation/finalization still run. No candidate Python is executed.
    (bindir / "python").write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[:2] == ['-m', 'apertus_lm_eval_ext']:
    Path(os.environ['HARNESS_CAPTURE']).write_text(json.dumps({
        'args': args, 'code_eval': os.environ.get('HF_ALLOW_CODE_EVAL'),
        'annotators': os.environ.get('ALPACA_EVAL_ANNOTATORS_CONFIG')}))
    out = Path(args[args.index('--output_path') + 1])
    task = args[args.index('--tasks') + 1]
    (out / 'results_fixture.json').write_text(json.dumps({'results': {task: {'acc,none': 0.5}}}))
else:
    os.execv(sys.executable, [sys.executable, *args])
''')
    (bindir / "python").chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('SLURM_', 'SBATCH_')) and key not in (
               'HF_ALLOW_CODE_EVAL', 'ALPACA_EVAL_ANNOTATORS_CONFIG', 'OPENAI_API_KEY', 'SKIP_PREFLIGHT',
               'LM_EVAL_CHAT_TEMPLATE')}
    env.update(ORCH_REPO_ROOT=str(root), EVAL_ENVIRONMENT=str(edf), EVAL_RESERVATION="",
               PATH=f"{bindir}:{os.environ['PATH']}", RUN_ID="test", SUITE_PY=sys.executable,
               LAUNCH_CAPTURE=str(tmp_path / "scheduler.jsonl"), HARNESS_CAPTURE=str(tmp_path / "harness.json"),
               LM_EVAL_DEV_PATH=str(tmp_path / "harness"), OUTPUT_BASE=str(tmp_path / "outputs"))
    return root, model, env


def launch(launch_env, *args, execute=False, **environment):
    root, model, env = launch_env
    env = dict(env, EXECUTE_JOB=str(int(execute)), **environment)
    result = subprocess.run(['bash', str(root / 'launchers/lm-eval/eval.sh'), str(model), *args],
                            cwd=root, env=env, text=True, capture_output=True)
    return result, env


def captured_run(env):
    captured = json.loads(Path(env['HARNESS_CAPTURE']).read_text())
    args = captured['args']
    out = Path(args[args.index('--output_path') + 1])
    return captured, json.loads((out / 'run_meta.json').read_text())


@pytest.mark.parametrize('task', ['humaneval_instruct', 'mbpp_instruct'])
def test_code_opt_in_reaches_harness_and_execution_environment(launch_env, task):
    process, env = launch(launch_env, '--tasks', task, '--confirm-run-unsafe-code', execute=True)
    assert process.returncode == 0, process.stdout + process.stderr
    captured, manifest = captured_run(env)
    assert '--confirm_run_unsafe_code' in captured['args']
    assert captured['code_eval'] == '1'
    assert '--apply_chat_template' in captured['args']
    assert manifest['generation']['apply_chat_template'] == 1


def test_default_launch_does_not_enable_code_execution(launch_env):
    process, env = launch(launch_env, '--suite', 'text-smoke', execute=True)
    assert process.returncode == 0, process.stdout + process.stderr
    captured, manifest = captured_run(env)
    assert '--confirm_run_unsafe_code' not in captured['args']
    assert captured['code_eval'] is None
    assert captured['args'][captured['args'].index('--tasks') + 1] == 'arc_easy'
    assert captured['args'].count('--apply_chat_template') == 1
    assert manifest['generation']['apply_chat_template'] == 1
    assert manifest['tokenizer']['chat_template_sha256']


@pytest.mark.parametrize('override, expected', [('0', 0), ('false', 0), ('1', 1), ('true', 1)])
def test_chat_template_environment_override_matches_harness_and_manifest(launch_env, override, expected):
    process, env = launch(launch_env, '--tasks', 'math500_verify', execute=True,
                          LM_EVAL_CHAT_TEMPLATE=override)
    assert process.returncode == 0, process.stdout + process.stderr
    captured, manifest = captured_run(env)
    assert captured['args'].count('--apply_chat_template') == expected
    assert manifest['generation']['apply_chat_template'] == expected


@pytest.mark.parametrize('environment, expected', [({}, 0), ({'LM_EVAL_CHAT_TEMPLATE': '1'}, 1)])
def test_per_task_chat_template_override_is_used_by_launcher(launch_env, environment, expected):
    registry = launch_env[0] / 'suite/tasks.toml'
    registry.write_text(registry.read_text().replace('[tasks.gsm8k]\n', '[tasks.gsm8k]\nchat_template = false\n'))
    process, env = launch(launch_env, '--tasks', 'gsm8k', execute=True, **environment)
    assert process.returncode == 0, process.stdout + process.stderr
    captured, manifest = captured_run(env)
    assert captured['args'].count('--apply_chat_template') == expected
    assert manifest['generation']['apply_chat_template'] == expected


def test_explicit_job_chat_template_still_overrides_base_model_default(launch_env):
    process, env = launch(launch_env, '--tasks', 'mmlu_pro', '--', '--apply-chat-template', execute=True,
                          LM_EVAL_CHAT_TEMPLATE='0')
    assert process.returncode == 0, process.stdout + process.stderr
    captured, manifest = captured_run(env)
    assert captured['args'].count('--apply_chat_template') == 1
    assert manifest['generation']['apply_chat_template'] == 1


def test_job_passthrough_code_opt_in_and_limit(launch_env):
    process, env = launch(launch_env, '--tasks', 'humaneval_instruct', '--',
                          '--confirm-run-unsafe-code', '--limit', '2', execute=True)
    assert process.returncode == 0, process.stdout + process.stderr
    captured = json.loads(Path(env['HARNESS_CAPTURE']).read_text())
    assert '--confirm_run_unsafe_code' in captured['args']
    assert captured['args'][captured['args'].index('--limit') + 1] == '2'
    assert captured['code_eval'] == '1'


def test_alpaca_configuration_reaches_job_without_openai_key(launch_env):
    process, env = launch(launch_env, '--tasks', 'alpaca_eval', execute=True,
                          ALPACA_EVAL_ANNOTATORS_CONFIG='local_test_judge')
    assert process.returncode == 0, process.stdout + process.stderr
    captured = json.loads(Path(env['HARNESS_CAPTURE']).read_text())
    assert captured['annotators'] == 'local_test_judge'


def test_alpaca_missing_configuration_stops_before_scheduler(launch_env):
    process, env = launch(launch_env, '--tasks', 'alpaca_eval')
    assert process.returncode == 2
    assert 'ALPACA_EVAL_ANNOTATORS_CONFIG' in process.stdout
    assert not Path(env['LAUNCH_CAPTURE']).exists()


def test_requested_suite_resolves_all_tasks_before_submission(launch_env):
    process, env = launch(launch_env, '--suite', 'text-requested', '--confirm-run-unsafe-code',
                          ALPACA_EVAL_ANNOTATORS_CONFIG='local_test_judge')
    assert process.returncode == 0, process.stdout + process.stderr
    records = [json.loads(line) for line in Path(env['LAUNCH_CAPTURE']).read_text().splitlines()]
    tasks = [row['args'][row['args'].index('--tasks') + 1] for row in records]
    assert len(tasks) == len(set(tasks)) == 31
    assert tasks[:3] == ['mmlu_flan_cot_zeroshot', 'mmlu_pro', 'truthfulqa_mc2']
    assert tasks[-3:] == ['bbq', 'toxigen', 'wmdp']
    assert all(row['args'].count('--apply-chat-template') == 1 for row in records)
