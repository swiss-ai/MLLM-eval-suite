"""Image entrypoints share immutable build inputs and never rotate production."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def build_environment(tmp_path):
    recipe = tmp_path / "suite/dockerfiles"
    shutil.copytree(REPO / "dockerfiles", recipe)
    output = tmp_path / "candidate"
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("podman", "enroot", "uname", "unsquashfs"):
        path = tools / name
        path.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ['BUILD_COMMANDS'], 'a') as log:
    log.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
if name == 'uname':
    print('aarch64')
elif name == 'podman' and sys.argv[1:3] == ['image', 'inspect']:
    print('{}')
elif name == 'enroot':
    flag = '--output' if '--output' in sys.argv else '-o'
    Path(sys.argv[sys.argv.index(flag) + 1]).write_bytes(b'candidate-image')
    sys.exit(int(os.environ.get('EXPORT_STATUS', '0')))
elif name == 'unsquashfs':
    print('test build stamp')
''')
        path.chmod(0o755)
    library = tmp_path / "enroot-library"
    library.mkdir()
    (library / "docker.sh").write_text("# fixture library\n")
    job_id = "cpu-test-" + uuid.uuid4().hex
    env = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", SLURM_JOB_ID=job_id,
               BUILD_COMMANDS=str(tmp_path / "commands.jsonl"), ENROOT_LIBRARY_PATH=str(library),
               UV_CACHE_DIR=str(tmp_path / "wheel-cache"))
    for key in ("APERTUS_RECIPE_DIR", "APERTUS_BUILD_SNAPSHOT", "APERTUS_APT_CONFIG_DIR"):
        env.pop(key, None)
    # The unmodified builder owns this unique store; no container command is real.
    key = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    store = Path(f"/dev/shm/apertus-build-{os.getuid()}-{job_id}-{key}")
    try:
        yield recipe, output, env
    finally:
        if store.exists():
            shutil.rmtree(store)


def invoke(recipe, output, env, entrypoint="generate_docker.sh"):
    return subprocess.run(["bash", str(recipe / entrypoint), str(output)],
                          env=env, capture_output=True, text=True)


def test_legacy_entrypoint_builds_versioned_candidate_preserving_site_options(build_environment):
    recipe, output, env = build_environment
    production = recipe / "apertus-vllm-vision-eval-prod.sqsh"
    production.write_bytes(b"certified-production")
    process = invoke(recipe, output, env)
    assert process.returncode == 0, process.stdout + process.stderr
    assert production.read_bytes() == b"certified-production"
    images = list(output.glob("*.sqsh"))
    assert len(images) == 1 and images[0].read_bytes() == b"candidate-image"
    assert images[0].with_suffix(".sqsh.sha256").is_file()
    context, = output.glob("context.*")
    commands = [json.loads(line) for line in Path(env["BUILD_COMMANDS"]).read_text().splitlines()]
    build, = [line for line in commands if line[:2] == ["podman", "build"]]
    assert build[-1] == str(context)
    assert f"{env['UV_CACHE_DIR']}:/root/.cache/uv:rw" in build
    for name, destination in (("empty.sources.list", "/etc/apt/sources.list"),
                              ("my-sources.d", "/etc/apt/sources.list.d"),
                              ("99-jfrog-proxy", "/etc/apt/apt.conf.d/99-jfrog-proxy")):
        assert f"{context}/apt/{name}:{destination}:ro,z" in build
    hashes = (output / "source-sha256.txt").read_text()
    assert "./apt/99-jfrog-proxy" in hashes and "./apt/my-sources.d/ubuntu.sources" in hashes
    # A published candidate is never overwritten by retrying either entrypoint.
    for entrypoint in ("generate_docker.sh", "build_trial_image.sh"):
        retry = invoke(recipe, output, env, entrypoint)
        assert retry.returncode != 0 and "Refusing to overwrite" in retry.stderr
        assert images[0].read_bytes() == b"candidate-image"


def test_export_failure_never_publishes_candidate_or_replaces_production(build_environment):
    recipe, output, env = build_environment
    production = recipe / "apertus-vllm-vision-eval-prod.sqsh"
    production.write_bytes(b"certified-production")
    env["EXPORT_STATUS"] = "17"
    process = invoke(recipe, output, env)
    assert process.returncode == 17, process.stdout + process.stderr
    assert not list(output.glob("apertus-vllm-*.sqsh"))
    assert production.read_bytes() == b"certified-production"
